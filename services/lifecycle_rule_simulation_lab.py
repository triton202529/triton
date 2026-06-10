"""
TRITON Phase 169 — Lifecycle Rule Simulation Lab (replay / what-if only).

Replays the historical decision timeline under the Phase 168 recommendations
to answer: would they have improved outcomes? Six scenarios are simulated
against actual price history — no production rule, threshold, signal,
execution, or broker behavior is touched, and no orders are written.

Scenarios:
    S1 EARLIER_EXIT_5D            exit LATE_EXITs 5 trading days earlier
    S2 EARLIER_EXIT_PEAK_DRAWDOWN exit when half of a 5%+ run-up is given back
    S3 MID_CONFIDENCE_FILTER      block decisions with confidence 0.60-0.80
    S4 HOLD_ADD_RECONFIRMATION    block HOLD→ADD without fresh improvement
    S5 RISK_FLAG_ACCELERATION     exit/review at flagged events before losses
    S6 COMBINED_OPTIMIZED_RULESET best safe combination of the above

Outputs (data/results/):
    lifecycle_rule_simulation_results.json
    lifecycle_rule_simulation_scenarios.csv
    lifecycle_rule_simulation_comparison.csv

Run:
    python -m services.lifecycle_rule_simulation_lab
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

from services.pnl_ledger_builder import _load_price_series, _safe_read_csv, _to_float
from services.ticker_timeline_builder import build_ticker_timelines

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "lifecycle_rule_simulation_results.json"
OUT_SCENARIOS = RESULTS / "lifecycle_rule_simulation_scenarios.csv"
OUT_COMPARISON = RESULTS / "lifecycle_rule_simulation_comparison.csv"

HELD_ACTIONS = {"BUY", "ADD", "HOLD"}


def _log(msg: str) -> None:
    print(f"[lifecycle_rule_simulation_lab] {msg}", flush=True)


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


def _close_n_days_before(sym: str, ts: pd.Timestamp, n: int) -> Tuple[float, Optional[str]]:
    """Close price n trading rows before the row covering ts."""
    prices = _load_price_series(sym)
    if prices.empty or pd.isna(ts):
        return np.nan, None
    base = pd.Timestamp(ts).tz_localize(None).normalize()
    upto = prices[prices["date"] <= base]
    if len(upto) <= n:
        return np.nan, None
    row = upto.iloc[-1 - n]
    return float(row["close"]), str(pd.Timestamp(row["date"]).date())


def _peak_drawdown_trigger(
    sym: str, ts: pd.Timestamp, actual_px: float
) -> Tuple[float, Optional[str]]:
    """
    Phase 168 P1 rule: once a position has run up 5%+ to a local peak, exit when
    half of that run-up is given back. Returns (simulated exit close, date) or
    (nan, None) when the trigger never fires before the actual exit.
    """
    prices = _load_price_series(sym)
    if prices.empty or pd.isna(ts) or not np.isfinite(actual_px):
        return np.nan, None
    base = pd.Timestamp(ts).tz_localize(None).normalize()
    window = prices[(prices["date"] >= base - pd.Timedelta(days=30)) & (prices["date"] <= base)]
    if len(window) < 3:
        return np.nan, None
    closes = window["close"].to_numpy()
    dates = window["date"].to_numpy()
    peak = closes[0]
    for px, dt in zip(closes[1:], dates[1:]):
        peak = max(peak, px)
        runup = peak / actual_px - 1.0  # run-up relative to the eventual exit price
        if runup >= 0.05:
            trigger = peak * (1.0 - runup / 2.0)
            if px <= trigger:
                return float(px), str(pd.Timestamp(dt).date())
    return np.nan, None


# ──────────────────────────────────────────────
# CARD 4 — Baseline replay
# ──────────────────────────────────────────────
def build_baseline(tl: pd.DataFrame, intel: Dict[str, Any], trades: pd.DataFrame) -> Dict[str, Any]:
    realized = (
        pd.to_numeric(trades.get("realized_pl"), errors="coerce")
        if not trades.empty
        else pd.Series(dtype=float)
    )
    rankings = pd.DataFrame(intel.get("decision_rankings") or [])
    deployed = (
        float(pd.to_numeric(rankings.get("capital_deployed"), errors="coerce").sum())
        if not rankings.empty
        else np.nan
    )
    exits = tl[(tl["action"] == "EXIT") & (tl["exit_outcome"].astype(str).str.len() > 0)]
    aligned = pd.to_numeric(tl["forward_return_5d"], errors="coerce") * tl["action"].map(
        lambda a: 1.0 if str(a) in HELD_ACTIONS else -1.0
    )
    drawdown_proxy = float(aligned[aligned < 0].mean()) if (aligned < 0).any() else 0.0

    return {
        "realized_pnl": _f(realized.sum()),
        "win_rate": _f((realized > 0).mean() * 100.0, 1) if len(realized) else None,
        "trade_count": int(len(realized)),
        "exit_outcomes": exits["exit_outcome"].value_counts().to_dict() if not exits.empty else {},
        "decision_alpha": _f((intel.get("summary") or {}).get("overall_alpha_score"), 1),
        "drawdown_proxy_pct": _f(drawdown_proxy),
        "capital_deployed": _f(deployed),
        "events_replayed": int(len(tl)),
    }


# ──────────────────────────────────────────────
# CARD 6 + S1/S2 — Exit timing simulations
# ──────────────────────────────────────────────
def simulate_exit_timing(tl: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    exits = tl[(tl["action"] == "EXIT") & (tl["exit_outcome"].astype(str).str.len() > 0)].copy()
    detail_rows: List[Dict[str, Any]] = []

    s1_delta = s2_delta = 0.0
    s1_improved = s1_worsened = 0
    s2_improved = s2_worsened = s2_fired = 0

    for _, r in exits.iterrows():
        sym = str(r["ticker"])
        ts = pd.Timestamp(r["timestamp"])
        qty = _to_float(r.get("qty"), np.nan)
        actual_px = _to_float(r.get("price"), np.nan)
        outcome = str(r["exit_outcome"])
        if not np.isfinite(qty) or not np.isfinite(actual_px):
            continue

        # S1 — only LATE exits move 5 days earlier.
        d1 = np.nan
        if outcome == "LATE_EXIT":
            px5, _date5 = _close_n_days_before(sym, ts, 5)
            if np.isfinite(px5):
                d1 = (px5 - actual_px) * qty
                s1_delta += d1
                s1_improved += int(d1 > 0)
                s1_worsened += int(d1 < 0)

        # S2 — peak-drawdown trigger applies to every exit (false positives counted).
        trig_px, trig_date = _peak_drawdown_trigger(sym, ts, actual_px)
        d2 = np.nan
        if np.isfinite(trig_px):
            d2 = (trig_px - actual_px) * qty
            s2_delta += d2
            s2_fired += 1
            s2_improved += int(d2 > 0)
            s2_worsened += int(d2 < 0)

        detail_rows.append(
            {
                "symbol": sym,
                "exit_date": str(ts.date()) if not pd.isna(ts) else "",
                "exit_outcome": outcome,
                "qty": qty,
                "actual_exit_price": _f(actual_px),
                "exit_5d_earlier_delta": _f(d1),
                "peak_drawdown_trigger_date": trig_date or "",
                "peak_drawdown_delta": _f(d2),
                "verdict": (
                    "IMPROVED"
                    if (np.isfinite(d2) and d2 > 0) or (np.isfinite(d1) and d1 > 0)
                    else (
                        "PREMATURE"
                        if (np.isfinite(d2) and d2 < 0) or (np.isfinite(d1) and d1 < 0)
                        else "UNCHANGED"
                    )
                ),
            }
        )

    detail = pd.DataFrame(detail_rows)
    late_n = int((exits["exit_outcome"] == "LATE_EXIT").sum())

    s1 = {
        "scenario": "EARLIER_EXIT_5D",
        "description": "Exit LATE_EXIT positions 5 trading days earlier",
        "simulated_pnl_delta": _f(s1_delta),
        "impacted_trades": late_n,
        "improved": s1_improved,
        "worsened": s1_worsened,
        "false_positive_risk": _f(s1_worsened / late_n * 100.0, 1) if late_n else None,
        "false_negative_risk": _f((len(exits) - late_n) / max(len(exits), 1) * 100.0, 1),
        "capital_protected": _f(max(s1_delta, 0.0)),
        "capital_lost": _f(max(-s1_delta, 0.0)),
        "rule_confidence": _f(min(late_n * 4.0, 80.0), 0),
    }
    s2 = {
        "scenario": "EARLIER_EXIT_PEAK_DRAWDOWN",
        "description": "Exit when half of a 5%+ run-up is given back (applied to all exits)",
        "simulated_pnl_delta": _f(s2_delta),
        "impacted_trades": s2_fired,
        "improved": s2_improved,
        "worsened": s2_worsened,
        "false_positive_risk": _f(s2_worsened / s2_fired * 100.0, 1) if s2_fired else None,
        "false_negative_risk": _f((late_n - s2_improved) / max(late_n, 1) * 100.0, 1),
        "capital_protected": _f(max(s2_delta, 0.0)),
        "capital_lost": _f(max(-s2_delta, 0.0)),
        "rule_confidence": _f(min(s2_fired * 4.0, 85.0), 0),
    }

    # Symbol benefit summary for the exit table.
    by_symbol: List[Dict[str, Any]] = []
    if not detail.empty:
        g = detail.copy()
        g["best_delta"] = g[["exit_5d_earlier_delta", "peak_drawdown_delta"]].max(axis=1)
        agg = (
            g.groupby("symbol", as_index=False)["best_delta"]
            .sum()
            .sort_values("best_delta", ascending=False)
        )
        by_symbol = agg.head(10).to_dict(orient="records")

    exit_block = {
        "late_exit_count": late_n,
        "exits_analyzed": int(len(exits)),
        "best_scenario_delta": _f(max(s1_delta, s2_delta)),
        "symbols_benefiting_most": by_symbol,
        "table": detail.to_dict(orient="records"),
    }
    return s1, s2, detail if not detail.empty else pd.DataFrame()


# ──────────────────────────────────────────────
# CARD 7 / S3 — Confidence filter simulation
# ──────────────────────────────────────────────
def simulate_confidence_filter(tl: pd.DataFrame, intel: Dict[str, Any]) -> Dict[str, Any]:
    df = tl.copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["fwd"] = pd.to_numeric(df["forward_return_5d"], errors="coerce")
    df["aligned"] = df["fwd"] * df["action"].map(lambda a: 1.0 if str(a) in HELD_ACTIONS else -1.0)

    zone = df[(df["confidence"] >= 0.60) & (df["confidence"] < 0.80) & df["aligned"].notna()]
    zone = zone[zone["action"].isin(["BUY", "ADD", "HOLD"])]
    n = int(len(zone))
    missed_winners = int((zone["aligned"] > 0).sum())
    avoided_losers = int((zone["aligned"] < 0).sum())
    mean_aligned = float(zone["aligned"].mean()) if n else np.nan
    # Blocking removes the bucket's mean aligned return: positive impact when mean < 0.
    net_alpha_impact = -mean_aligned if np.isfinite(mean_aligned) else np.nan

    # Dollar approximation: average ADD notional from Phase 167 rankings.
    rankings = pd.DataFrame(intel.get("decision_rankings") or [])
    add_row = rankings[rankings["action"] == "ADD"] if not rankings.empty else pd.DataFrame()
    avg_notional = 0.0
    if not add_row.empty:
        deployed = _to_float(add_row["capital_deployed"].iloc[0], 0.0)
        cnt = max(_to_float(add_row["decision_count"].iloc[0], 1.0), 1.0)
        avg_notional = deployed / cnt
    pnl_delta = -mean_aligned / 100.0 * avg_notional * n if np.isfinite(mean_aligned) else np.nan

    return {
        "scenario": "MID_CONFIDENCE_FILTER",
        "description": "Block/require extra confirmation for entries with confidence 0.60-0.80",
        "simulated_pnl_delta": _f(pnl_delta),
        "impacted_trades": n,
        "missed_winners": missed_winners,
        "avoided_losers": avoided_losers,
        "net_alpha_impact_pct": _f(net_alpha_impact, 3),
        "false_positive_risk": _f(missed_winners / n * 100.0, 1) if n else None,
        "false_negative_risk": None,
        "capital_protected": _f(max(pnl_delta, 0.0)) if np.isfinite(pnl_delta) else None,
        "capital_lost": _f(max(-pnl_delta, 0.0)) if np.isfinite(pnl_delta) else None,
        "rule_confidence": _f(min(n / 4.0, 75.0), 0),
        "method": "dollar impact approximated with the average ADD notional; returns are 5-day alpha-aligned",
    }


# ──────────────────────────────────────────────
# CARD 8 / S4 — HOLD → ADD reconfirmation
# ──────────────────────────────────────────────
def simulate_hold_add(tl: pd.DataFrame, intel: Dict[str, Any]) -> Dict[str, Any]:
    df = (
        tl[tl["action"].isin(["BUY", "ADD", "HOLD", "TRIM", "EXIT"])]
        .sort_values(["ticker", "timestamp"])
        .copy()
    )
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["expected_return"] = pd.to_numeric(df["expected_return"], errors="coerce")
    df["fwd"] = pd.to_numeric(df["forward_return_5d"], errors="coerce")
    df["prev_action"] = df.groupby("ticker")["action"].shift(1)
    df["prev_conf"] = df.groupby("ticker")["confidence"].shift(1)
    df["prev_er"] = df.groupby("ticker")["expected_return"].shift(1)
    df["prev_rationale"] = df.groupby("ticker")["rationale"].shift(1)
    df["prev_flags"] = df.groupby("ticker")["risk_flags"].shift(1)

    adds = df[(df["action"] == "ADD") & (df["prev_action"] == "HOLD")].copy()
    if adds.empty:
        return {
            "scenario": "HOLD_ADD_RECONFIRMATION",
            "description": "Block HOLD→ADD without fresh confidence/expected-return/rationale improvement",
            "simulated_pnl_delta": 0.0,
            "impacted_trades": 0,
            "rule_confidence": 0.0,
        }

    # Reconfirmation = measurable improvement. Rationale text is excluded because
    # it is regenerated daily (prices embedded in the text always differ), which
    # would make every ADD count as "fresh" and the rule untestable.
    conf_improved = adds["confidence"] > adds["prev_conf"] + 0.01
    er_improved = adds["expected_return"] > adds["prev_er"] + 0.05
    flags_reduced = adds["risk_flags"].astype(str).str.len().eq(0) & adds["prev_flags"].astype(
        str
    ).str.len().gt(0)
    reconfirmed = (
        conf_improved.fillna(False) | er_improved.fillna(False) | flags_reduced.fillna(False)
    )
    blocked = adds[~reconfirmed]

    fwd = blocked["fwd"].dropna()
    avoided_losses = int((fwd < 0).sum())
    missed_gains = int((fwd > 0).sum())
    mean_fwd = float(fwd.mean()) if len(fwd) else np.nan

    rankings = pd.DataFrame(intel.get("decision_rankings") or [])
    add_row = rankings[rankings["action"] == "ADD"] if not rankings.empty else pd.DataFrame()
    avg_notional = 0.0
    if not add_row.empty:
        deployed = _to_float(add_row["capital_deployed"].iloc[0], 0.0)
        cnt = max(_to_float(add_row["decision_count"].iloc[0], 1.0), 1.0)
        avg_notional = deployed / cnt
    pnl_delta = -mean_fwd / 100.0 * avg_notional * len(blocked) if np.isfinite(mean_fwd) else np.nan

    return {
        "scenario": "HOLD_ADD_RECONFIRMATION",
        "description": "Block HOLD→ADD without fresh confidence/expected-return/rationale improvement",
        "simulated_pnl_delta": _f(pnl_delta),
        "impacted_trades": int(len(blocked)),
        "blocked_adds": int(len(blocked)),
        "total_hold_add": int(len(adds)),
        "avoided_losses": avoided_losses,
        "missed_gains": missed_gains,
        "false_positive_risk": _f(missed_gains / max(len(fwd), 1) * 100.0, 1),
        "false_negative_risk": None,
        "capital_protected": _f(max(pnl_delta, 0.0)) if np.isfinite(pnl_delta) else None,
        "capital_lost": _f(max(-pnl_delta, 0.0)) if np.isfinite(pnl_delta) else None,
        "rule_confidence": _f(min(len(blocked) * 1.0, 70.0), 0),
        "method": "blocked ADDs' 5-day forward move inverted; dollars use average ADD notional",
    }


# ──────────────────────────────────────────────
# CARD 9 / S5 — Risk flag acceleration
# ──────────────────────────────────────────────
def simulate_risk_flags(tl: pd.DataFrame) -> Dict[str, Any]:
    df = tl.copy()
    df["fwd"] = pd.to_numeric(df["forward_return_5d"], errors="coerce")
    flagged = df[
        (df["risk_flags"].astype(str).str.len() > 0)
        & df["action"].isin(["BUY", "ADD", "HOLD"])
        & df["fwd"].notna()
    ].copy()
    n = int(len(flagged))
    if not n:
        return {
            "scenario": "RISK_FLAG_ACCELERATION",
            "description": "Exit/trim earlier when a risk flag fires on a held position",
            "simulated_pnl_delta": 0.0,
            "impacted_trades": 0,
            "rule_confidence": 0.0,
            "note": "no flagged held-position events with forward returns in history",
        }

    mv = (
        pd.to_numeric(flagged["position_size"], errors="coerce")
        * pd.to_numeric(flagged["price"], errors="coerce")
    ).fillna(0.0)
    protected = float((mv * (-flagged["fwd"]).clip(lower=0) / 100.0).sum())
    missed = float((mv * flagged["fwd"].clip(lower=0) / 100.0).sum())
    false_exits = int((flagged["fwd"] > 0).sum())
    correct = int((flagged["fwd"] < 0).sum())
    net = protected - missed

    return {
        "scenario": "RISK_FLAG_ACCELERATION",
        "description": "Exit/trim earlier when a risk flag fires on a held position",
        "simulated_pnl_delta": _f(net),
        "impacted_trades": n,
        "correct_predictions": correct,
        "false_exits": false_exits,
        "false_positive_risk": _f(false_exits / n * 100.0, 1),
        "false_negative_risk": None,
        "capital_protected": _f(protected),
        "capital_lost": _f(missed),
        "rule_confidence": _f(min(n * 2.0, 60.0), 0),
        "note": (
            "canonical flags (TRIM_PRIORITY/FORCE_EXIT/...) never reached decision records; "
            "simulation uses the observed flags (caps, duplicate blocks) as proxies"
        ),
    }


# ──────────────────────────────────────────────
# S6 — Combined ruleset + ranking
# ──────────────────────────────────────────────
def combine_scenarios(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best safe combination: best exit rule + every other net-positive rule."""
    by_name = {s["scenario"]: s for s in scenarios}
    s1 = by_name.get("EARLIER_EXIT_5D", {})
    s2 = by_name.get("EARLIER_EXIT_PEAK_DRAWDOWN", {})
    exit_pick = (
        s2
        if _to_float(s2.get("simulated_pnl_delta"), -1e9)
        >= _to_float(s1.get("simulated_pnl_delta"), -1e9)
        else s1
    )

    total = _to_float(exit_pick.get("simulated_pnl_delta"), 0.0)
    included = [exit_pick.get("scenario", "")]
    impacted = int(_to_float(exit_pick.get("impacted_trades"), 0))
    for name in ("MID_CONFIDENCE_FILTER", "HOLD_ADD_RECONFIRMATION", "RISK_FLAG_ACCELERATION"):
        s = by_name.get(name, {})
        delta = _to_float(s.get("simulated_pnl_delta"), np.nan)
        if np.isfinite(delta) and delta > 0:
            total += delta
            included.append(name)
            impacted += int(_to_float(s.get("impacted_trades"), 0))

    confs = [_to_float(by_name[n].get("rule_confidence"), np.nan) for n in included if n in by_name]
    confs = [c for c in confs if np.isfinite(c)]
    return {
        "scenario": "COMBINED_OPTIMIZED_RULESET",
        "description": "Best exit rule plus all net-positive filters (non-overlapping sum)",
        "simulated_pnl_delta": _f(total),
        "impacted_trades": impacted,
        "included_rules": included,
        "false_positive_risk": (
            _f(
                np.mean(
                    [
                        _to_float(by_name[n].get("false_positive_risk"), np.nan)
                        for n in included
                        if np.isfinite(_to_float(by_name[n].get("false_positive_risk"), np.nan))
                    ]
                ),
                1,
            )
            if included
            else None
        ),
        "capital_protected": _f(max(total, 0.0)),
        "capital_lost": _f(max(-total, 0.0)),
        "rule_confidence": _f(min(confs) if confs else 0.0, 0),
    }


def classify_scenario(s: Dict[str, Any], baseline_pnl: float) -> str:
    delta = _to_float(s.get("simulated_pnl_delta"), np.nan)
    fp = _to_float(s.get("false_positive_risk"), np.nan)
    n = int(_to_float(s.get("impacted_trades"), 0))
    if not np.isfinite(delta) or n == 0:
        return "NEEDS_MORE_DATA"
    if delta > 0 and n < 10:
        return "NEEDS_MORE_DATA"
    if delta > 0 and (not np.isfinite(fp) or fp <= 50.0):
        return "PROMISING"
    if delta > 0:
        return "NEEDS_MORE_DATA"
    if n < 15 and abs(delta) < 200.0:
        return "NEEDS_MORE_DATA"  # too few samples to condemn the rule
    if abs(delta) > max(abs(baseline_pnl) * 0.5, 500.0) or (np.isfinite(fp) and fp >= 70.0):
        return "DANGEROUS"
    return "REJECT"


def rank_scenarios(scenarios: List[Dict[str, Any]], baseline: Dict[str, Any]) -> pd.DataFrame:
    baseline_pnl = _to_float(baseline.get("realized_pnl"), 0.0)
    rows: List[Dict[str, Any]] = []
    for s in scenarios:
        delta = _to_float(s.get("simulated_pnl_delta"), np.nan)
        fp = _to_float(s.get("false_positive_risk"), np.nan)
        conf = _to_float(s.get("rule_confidence"), np.nan)
        n = int(_to_float(s.get("impacted_trades"), 0))
        simplicity = {
            "EARLIER_EXIT_5D": 90,
            "EARLIER_EXIT_PEAK_DRAWDOWN": 75,
            "MID_CONFIDENCE_FILTER": 80,
            "HOLD_ADD_RECONFIRMATION": 70,
            "RISK_FLAG_ACCELERATION": 50,
            "COMBINED_OPTIMIZED_RULESET": 40,
        }.get(str(s.get("scenario")), 50)
        # Composite rank score: dollars (normalized), FP penalty, confidence, simplicity.
        pnl_component = float(np.clip(50.0 + (delta / 50.0 if np.isfinite(delta) else 0.0), 0, 100))
        fp_component = float(np.clip(100.0 - (fp if np.isfinite(fp) else 50.0), 0, 100))
        conf_component = float(np.clip(conf if np.isfinite(conf) else 0.0, 0, 100))
        rank_score = (
            0.40 * pnl_component + 0.25 * fp_component + 0.20 * conf_component + 0.15 * simplicity
        )
        rows.append(
            {
                "scenario": s.get("scenario"),
                "simulated_pnl_delta": _f(delta),
                "impacted_trades": n,
                "false_positive_risk": _f(fp, 1),
                "rule_confidence": _f(conf, 0),
                "simplicity": simplicity,
                "rank_score": _f(rank_score, 1),
                "classification": classify_scenario(s, baseline_pnl),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# CARD 12 — Certification
# ──────────────────────────────────────────────
def build_certification(
    baseline: Dict[str, Any], intel: Dict[str, Any], rankings_df: pd.DataFrame
) -> Dict[str, Any]:
    reasons: List[str] = []

    # Baseline replay validity: must agree with Phase 167 intelligence.
    intel_summary = intel.get("summary") or {}
    intel_alpha = _to_float(intel_summary.get("overall_alpha_score"), np.nan)
    base_alpha = _to_float(baseline.get("decision_alpha"), np.nan)
    baseline_valid = (
        np.isfinite(intel_alpha) and np.isfinite(base_alpha) and abs(intel_alpha - base_alpha) < 1.0
    )
    if not baseline_valid:
        reasons.append("baseline replay does not match Phase 167 decision alpha")

    late_n = int((baseline.get("exit_outcomes") or {}).get("LATE_EXIT", 0))
    if late_n < 30:
        reasons.append(
            f"exit scenarios rest on only {late_n} LATE_EXIT samples — directional, not statistical"
        )
    trade_n = int(_to_float(baseline.get("trade_count"), 0))
    if trade_n < 100:
        reasons.append(f"only {trade_n} realized trades in baseline")

    dangerous = (
        rankings_df[rankings_df["classification"] == "DANGEROUS"]
        if not rankings_df.empty
        else pd.DataFrame()
    )
    if not dangerous.empty:
        reasons.append(
            f"{len(dangerous)} scenario(s) classified DANGEROUS — excluded from recommendation"
        )

    if not baseline_valid:
        status = "NOT_CERTIFIED"
    elif reasons:
        status = "PARTIALLY_CERTIFIED"
    else:
        status = "CERTIFIED"
    return {
        "status": status,
        "baseline_replay_valid": bool(baseline_valid),
        "late_exit_samples": late_n,
        "realized_trade_samples": trade_n,
        "reasons": reasons,
        "policy": "no scenario is recommended for Phase 170 unless classified PROMISING",
        "safety": "simulation-only — no rules, thresholds, signals, execution, or orders modified",
    }


def build_phase_170_recommendation(
    rankings_df: pd.DataFrame, cert: Dict[str, Any]
) -> Dict[str, Any]:
    promising = (
        rankings_df[rankings_df["classification"] == "PROMISING"]
        if not rankings_df.empty
        else pd.DataFrame()
    )
    rejected = (
        rankings_df[rankings_df["classification"].isin(["REJECT", "DANGEROUS"])]
        if not rankings_df.empty
        else pd.DataFrame()
    )
    needs_data = (
        rankings_df[rankings_df["classification"] == "NEEDS_MORE_DATA"]
        if not rankings_df.empty
        else pd.DataFrame()
    )

    implement = promising["scenario"].tolist() if not promising.empty else []
    return {
        "implement": implement,
        "do_not_implement": rejected["scenario"].tolist() if not rejected.empty else [],
        "collect_more_data": needs_data["scenario"].tolist() if not needs_data.empty else [],
        "headline": (
            f"Phase 170 should implement: {', '.join(implement)}"
            if implement
            else "No scenario met the PROMISING bar — Phase 170 should extend data collection before changing rules"
        ),
        "caveat": (
            "certification is "
            + str(cert.get("status"))
            + "; all dollar impacts are replay approximations against historical prices"
        ),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    tl = build_ticker_timelines()
    intel = _read_json(RESULTS / "lifecycle_decision_intelligence.json")
    opt = _read_json(RESULTS / "adaptive_lifecycle_optimization.json")

    trades = _safe_read_csv(RESULTS / "trade_outcomes.csv")
    if not trades.empty:
        trades.columns = [str(c).strip().lower() for c in trades.columns]
        trades = trades[trades.get("outcome_type", "").astype(str).str.upper() == "REALIZED"].copy()
        trades["realized_pl"] = pd.to_numeric(trades.get("realized_pl"), errors="coerce")

    baseline = build_baseline(tl, intel, trades)
    _log(f"baseline: pnl {baseline.get('realized_pnl')} | alpha {baseline.get('decision_alpha')}")

    s1, s2, exit_detail = simulate_exit_timing(tl)
    s3 = simulate_confidence_filter(tl, intel)
    s4 = simulate_hold_add(tl, intel)
    s5 = simulate_risk_flags(tl)
    s6 = combine_scenarios([s1, s2, s3, s4, s5])
    scenarios = [s1, s2, s3, s4, s5, s6]

    rankings_df = rank_scenarios(scenarios, baseline)
    cert = build_certification(baseline, intel, rankings_df)
    phase170 = build_phase_170_recommendation(rankings_df, cert)

    # Exit timing block for the dashboard (Card 6).
    exits_block = {
        "late_exit_count": int((tl[(tl["action"] == "EXIT")]["exit_outcome"] == "LATE_EXIT").sum()),
        "table": exit_detail.to_dict(orient="records") if not exit_detail.empty else [],
    }

    pd.DataFrame(scenarios).to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")

    comparison_rows = [{"metric": "realized_pnl", "baseline": baseline.get("realized_pnl")}]
    for s in scenarios:
        comparison_rows.append(
            {
                "metric": f"pnl_delta::{s['scenario']}",
                "baseline": baseline.get("realized_pnl"),
                "simulated_delta": s.get("simulated_pnl_delta"),
                "simulated_total": _f(
                    _to_float(baseline.get("realized_pnl"), 0.0)
                    + _to_float(s.get("simulated_pnl_delta"), 0.0)
                ),
                "impacted_trades": s.get("impacted_trades"),
                "false_positive_risk": s.get("false_positive_risk"),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(OUT_COMPARISON, index=False, encoding="utf-8")

    report = {
        "generated_at": _now_iso(),
        "baseline": baseline,
        "scenarios": scenarios,
        "scenario_rankings": rankings_df.to_dict(orient="records"),
        "exit_timing_simulation": exits_block,
        "phase_170_recommendation": phase170,
        "certification": cert,
        "method_notes": (
            "Replay-only what-if engine: simulated exits use actual historical closes; "
            "dollar deltas are (simulated price - actual price) x actual quantity. "
            "Confidence/HOLD-ADD filters approximate dollars with average ADD notional. "
            "No trades placed, no rules modified."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_SCENARIOS.name}, {OUT_COMPARISON.name}")
    _log(
        f"rankings:\n{rankings_df[['scenario','simulated_pnl_delta','classification']].to_string(index=False)}"
    )
    _log(f"phase 170: {phase170['headline']}")
    return report


if __name__ == "__main__":
    run()
