"""
TRITON Phase 167 — Lifecycle Decision Intelligence Audit.

Mines the Phase 166 decision timelines to determine which lifecycle decisions
create alpha and which destroy value: per-action effectiveness, confidence
calibration, risk-flag predictive value, lifecycle transitions, exit timing
cost, and per-symbol trust rankings — all rolled into 0-100 Decision Alpha
Scores.

Audit and analytics only — no trading, execution, or lifecycle rule changes.

Outputs (data/results/):
    lifecycle_decision_intelligence.json
    lifecycle_decision_rankings.csv
    lifecycle_alpha_scores.csv
    transition_rankings.csv

Run:
    python -m services.lifecycle_decision_intelligence
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

from services.pnl_ledger_builder import _norm_sym, _safe_read_csv, _to_float
from services.ticker_timeline_builder import build_ticker_timelines

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "lifecycle_decision_intelligence.json"
OUT_RANKINGS = RESULTS / "lifecycle_decision_rankings.csv"
OUT_ALPHA = RESULTS / "lifecycle_alpha_scores.csv"
OUT_TRANSITIONS = RESULTS / "transition_rankings.csv"

ACTIONS = ["BUY", "ADD", "HOLD", "TRIM", "EXIT"]
HELD_ACTIONS = {"BUY", "ADD", "HOLD"}  # long exposure after the decision
CANONICAL_FLAGS = [
    "TRIM_PRIORITY",
    "FORCE_EXIT",
    "ROTATE_EXIT",
    "CONCENTRATION_RISK",
    "HIGH_DRAG",
    "BLOCK_NEW_BUY",
]
CONF_BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}


def _log(msg: str) -> None:
    print(f"[lifecycle_decision_intelligence] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v: Any) -> Optional[float]:
    x = _to_float(v, np.nan)
    return round(float(x), 4) if np.isfinite(x) else None


def classify_alpha(score: float) -> str:
    if not np.isfinite(score):
        return "UNSCORED"
    if score >= 90:
        return "Elite"
    if score >= 75:
        return "Strong"
    if score >= 60:
        return "Average"
    if score >= 40:
        return "Weak"
    return "Harmful"


def _alpha_score(
    win_rate: float = np.nan,
    avg_return: float = np.nan,
    drawdown: float = np.nan,
    accuracy: float = np.nan,
    exit_quality: float = np.nan,
    risk_control: float = np.nan,
) -> float:
    """
    Unified 0-100 Decision Alpha Score.

    Components (each normalized to 0-100, missing components excluded):
      win_rate      already 0-100
      avg_return    alpha-aligned % return; ±5% maps to 0-100 around 50
      drawdown      avg adverse move %; 0 -> 100, -10% -> 0
      accuracy      share of graded decisions at C or better, 0-100
      exit_quality  share of correct/neutral exits, 0-100
      risk_control  share of flagged events that avoided losses, 0-100
    """
    comps: List[float] = []
    weights: List[float] = []

    def add(value: float, weight: float) -> None:
        if np.isfinite(value):
            comps.append(float(np.clip(value, 0.0, 100.0)))
            weights.append(weight)

    add(win_rate, 0.30)
    if np.isfinite(avg_return):
        add(50.0 + avg_return * 10.0, 0.25)
    if np.isfinite(drawdown):
        add(100.0 + drawdown * 10.0, 0.15)
    add(accuracy, 0.15)
    add(exit_quality, 0.10)
    add(risk_control, 0.05)

    if not comps:
        return np.nan
    return float(np.average(comps, weights=weights))


def _aligned_return(df: pd.DataFrame) -> pd.Series:
    """5-day forward return aligned with the decision's intent (exits invert)."""
    fwd = pd.to_numeric(df["forward_return_5d"], errors="coerce")
    sign = df["action"].map(lambda a: 1.0 if str(a) in HELD_ACTIONS else -1.0)
    return fwd * sign


# ──────────────────────────────────────────────
# CARD 3 — Decision effectiveness per action
# ──────────────────────────────────────────────
def build_decision_rankings(tl: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "action",
        "decision_count",
        "win_rate",
        "average_return",
        "median_return",
        "average_hold_time",
        "average_confidence",
        "capital_deployed",
        "capital_returned",
        "realized_pnl",
        "unrealized_pnl",
        "alpha_score",
        "classification",
    ]
    if tl.empty:
        return pd.DataFrame(columns=cols)

    # Hold time proxy: days until the next event for the same ticker.
    tl = tl.sort_values(["ticker", "timestamp"]).copy()
    tl["next_ts"] = tl.groupby("ticker")["timestamp"].shift(-1)
    tl["days_to_next"] = (tl["next_ts"] - tl["timestamp"]).dt.total_seconds() / 86400.0

    open_unreal = (
        float(pd.to_numeric(trades.get("unrealized_pl"), errors="coerce").sum())
        if not trades.empty
        else np.nan
    )

    rows: List[Dict[str, Any]] = []
    for action in ACTIONS:
        grp = tl[tl["action"] == action]
        if grp.empty:
            rows.append({"action": action, "decision_count": 0})
            continue
        aligned = _aligned_return(grp).dropna()
        win_rate = float((aligned > 0).mean() * 100.0) if len(aligned) else np.nan
        drawdown = float(aligned[aligned < 0].mean()) if (aligned < 0).any() else 0.0

        qty = pd.to_numeric(grp["qty"], errors="coerce")
        px = pd.to_numeric(grp["price"], errors="coerce")
        notional = (qty * px).where(grp["event_source"] == "fill_ledger")
        deployed = float(notional[grp["action"].isin(["BUY", "ADD"])].sum())
        returned = float(notional[grp["action"].isin(["TRIM", "EXIT"])].sum())

        graded = grp[grp["decision_grade"].astype(str).str.len() > 0]
        pts = graded["decision_grade"].map(GRADE_POINTS).dropna()
        accuracy = float((pts >= 2.0).mean() * 100.0) if len(pts) else np.nan

        exit_quality = np.nan
        if action == "EXIT":
            outcomes = grp["exit_outcome"].astype(str)
            scored = outcomes[outcomes.str.len() > 0]
            if len(scored):
                exit_quality = float(scored.isin(["CORRECT_EXIT", "NEUTRAL_EXIT"]).mean() * 100.0)

        score = _alpha_score(
            win_rate=win_rate,
            avg_return=float(aligned.mean()) if len(aligned) else np.nan,
            drawdown=drawdown,
            accuracy=accuracy,
            exit_quality=exit_quality,
        )
        rows.append(
            {
                "action": action,
                "decision_count": int(len(grp)),
                "win_rate": _f(win_rate),
                "average_return": _f(aligned.mean()) if len(aligned) else None,
                "median_return": _f(aligned.median()) if len(aligned) else None,
                "average_hold_time": _f(grp["days_to_next"].mean()),
                "average_confidence": _f(pd.to_numeric(grp["confidence"], errors="coerce").mean()),
                "capital_deployed": _f(deployed),
                "capital_returned": _f(returned),
                "realized_pnl": _f(pd.to_numeric(grp["realized_pnl"], errors="coerce").sum()),
                "unrealized_pnl": _f(open_unreal) if action == "HOLD" else None,
                "alpha_score": _f(score),
                "classification": classify_alpha(score),
            }
        )
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return (
        df[cols]
        .sort_values("alpha_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────
# CARD 4 — Confidence analysis
# ──────────────────────────────────────────────
def build_confidence_table(tl: pd.DataFrame) -> pd.DataFrame:
    cols = ["bucket", "decision_count", "win_rate", "avg_return", "alpha_score", "classification"]
    if tl.empty:
        return pd.DataFrame(columns=cols)
    df = tl.copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence"])
    df["aligned"] = _aligned_return(df)

    rows: List[Dict[str, Any]] = []
    for lo, hi in CONF_BUCKETS:
        grp = df[
            (df["confidence"] >= lo)
            & (df["confidence"] < hi if hi < 1.0 else df["confidence"] <= hi)
        ]
        aligned = grp["aligned"].dropna()
        win = float((aligned > 0).mean() * 100.0) if len(aligned) else np.nan
        avg = float(aligned.mean()) if len(aligned) else np.nan
        dd = (
            float(aligned[aligned < 0].mean())
            if (aligned < 0).any()
            else (0.0 if len(aligned) else np.nan)
        )
        score = _alpha_score(win_rate=win, avg_return=avg, drawdown=dd)
        rows.append(
            {
                "bucket": f"{lo:.2f}–{hi:.2f}",
                "decision_count": int(len(grp)),
                "win_rate": _f(win),
                "avg_return": _f(avg),
                "alpha_score": _f(score),
                "classification": classify_alpha(score),
            }
        )
    return pd.DataFrame(rows, columns=cols)


# ──────────────────────────────────────────────
# CARD 5 — Risk flag intelligence
# ──────────────────────────────────────────────
def build_flag_table(tl: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "flag",
        "frequency",
        "avg_return_after_flag",
        "avg_drawdown_after_flag",
        "win_rate",
        "alpha_score",
        "classification",
        "observed",
    ]
    if tl.empty:
        return pd.DataFrame(columns=cols)

    df = tl.copy()
    df["flag_text"] = (
        df["risk_flags"].astype(str).str.upper()
        + " "
        + df["lifecycle_state"].astype(str).str.upper()
        + " "
        + df["execution_session"].astype(str).str.upper()
    )
    df["fwd"] = pd.to_numeric(df["forward_return_5d"], errors="coerce")

    # Observed non-canonical flags (lifecycle decision reasons, skip reasons).
    observed: Dict[str, pd.DataFrame] = {}
    flagged = df[df["risk_flags"].astype(str).str.len() > 0]
    for raw in flagged["risk_flags"].astype(str):
        key = raw.split(";")[0].strip().split(":")[0].strip()
        if key and key.upper() not in CANONICAL_FLAGS:
            observed.setdefault(key, None)

    rows: List[Dict[str, Any]] = []

    def _flag_row(name: str, grp: pd.DataFrame, is_observed: bool) -> Dict[str, Any]:
        fwd = grp["fwd"].dropna()
        # A risk flag "predicts" correctly when the price subsequently falls.
        win = float((fwd < 0).mean() * 100.0) if len(fwd) else np.nan
        avg = float(fwd.mean()) if len(fwd) else np.nan
        dd = float(fwd[fwd < 0].mean()) if (fwd < 0).any() else (0.0 if len(fwd) else np.nan)
        score = _alpha_score(
            win_rate=win,
            avg_return=-avg if np.isfinite(avg) else np.nan,
            drawdown=np.nan,
            risk_control=win,
        )
        return {
            "flag": name,
            "frequency": int(len(grp)),
            "avg_return_after_flag": _f(avg),
            "avg_drawdown_after_flag": _f(dd),
            "win_rate": _f(win),
            "alpha_score": _f(score),
            "classification": classify_alpha(score),
            "observed": is_observed,
        }

    for flag in CANONICAL_FLAGS:
        grp = df[df["flag_text"].str.contains(flag, regex=False, na=False)]
        rows.append(_flag_row(flag, grp, not grp.empty))
    for flag in observed:
        grp = flagged[flagged["risk_flags"].astype(str).str.startswith(flag)]
        rows.append(_flag_row(flag, grp, True))

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["frequency", "alpha_score"], ascending=[False, False]).reset_index(
        drop=True
    )


# ──────────────────────────────────────────────
# CARD 6 — Lifecycle transition analysis
# ──────────────────────────────────────────────
def build_transition_table(tl: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "transition",
        "count",
        "avg_return",
        "avg_drawdown",
        "success_rate",
        "alpha_score",
        "classification",
    ]
    if tl.empty:
        return pd.DataFrame(columns=cols)

    df = tl[tl["action"].isin(ACTIONS)].sort_values(["ticker", "timestamp"]).copy()
    df["price_n"] = pd.to_numeric(df["price"], errors="coerce")
    df["next_action"] = df.groupby("ticker")["action"].shift(-1)
    df["next_price"] = df.groupby("ticker")["price_n"].shift(-1)
    df = df.dropna(subset=["next_action"])
    df["ret_between"] = (df["next_price"] - df["price_n"]) / df["price_n"] * 100.0
    # Long exposure during the leg only when the first action keeps us in the position.
    df["aligned"] = np.where(
        df["action"].isin(list(HELD_ACTIONS)), df["ret_between"], -df["ret_between"]
    )

    rows: List[Dict[str, Any]] = []
    for (a, b), grp in df.groupby(["action", "next_action"]):
        aligned = pd.to_numeric(grp["aligned"], errors="coerce").dropna()
        success = float((aligned > 0).mean() * 100.0) if len(aligned) else np.nan
        avg = float(aligned.mean()) if len(aligned) else np.nan
        dd = (
            float(aligned[aligned < 0].mean())
            if (aligned < 0).any()
            else (0.0 if len(aligned) else np.nan)
        )
        score = _alpha_score(win_rate=success, avg_return=avg, drawdown=dd)
        rows.append(
            {
                "transition": f"{a} → {b}",
                "count": int(len(grp)),
                "avg_return": _f(avg),
                "avg_drawdown": _f(dd),
                "success_rate": _f(success),
                "alpha_score": _f(score),
                "classification": classify_alpha(score),
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("alpha_score", ascending=False, na_position="last").reset_index(
        drop=True
    )


# ──────────────────────────────────────────────
# CARD 7 — Exit timing audit
# ──────────────────────────────────────────────
def build_exit_timing(tl: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "exit_outcome",
        "count",
        "capital_saved",
        "capital_lost",
        "avg_post_exit_return",
        "alpha_score",
        "classification",
    ]
    exits = tl[(tl["action"] == "EXIT") & (tl["exit_outcome"].astype(str).str.len() > 0)].copy()
    if exits.empty:
        return pd.DataFrame(columns=cols)

    exits["mv"] = pd.to_numeric(exits["qty"], errors="coerce") * pd.to_numeric(
        exits["price"], errors="coerce"
    )
    exits["pe10"] = pd.to_numeric(exits["post_exit_return_10d"], errors="coerce")
    # Negative post-exit return = loss avoided (saved); positive = gain missed (lost).
    exits["saved"] = (exits["mv"] * (-exits["pe10"]).clip(lower=0) / 100.0).fillna(0.0)
    exits["lost"] = (exits["mv"] * exits["pe10"].clip(lower=0) / 100.0).fillna(0.0)

    rows: List[Dict[str, Any]] = []
    for outcome in ("CORRECT_EXIT", "EARLY_EXIT", "LATE_EXIT", "NEUTRAL_EXIT"):
        grp = exits[exits["exit_outcome"] == outcome]
        if grp.empty:
            rows.append(
                {"exit_outcome": outcome, "count": 0, "capital_saved": 0.0, "capital_lost": 0.0}
            )
            continue
        pe = grp["pe10"].dropna()
        win = float((pe < 0).mean() * 100.0) if len(pe) else np.nan
        score = _alpha_score(
            win_rate=win,
            avg_return=float(-pe.mean()) if len(pe) else np.nan,
            exit_quality=(
                100.0 if outcome == "CORRECT_EXIT" else (50.0 if outcome == "NEUTRAL_EXIT" else 0.0)
            ),
        )
        rows.append(
            {
                "exit_outcome": outcome,
                "count": int(len(grp)),
                "capital_saved": _f(grp["saved"].sum()),
                "capital_lost": _f(grp["lost"].sum()),
                "avg_post_exit_return": _f(pe.mean()) if len(pe) else None,
                "alpha_score": _f(score),
                "classification": classify_alpha(score),
            }
        )
    return pd.DataFrame(rows, columns=cols)


# ──────────────────────────────────────────────
# CARD 8 — Symbol intelligence
# ──────────────────────────────────────────────
def build_symbol_table(tl: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "symbol",
        "realized_pnl",
        "decision_count",
        "decision_accuracy",
        "average_confidence",
        "alpha_score",
        "classification",
    ]
    if tl.empty:
        return pd.DataFrame(columns=cols)

    realized_by_sym: Dict[str, float] = {}
    if not trades.empty:
        t = trades.copy()
        t["symbol"] = t["symbol"].map(_norm_sym)
        realized_by_sym = t.groupby("symbol")["realized_pl"].sum().astype(float).to_dict()

    rows: List[Dict[str, Any]] = []
    for sym, grp in tl.groupby("ticker"):
        graded = grp[grp["decision_grade"].astype(str).str.len() > 0]
        pts = graded["decision_grade"].map(GRADE_POINTS).dropna()
        accuracy = float((pts >= 2.0).mean() * 100.0) if len(pts) else np.nan
        aligned = _aligned_return(grp).dropna()
        win = float((aligned > 0).mean() * 100.0) if len(aligned) else np.nan
        realized = float(realized_by_sym.get(str(sym), 0.0))
        score = _alpha_score(
            win_rate=win,
            avg_return=float(aligned.mean()) if len(aligned) else np.nan,
            accuracy=accuracy,
        )
        # Realized P&L tilts the score so money made/lost dominates ties.
        if np.isfinite(score):
            score = float(np.clip(score + np.clip(realized / 50.0, -15.0, 15.0), 0.0, 100.0))
        rows.append(
            {
                "symbol": sym,
                "realized_pnl": _f(realized),
                "decision_count": int(grp["action"].isin(ACTIONS).sum()),
                "decision_accuracy": _f(accuracy),
                "average_confidence": _f(pd.to_numeric(grp["confidence"], errors="coerce").mean()),
                "alpha_score": _f(score),
                "classification": classify_alpha(score),
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("alpha_score", ascending=False, na_position="last").reset_index(
        drop=True
    )


# ──────────────────────────────────────────────
# CARD 10 — Executive summary
# ──────────────────────────────────────────────
def _pick(
    df: pd.DataFrame, col: str, by: str, best: bool, min_count_col: str = "", min_count: int = 1
) -> str:
    if df.empty or by not in df.columns:
        return "—"
    d = df.copy()
    d[by] = pd.to_numeric(d[by], errors="coerce")
    if min_count_col and min_count_col in d.columns:
        d = d[pd.to_numeric(d[min_count_col], errors="coerce").fillna(0) >= min_count]
    d = d.dropna(subset=[by])
    if d.empty:
        return "—"
    row = d.loc[d[by].idxmax()] if best else d.loc[d[by].idxmin()]
    return str(row[col])


def build_summary(
    rankings: pd.DataFrame,
    confidence: pd.DataFrame,
    flags: pd.DataFrame,
    transitions: pd.DataFrame,
    exits: pd.DataFrame,
    symbols: pd.DataFrame,
) -> Dict[str, Any]:
    scored = rankings[pd.to_numeric(rankings["alpha_score"], errors="coerce").notna()]
    overall = (
        float(pd.to_numeric(scored["alpha_score"], errors="coerce").mean())
        if not scored.empty
        else np.nan
    )
    grade = (
        (
            "A"
            if overall >= 85
            else "B" if overall >= 70 else "C" if overall >= 55 else "D" if overall >= 40 else "F"
        )
        if np.isfinite(overall)
        else "—"
    )

    late = exits[exits["exit_outcome"] == "LATE_EXIT"] if not exits.empty else pd.DataFrame()
    late_cost = (
        float(pd.to_numeric(late.get("capital_lost"), errors="coerce").sum())
        if not late.empty
        else 0.0
    )

    observed_flags = (
        flags[flags.get("observed", pd.Series(dtype=bool)) == True] if not flags.empty else flags
    )  # noqa: E712

    return {
        "best_decision_type": _pick(rankings, "action", "alpha_score", True, "decision_count", 1),
        "worst_decision_type": _pick(rankings, "action", "alpha_score", False, "decision_count", 1),
        "best_confidence_range": _pick(
            confidence, "bucket", "alpha_score", True, "decision_count", 5
        ),
        "worst_confidence_range": _pick(
            confidence, "bucket", "alpha_score", False, "decision_count", 5
        ),
        "most_valuable_risk_flag": _pick(
            observed_flags, "flag", "alpha_score", True, "frequency", 1
        ),
        "least_valuable_risk_flag": _pick(
            observed_flags, "flag", "alpha_score", False, "frequency", 1
        ),
        "best_lifecycle_transition": _pick(
            transitions, "transition", "alpha_score", True, "count", 2
        ),
        "worst_lifecycle_transition": _pick(
            transitions, "transition", "alpha_score", False, "count", 2
        ),
        "top_alpha_generator": _pick(symbols, "symbol", "alpha_score", True),
        "top_alpha_destroyer": _pick(symbols, "symbol", "alpha_score", False),
        "late_exit_cost": _f(late_cost),
        "overall_alpha_score": _f(overall),
        "overall_lifecycle_grade": grade,
    }


def build_certification(tl: pd.DataFrame) -> Dict[str, Any]:
    """Data-sufficiency certification for this intelligence report."""
    graded = int((tl["decision_grade"].astype(str).str.len() > 0).sum()) if not tl.empty else 0
    exits = int((tl["exit_outcome"].astype(str).str.len() > 0).sum()) if not tl.empty else 0
    conf = (
        int(pd.to_numeric(tl.get("confidence"), errors="coerce").notna().sum())
        if not tl.empty
        else 0
    )
    reasons: List[str] = []
    if graded < 30:
        reasons.append(f"only {graded} graded decisions (need 30+ for statistical weight)")
    if exits < 10:
        reasons.append(f"only {exits} classified exits")
    if conf < 30:
        reasons.append(f"only {conf} events carry confidence values")

    integrity = {}
    p = RESULTS / "ledger_integrity_report.json"
    if p.is_file():
        try:
            integrity = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            integrity = {}
    if integrity and integrity.get("status") != "CERTIFIED":
        reasons.append(f"ledger integrity status is {integrity.get('status')}")

    status = (
        "CERTIFIED" if not reasons else ("PARTIALLY_CERTIFIED" if graded >= 30 else "NOT_CERTIFIED")
    )
    return {
        "status": status,
        "graded_decisions": graded,
        "classified_exits": exits,
        "confidence_events": conf,
        "ledger_integrity_status": integrity.get("status", "UNKNOWN"),
        "reasons": reasons,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    tl = build_ticker_timelines()
    _log(f"timeline events: {len(tl)}")

    trades = _safe_read_csv(RESULTS / "trade_outcomes.csv")
    if not trades.empty:
        trades.columns = [str(c).strip().lower() for c in trades.columns]
        trades = trades[trades.get("outcome_type", "").astype(str).str.upper() == "REALIZED"].copy()
        trades["realized_pl"] = pd.to_numeric(trades.get("realized_pl"), errors="coerce")

    rankings = build_decision_rankings(tl, trades)
    confidence = build_confidence_table(tl)
    flags = build_flag_table(tl)
    transitions = build_transition_table(tl)
    exits = build_exit_timing(tl)
    symbols = build_symbol_table(tl, trades)
    summary = build_summary(rankings, confidence, flags, transitions, exits, symbols)
    certification = build_certification(tl)

    rankings.to_csv(OUT_RANKINGS, index=False, encoding="utf-8")
    transitions.to_csv(OUT_TRANSITIONS, index=False, encoding="utf-8")

    # Unified alpha-score file across every scored entity.
    alpha_rows: List[Dict[str, Any]] = []
    for _, r in rankings.iterrows():
        alpha_rows.append(
            {
                "entity_type": "action",
                "entity": r["action"],
                "alpha_score": r["alpha_score"],
                "classification": r["classification"],
            }
        )
    for _, r in confidence.iterrows():
        alpha_rows.append(
            {
                "entity_type": "confidence_bucket",
                "entity": r["bucket"],
                "alpha_score": r["alpha_score"],
                "classification": r["classification"],
            }
        )
    for _, r in flags.iterrows():
        alpha_rows.append(
            {
                "entity_type": "risk_flag",
                "entity": r["flag"],
                "alpha_score": r["alpha_score"],
                "classification": r["classification"],
            }
        )
    for _, r in transitions.iterrows():
        alpha_rows.append(
            {
                "entity_type": "transition",
                "entity": r["transition"],
                "alpha_score": r["alpha_score"],
                "classification": r["classification"],
            }
        )
    for _, r in exits.iterrows():
        alpha_rows.append(
            {
                "entity_type": "exit_outcome",
                "entity": r["exit_outcome"],
                "alpha_score": r["alpha_score"],
                "classification": r["classification"],
            }
        )
    for _, r in symbols.iterrows():
        alpha_rows.append(
            {
                "entity_type": "symbol",
                "entity": r["symbol"],
                "alpha_score": r["alpha_score"],
                "classification": r["classification"],
            }
        )
    pd.DataFrame(alpha_rows).to_csv(OUT_ALPHA, index=False, encoding="utf-8")

    report = {
        "generated_at": _now_iso(),
        "events_analyzed": int(len(tl)),
        "summary": summary,
        "certification": certification,
        "decision_rankings": rankings.to_dict(orient="records"),
        "confidence_performance_table": confidence.to_dict(orient="records"),
        "risk_flag_intelligence": flags.to_dict(orient="records"),
        "transition_rankings": transitions.to_dict(orient="records"),
        "exit_timing_audit": exits.to_dict(orient="records"),
        "symbol_intelligence": symbols.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_RANKINGS.name}, {OUT_ALPHA.name}, {OUT_TRANSITIONS.name}")
    _log(f"summary: {json.dumps(summary, default=str)}")
    return report


if __name__ == "__main__":
    run()
