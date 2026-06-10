"""
TRITON Phase 166 — 🧬 Ticker Lifecycle Tracker (read-only).

Explainability and forensic-analysis page: select any ticker and see the full
BUY / ADD / HOLD / TRIM / EXIT decision history, rationale evolution, P&L
attribution, decision-quality grades, and post-exit analysis.

Observability only — never modifies trading, execution, or lifecycle logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import (
    _money,
    _num,
    _pct,
    _plotly_bar,
    _plotly_line,
    _safe_float,
)
from services.ticker_timeline_builder import empty_timeline, load_timelines

_GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
_ACTION_ICONS = {
    "BUY": "🟢",
    "ADD": "➕",
    "HOLD": "⏸️",
    "TRIM": "✂️",
    "EXIT": "🔴",
    "WAIT": "⏳",
}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=120)
def _cached_timelines(_seed: str) -> pd.DataFrame:
    try:
        return load_timelines()
    except Exception:
        return empty_timeline()


def _grade_letter(points: float) -> str:
    if not np.isfinite(points):
        return "—"
    for letter, threshold in (("A", 3.5), ("B", 2.5), ("C", 1.5), ("D", 0.5)):
        if points >= threshold:
            return letter
    return "F"


def _realized_trades_for(results_dir: Path, ticker: str) -> pd.DataFrame:
    t = _read_csv_safe(results_dir / "trade_outcomes.csv")
    if t.empty:
        return pd.DataFrame()
    t.columns = [str(c).strip().lower() for c in t.columns]
    t["symbol"] = (
        t.get("symbol", pd.Series(dtype=str))
        .astype(str)
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    t = t[
        (t["symbol"] == ticker) & (t.get("outcome_type", "").astype(str).str.upper() == "REALIZED")
    ]
    if t.empty:
        return t
    t["realized_pl"] = pd.to_numeric(t.get("realized_pl"), errors="coerce")
    t["opened_at"] = pd.to_datetime(t.get("opened_at"), errors="coerce", utc=True)
    t["closed_at"] = pd.to_datetime(t.get("closed_at"), errors="coerce", utc=True)
    return t.sort_values("closed_at").reset_index(drop=True)


def _position_for(results_dir: Path, ticker: str) -> Dict[str, Any]:
    snap = _read_csv_safe(results_dir / "positions_snapshot.csv")
    if snap.empty:
        return {}
    snap.columns = [str(c).strip().lower() for c in snap.columns]
    sym_col = "ticker" if "ticker" in snap.columns else "symbol"
    snap[sym_col] = snap[sym_col].astype(str).str.upper().str.replace(".", "-", regex=False)
    rows = snap[snap[sym_col] == ticker]
    if rows.empty:
        return {}
    r = rows.iloc[-1]
    return {
        "qty": _safe_float(r.get("qty")),
        "market_value": _safe_float(r.get("market_value")),
        "avg_entry_price": _safe_float(r.get("avg_entry_price")),
        "unrealized_pl": _safe_float(r.get("unrealized_pl")),
        "current_price": _safe_float(r.get("current_price")),
    }


def _verdict(
    total_pnl: float,
    realized: float,
    has_open: bool,
    exit_outcomes: List[str],
    avg_grade_pts: float,
) -> str:
    if not np.isfinite(total_pnl):
        return "NEEDS REVIEW"
    correct_exits = exit_outcomes.count("CORRECT_EXIT")
    if total_pnl > 100 and (not np.isfinite(avg_grade_pts) or avg_grade_pts >= 1.5):
        return "STRONG PERFORMER"
    if not has_open and realized > 0 and exit_outcomes:
        return "SUCCESSFUL ROTATION"
    if not has_open and realized <= 0 and correct_exits >= max(1, len(exit_outcomes) // 2):
        return "CAPITAL PRESERVATION SUCCESS"
    if total_pnl < -100:
        return "WEAK PERFORMER"
    if abs(total_pnl) <= 100:
        return "NEUTRAL / SMALL IMPACT"
    return "NEEDS REVIEW"


# ──────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────
def _render_executive_summary(
    ticker: str,
    tl: pd.DataFrame,
    pos: Dict[str, Any],
    trades: pd.DataFrame,
) -> None:
    snap = tl[tl["event_source"] == "current_snapshot"]
    current = snap.iloc[-1] if not snap.empty else (tl.iloc[-1] if not tl.empty else None)

    realized = float(trades["realized_pl"].sum()) if not trades.empty else 0.0
    unreal = _safe_float(pos.get("unrealized_pl")) if pos else float("nan")
    total = realized + (unreal if np.isfinite(unreal) else 0.0)

    decisions = tl[tl["action"].isin(["BUY", "ADD", "TRIM", "EXIT"])]
    graded = tl[tl["decision_grade"].astype(str).str.len() > 0]
    pts = (
        graded["decision_grade"].map(_GRADE_POINTS).dropna()
        if not graded.empty
        else pd.Series(dtype=float)
    )
    avg_pts = float(pts.mean()) if len(pts) else float("nan")
    avg_conf = (
        float(pd.to_numeric(tl["confidence"], errors="coerce").dropna().mean())
        if not tl.empty
        else float("nan")
    )
    exit_outcomes = [o for o in tl["exit_outcome"].astype(str) if o]

    has_open = bool(pos) and _safe_float(pos.get("qty")) > 0
    verdict = _verdict(total, realized, has_open, exit_outcomes, avg_pts)

    status = (
        "HELD"
        if has_open
        else ("EXITED" if not trades.empty or not decisions.empty else "NEVER TRADED")
    )
    lifecycle = (
        str(current.get("effective_lifecycle") or current.get("action") or "—")
        if current is not None
        else "—"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", ticker)
    c2.metric("Current Status", status)
    c3.metric("Current Lifecycle", lifecycle or "—")
    c4.metric("Current P&L", _money(total))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Decision Count", f"{len(decisions):,}")
    c6.metric("Avg Confidence", _num(avg_conf))
    c7.metric("Avg Grade", _grade_letter(avg_pts))
    c8.metric("Verdict", verdict)


def _render_current_status(ticker: str, tl: pd.DataFrame, pos: Dict[str, Any]) -> None:
    st.markdown("#### Current Status")
    snap = tl[tl["event_source"] == "current_snapshot"]
    if snap.empty and not pos:
        st.info("No current lifecycle snapshot or open position for this ticker.", icon="ℹ️")
        return
    cols = st.columns(4)
    if pos:
        cols[0].metric("Position", f"{_num(pos.get('qty'))} sh")
        cols[1].metric("Market Value", _money(pos.get("market_value")))
        cols[2].metric("Avg Entry", _money(pos.get("avg_entry_price")))
        cols[3].metric("Unrealized P&L", _money(pos.get("unrealized_pl")))
    else:
        cols[0].metric("Position", "FLAT")

    if not snap.empty:
        cur = snap.iloc[-1]
        st.metric("Current Confidence", _num(cur.get("confidence")))
        rationale = str(cur.get("rationale") or "").strip()
        if rationale:
            st.markdown(f"**Current Rationale:** {rationale}")
        flags = str(cur.get("risk_flags") or "").strip()
        if flags:
            st.warning(f"Risk flags: {flags}", icon="⚠️")


def _render_timeline(tl: pd.DataFrame) -> None:
    st.markdown("#### 🧬 Decision Timeline")
    if tl.empty:
        st.info("No decision history recorded for this ticker yet.", icon="ℹ️")
        return

    show_holds = st.checkbox("Include HOLD / WAIT records", value=False, key="tlt_show_holds")
    view = tl if show_holds else tl[~tl["action"].isin(["HOLD", "WAIT"])]
    if view.empty:
        st.info("Only HOLD/WAIT records exist — enable the checkbox above to see them.", icon="ℹ️")
        return

    st.caption(f"{len(view):,} events, oldest first. Source files are never modified.")
    max_cards = 60
    cards = view.tail(max_cards)
    if len(view) > max_cards:
        st.caption(f"Showing the most recent {max_cards} events (full history in the table below).")

    for _, r in cards.iterrows():
        action = str(r["action"])
        icon = _ACTION_ICONS.get(action, "•")
        ts = pd.Timestamp(r["timestamp"])
        date_str = ts.strftime("%Y-%m-%d %H:%M UTC") if not pd.isna(ts) else "—"
        bits = []
        conf = _safe_float(r.get("confidence"))
        if np.isfinite(conf):
            bits.append(f"confidence={conf:.2f}")
        qty = _safe_float(r.get("qty"))
        px = _safe_float(r.get("price"))
        if np.isfinite(qty) and qty > 0 and np.isfinite(px):
            bits.append(f"{qty:g} sh @ {px:,.2f}")
        rp = _safe_float(r.get("realized_pnl"))
        if np.isfinite(rp) and action in ("TRIM", "EXIT"):
            bits.append(f"realized_pnl={rp:+,.2f}")
        grade = str(r.get("decision_grade") or "")
        if grade:
            bits.append(f"grade={grade}")
        flags = str(r.get("risk_flags") or "").strip()
        if flags:
            bits.append(f"risk_flag={flags}")
        sess = str(r.get("execution_session") or "").strip()
        if sess and sess != "current_snapshot":
            bits.append(f"session={sess}")
        header = f"{icon} **{date_str} — {action}**"
        body = " · ".join(bits)
        rationale = str(r.get("rationale") or "").strip()
        st.markdown(header)
        if body:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{body}")
        if rationale:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🧠 *{rationale}*")
        st.markdown("<div style='margin-left:14px;color:#777'>↓</div>", unsafe_allow_html=True)

    with st.expander("Full event table", expanded=False):
        full = tl.copy()
        full["timestamp"] = full["timestamp"].astype(str)
        st.dataframe(full, use_container_width=True, hide_index=True)


def _render_rationale_evolution(tl: pd.DataFrame) -> None:
    st.markdown("#### 🧠 Rationale Evolution")
    rows = tl[
        (pd.to_numeric(tl["confidence"], errors="coerce").notna())
        | (tl["rationale"].astype(str).str.len() > 0)
    ].copy()
    if rows.empty:
        st.info("No rationale or confidence history available for this ticker.", icon="ℹ️")
        return

    rows = rows.sort_values("timestamp")
    rows["confidence"] = pd.to_numeric(rows["confidence"], errors="coerce")
    rows["expected_return"] = pd.to_numeric(rows["expected_return"], errors="coerce")

    conf = rows.dropna(subset=["confidence"])[["timestamp", "confidence"]]
    if len(conf) >= 2:
        chart = conf.copy()
        chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True).dt.tz_localize(None)
        if not _plotly_line(chart, "timestamp", "confidence", "Confidence over time"):
            st.line_chart(chart.set_index("timestamp"))

    # "What changed?" between consecutive decisions.
    changes: List[Dict[str, Any]] = []
    prev = None
    for _, r in rows.iterrows():
        if prev is not None:
            deltas = []
            dc = _safe_float(r["confidence"]) - _safe_float(prev["confidence"])
            if np.isfinite(dc) and abs(dc) >= 0.01:
                deltas.append(f"confidence {dc:+.2f}")
            de = _safe_float(r["expected_return"]) - _safe_float(prev["expected_return"])
            if np.isfinite(de) and abs(de) >= 0.05:
                deltas.append(f"expected_return {de:+.2f}")
            if str(r.get("signal") or "") != str(prev.get("signal") or "") and str(
                r.get("signal") or ""
            ):
                deltas.append(f"signal → {r.get('signal')}")
            r_txt, p_txt = (
                str(r.get("rationale") or "").strip(),
                str(prev.get("rationale") or "").strip(),
            )
            if r_txt and r_txt != p_txt:
                deltas.append("rationale updated")
            f_txt = str(r.get("risk_flags") or "").strip()
            if f_txt and f_txt != str(prev.get("risk_flags") or "").strip():
                deltas.append(f"risk flags → {f_txt}")
            if deltas:
                changes.append(
                    {
                        "timestamp": str(r["timestamp"]),
                        "action": r["action"],
                        "what_changed": "; ".join(deltas),
                        "confidence": r["confidence"],
                        "rationale": r_txt[:160],
                    }
                )
        prev = r
    if changes:
        st.markdown("**What changed between decisions**")
        st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
    else:
        st.caption("No measurable changes between recorded decisions.")


def _render_pnl_attribution(
    ticker: str, tl: pd.DataFrame, trades: pd.DataFrame, pos: Dict[str, Any], results_dir: Path
) -> None:
    st.markdown("#### 💵 P&L Attribution")
    if trades.empty and not pos:
        st.info("No realized trades or open position for this ticker.", icon="ℹ️")
        return

    realized = float(trades["realized_pl"].sum()) if not trades.empty else 0.0
    unreal = _safe_float(pos.get("unrealized_pl")) if pos else float("nan")
    wins = int((trades["realized_pl"] > 0).sum()) if not trades.empty else 0
    win_rate = wins / len(trades) * 100.0 if len(trades) else float("nan")
    best = float(trades["realized_pl"].max()) if not trades.empty else float("nan")
    worst = float(trades["realized_pl"].min()) if not trades.empty else float("nan")
    hold_days = (
        (trades["closed_at"] - trades["opened_at"]).dt.total_seconds().div(86400).dropna()
        if not trades.empty
        else pd.Series(dtype=float)
    )
    avg_hold = float(hold_days.mean()) if len(hold_days) else float("nan")

    fills = tl[tl["event_source"] == "fill_ledger"]
    qty = pd.to_numeric(fills["qty"], errors="coerce")
    px = pd.to_numeric(fills["price"], errors="coerce")
    notional = (qty * px).fillna(0.0)
    deployed = float(notional[fills["action"].isin(["BUY", "ADD"])].sum())
    returned = float(notional[fills["action"].isin(["TRIM", "EXIT"])].sum())

    # Contribution vs total portfolio realized P&L.
    all_t = _read_csv_safe(results_dir / "trade_outcomes.csv")
    contribution = float("nan")
    if not all_t.empty:
        all_t.columns = [str(c).strip().lower() for c in all_t.columns]
        all_real = all_t[all_t.get("outcome_type", "").astype(str).str.upper() == "REALIZED"]
        total_real = float(pd.to_numeric(all_real.get("realized_pl"), errors="coerce").sum())
        if abs(total_real) > 1e-9:
            contribution = realized / total_real * 100.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Realized P&L", _money(realized))
    c2.metric("Total Unrealized P&L", _money(unreal))
    c3.metric("Best Trade", _money(best))
    c4.metric("Worst Trade", _money(worst))
    c5.metric("Avg Hold Time", f"{avg_hold:.1f} d" if np.isfinite(avg_hold) else "—")
    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Win Rate", _pct(win_rate))
    c7.metric("Capital Deployed", _money(deployed))
    c8.metric("Capital Returned", _money(returned))
    c9.metric("Portfolio Contribution", _pct(contribution))

    if not trades.empty:
        col_l, col_r = st.columns(2)
        with col_l:
            cum = trades.dropna(subset=["closed_at"]).copy()
            cum["cumulative_pnl"] = cum["realized_pl"].cumsum()
            cum["closed_at"] = cum["closed_at"].dt.tz_localize(None)
            if not _plotly_line(cum, "closed_at", "cumulative_pnl", "Cumulative Realized P&L"):
                st.line_chart(cum.set_index("closed_at")["cumulative_pnl"])
        with col_r:
            by_trade = trades.reset_index().rename(columns={"index": "trade_n"})
            by_trade["trade_n"] = by_trade["trade_n"] + 1
            if not _plotly_bar(by_trade, "trade_n", "realized_pl", "P&L By Trade"):
                st.bar_chart(by_trade.set_index("trade_n")["realized_pl"])


def _render_decision_quality(tl: pd.DataFrame) -> None:
    st.markdown("#### 🎓 Decision Quality Review")
    graded = tl[tl["decision_grade"].astype(str).str.len() > 0].copy()
    if graded.empty:
        st.info(
            "No graded decisions yet (grades need price history after each decision).", icon="ℹ️"
        )
        return

    graded["points"] = graded["decision_grade"].map(_GRADE_POINTS)
    avg_pts = float(graded["points"].mean())
    accuracy = float((graded["points"] >= 2.0).mean() * 100.0)

    per_action = (
        graded.groupby("action")
        .agg(decisions=("points", "size"), avg_points=("points", "mean"))
        .reset_index()
    )
    per_action["avg_grade"] = per_action["avg_points"].map(_grade_letter)

    impact = pd.to_numeric(graded["realized_pnl"], errors="coerce").fillna(
        pd.to_numeric(graded["forward_return_5d"], errors="coerce")
    )
    graded = graded.assign(_impact=impact)
    worst_row = graded.loc[graded["_impact"].idxmin()] if graded["_impact"].notna().any() else None
    best_row = graded.loc[graded["_impact"].idxmax()] if graded["_impact"].notna().any() else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Decision Grade", _grade_letter(avg_pts))
    c2.metric("Decision Accuracy", _pct(accuracy))
    c3.metric(
        "Most Costly Decision",
        (
            f"{worst_row['action']} {pd.Timestamp(worst_row['timestamp']).date()}"
            if worst_row is not None
            else "—"
        ),
    )
    c4.metric(
        "Best Decision",
        (
            f"{best_row['action']} {pd.Timestamp(best_row['timestamp']).date()}"
            if best_row is not None
            else "—"
        ),
    )

    st.dataframe(
        per_action[["action", "decisions", "avg_grade"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Grades compare each decision against the subsequent 5-day price move: "
        "entries graded up when price rose, exits graded up when price fell."
    )


def _render_post_exit(tl: pd.DataFrame) -> None:
    st.markdown("#### 🚪 Post-Exit Analysis")
    exits = tl[(tl["action"] == "EXIT") & (tl["exit_outcome"].astype(str).str.len() > 0)].copy()
    if exits.empty:
        st.info("No exits recorded for this ticker.", icon="ℹ️")
        return

    view = exits[
        [
            "timestamp",
            "price",
            "realized_pnl",
            "post_exit_return_5d",
            "post_exit_return_10d",
            "post_exit_return_20d",
            "exit_outcome",
        ]
    ].copy()
    view["timestamp"] = view["timestamp"].astype(str)
    st.dataframe(view, use_container_width=True, hide_index=True)

    better = pd.to_numeric(exits["post_exit_return_10d"], errors="coerce") > 0
    n_better = int(better.sum())
    st.metric(
        "Would Holding Have Been Better?",
        f"YES for {n_better}/{len(exits)} exits" if n_better else f"NO ({len(exits)} exits)",
    )
    outcome_counts = exits["exit_outcome"].value_counts().reset_index()
    outcome_counts.columns = ["outcome", "count"]
    st.dataframe(outcome_counts, use_container_width=True, hide_index=True)
    st.caption(
        "CORRECT_EXIT: price fell ≥2% in 10 days · EARLY_EXIT: price rose ≥5% · "
        "LATE_EXIT: ≥5% run-up was given back before exiting · otherwise NEUTRAL_EXIT."
    )


# ──────────────────────────────────────────────
# Page entry point
# ──────────────────────────────────────────────
def render_ticker_lifecycle_tracker(results_dir: Path) -> None:
    st.markdown("### 🧬 Ticker Lifecycle Tracker")
    st.caption(
        "Forensic decision history per ticker — why TRITON bought, held, added, "
        "trimmed, and exited. Observability only; refresh the timeline with "
        "`python -m services.ticker_timeline_builder`."
    )

    timelines = _cached_timelines("v1")
    if timelines is None or timelines.empty:
        st.info(
            "No decision timelines available yet. Run "
            "`python -m services.ticker_timeline_builder` to build them.",
            icon="ℹ️",
        )
        return

    tickers = sorted(timelines["ticker"].dropna().astype(str).unique())
    if not tickers:
        st.info("Timeline file contains no tickers.", icon="ℹ️")
        return

    # Tickers with actual executions first.
    traded = sorted(
        timelines.loc[timelines["event_source"] == "fill_ledger", "ticker"].astype(str).unique()
    )
    ordered = traded + [t for t in tickers if t not in traded]
    ticker = st.selectbox("Select ticker", options=ordered, index=0, key="tlt_ticker")
    if not ticker:
        return

    tl = timelines[timelines["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
    pos = _position_for(results_dir, ticker)
    trades = _realized_trades_for(results_dir, ticker)

    _render_executive_summary(ticker, tl, pos, trades)
    st.divider()
    _render_current_status(ticker, tl, pos)
    st.divider()

    tabs = st.radio(
        "Section",
        [
            "🧬 Timeline",
            "🧠 Rationale Evolution",
            "💵 P&L Attribution",
            "🎓 Decision Quality",
            "🚪 Post-Exit",
        ],
        horizontal=True,
        key="tlt_section",
    )
    if tabs == "🧬 Timeline":
        _render_timeline(tl)
    elif tabs == "🧠 Rationale Evolution":
        _render_rationale_evolution(tl)
    elif tabs == "💵 P&L Attribution":
        _render_pnl_attribution(ticker, tl, trades, pos, results_dir)
    elif tabs == "🎓 Decision Quality":
        _render_decision_quality(tl)
    else:
        _render_post_exit(tl)
