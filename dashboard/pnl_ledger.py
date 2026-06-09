"""
TRITON Phase 164 — Profit & Loss Ledger Dashboard (read-only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from services.pnl_ledger_builder import PnlLedgerBundle, build_pnl_ledger


def _money(v: Any) -> str:
    try:
        x = float(v)
        if not np.isfinite(x):
            return "—"
        return f"${x:,.2f}"
    except Exception:
        return "—"


def _pct(v: Any) -> str:
    try:
        x = float(v)
        if not np.isfinite(x):
            return "—"
        return f"{x:.2f}%"
    except Exception:
        return "—"


def _num(v: Any) -> str:
    try:
        x = float(v)
        if not np.isfinite(x):
            return "—"
        return f"{x:,.2f}"
    except Exception:
        return str(v) if v not in (None, "") else "—"


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _color_pl(val: Any) -> str:
    try:
        x = float(val)
        if x > 0:
            return "color: #22c55e;"
        if x < 0:
            return "color: #ef4444;"
    except Exception:
        pass
    return ""


@st.cache_data(show_spinner=False, ttl=120)
def _cached_ledger(results_dir_str: str, fingerprint_seed: str) -> Dict[str, Any]:
    bundle = build_pnl_ledger(results_dir=Path(results_dir_str), force=False)
    return {
        "fingerprint": bundle.fingerprint,
        "generated_at": bundle.generated_at,
        "ledger": bundle.ledger,
        "ledger_metrics": bundle.ledger_metrics,
        "realized_summary": bundle.realized_summary,
        "daily_realized": bundle.daily_realized,
        "open_positions": bundle.open_positions,
        "open_metrics": bundle.open_metrics,
        "cash_flow_summary": bundle.cash_flow_summary,
        "cash_flow_series": bundle.cash_flow_series,
        "attribution": bundle.attribution,
        "exit_analysis": bundle.exit_analysis,
        "exit_metrics": bundle.exit_metrics,
        "equity_curve": bundle.equity_curve,
        "equity_metrics": bundle.equity_metrics,
        "scorecard": bundle.scorecard,
        "reconciliation": bundle.reconciliation,
        "has_realized_trades": bundle.has_realized_trades,
        "has_open_positions": bundle.has_open_positions,
        "has_pnl_history": bundle.has_pnl_history,
    }


def _load_bundle(results_dir: Path) -> PnlLedgerBundle:
    raw = _cached_ledger(str(results_dir.resolve()), "v1")
    return PnlLedgerBundle(**raw)


def _plotly_bar(df: pd.DataFrame, x: str, y: str, title: str) -> bool:
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return False
    try:
        import plotly.express as px

        fig = px.bar(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
        return True
    except Exception:
        return False


def _plotly_line(df: pd.DataFrame, x: str, y: str, title: str) -> bool:
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return False
    try:
        import plotly.express as px

        fig = px.line(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
        return True
    except Exception:
        return False


def _render_ledger_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 💰 P&L Ledger")
    if not bundle.has_realized_trades:
        st.info("No realized trades yet.", icon="ℹ️")
        return

    m = bundle.ledger_metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", f"{int(m.get('total_trades', 0)):,}")
    c2.metric("Win Rate", _pct(m.get("win_rate_pct")))
    c3.metric("Profit Factor", _num(m.get("profit_factor")))
    c4.metric("Net Realized", _money(bundle.realized_summary.get("total_realized_pnl")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Winning Trades", f"{int(m.get('winning_trades', 0)):,}")
    c6.metric("Losing Trades", f"{int(m.get('losing_trades', 0)):,}")
    c7.metric("Avg Winner", _money(m.get("avg_winner")))
    c8.metric("Avg Loser", _money(m.get("avg_loser")))

    c9, c10, _ = st.columns(3)
    c9.metric("Largest Winner", _money(m.get("largest_winner")))
    c10.metric("Largest Loser", _money(m.get("largest_loser")))

    view = bundle.ledger.copy()
    for col in ("entry_date", "exit_date"):
        if col in view.columns:
            view[col] = pd.to_datetime(view[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    for col in ("entry_price", "exit_price", "gross_pnl", "net_pnl", "fees"):
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(
                lambda x: f"{x:,.2f}" if np.isfinite(x) else "—"
            )
    if "gross_pnl_pct" in view.columns:
        view["gross_pnl_pct"] = pd.to_numeric(view["gross_pnl_pct"], errors="coerce").map(
            lambda x: f"{x:.2f}%" if np.isfinite(x) else "—"
        )

    max_rows = st.slider("Max rows displayed", 50, 5000, 500, step=50, key="pnl_ledger_rows")
    st.caption(f"Showing {min(len(view), max_rows):,} of {len(view):,} completed trades")
    st.dataframe(view.head(max_rows), use_container_width=True, hide_index=True)


def _render_realized_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 📊 Realized Performance")
    if not bundle.has_realized_trades:
        st.info("No realized trades yet.", icon="ℹ️")
        return

    s = bundle.realized_summary
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Realized P&L", _money(s.get("total_realized_pnl")))
    c2.metric("Daily", _money(s.get("daily_realized_pnl")))
    c3.metric("Weekly", _money(s.get("weekly_realized_pnl")))
    c4.metric("Monthly", _money(s.get("monthly_realized_pnl")))
    c5.metric("YTD", _money(s.get("ytd_realized_pnl")))

    daily = bundle.daily_realized.copy()
    if daily.empty:
        st.info("No P&L history available.", icon="ℹ️")
        return

    if not _plotly_line(daily, "date", "cumulative_pnl", "Cumulative Realized P&L"):
        st.line_chart(daily.set_index("date")[["cumulative_pnl"]])

    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("Daily P&L")
        if not _plotly_bar(daily, "date", "daily_pnl", "Daily Realized P&L"):
            st.bar_chart(daily.set_index("date")[["daily_pnl"]])
    with col_r:
        st.caption("Monthly P&L")
        monthly = daily.copy()
        monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").astype(str)
        m_agg = monthly.groupby("month", as_index=False).agg(monthly_pnl=("daily_pnl", "sum"))
        if not _plotly_bar(m_agg, "month", "monthly_pnl", "Monthly Realized P&L"):
            st.bar_chart(m_agg.set_index("month"))

    st.caption("Trade Outcome Histogram")
    pnl = pd.to_numeric(bundle.ledger["net_pnl"], errors="coerce").dropna()
    if pnl.empty:
        st.caption("No trade outcomes to chart.")
    else:
        hist_df = pnl.rename("net_pnl").to_frame()
        if not _plotly_bar(hist_df.reset_index(drop=True), "net_pnl", "net_pnl", ""):
            try:
                counts, bins = np.histogram(pnl, bins=30)
                h = pd.DataFrame({"bucket_mid": (bins[:-1] + bins[1:]) / 2, "count": counts})
                st.bar_chart(h.set_index("bucket_mid"))
            except Exception:
                st.caption("Histogram unavailable.")


def _render_open_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 📈 Open Positions")
    if not bundle.has_open_positions:
        st.info("No open positions in snapshot.", icon="ℹ️")
        return

    om = bundle.open_metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Unrealized P&L", _money(om.get("total_unrealized_pnl")))
    c2.metric("Largest Winner", _money(om.get("largest_winner")))
    c3.metric("Largest Loser", _money(om.get("largest_loser")))

    view = bundle.open_positions.copy()
    disp_cols = [
        "ticker",
        "quantity",
        "market_value",
        "cost_basis",
        "current_price",
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "position_weight",
        "days_held",
    ]
    view = view[[c for c in disp_cols if c in view.columns]]
    try:
        sty = view.style.applymap(
            _color_pl, subset=[c for c in ("unrealized_pnl",) if c in view.columns]
        )
        st.dataframe(sty, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(view, use_container_width=True, hide_index=True)

    st.caption("Exposure by Asset")
    exp = (
        view[["ticker", "market_value"]].dropna()
        if "market_value" in view.columns
        else pd.DataFrame()
    )
    if not exp.empty:
        if not _plotly_bar(exp, "ticker", "market_value", "Market Value by Ticker"):
            st.bar_chart(exp.set_index("ticker"))


def _render_cash_flow_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 💵 Cash Flow")
    cf = bundle.cash_flow_summary
    if not bundle.has_pnl_history and not bundle.has_realized_trades:
        st.info("No P&L history available.", icon="ℹ️")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opening Cash", _money(cf.get("opening_cash")))
    c2.metric("Capital Deployed", _money(cf.get("capital_deployed")))
    c3.metric("Capital Returned", _money(cf.get("capital_returned")))
    c4.metric("Cash Available", _money(cf.get("cash_available")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Realized Gains", _money(cf.get("realized_gains")))
    c6.metric("Realized Losses", _money(cf.get("realized_losses")))
    c7.metric("Current Equity", _money(cf.get("current_equity")))
    c8.metric("Net Account Change", _money(cf.get("net_account_change")))

    recon_cash = cf.get("cash_reconstructed")
    if recon_cash is not None and np.isfinite(_safe_float(recon_cash)):
        st.caption(
            f"Capital Deployed / Returned come from the actual fill ledger "
            f"(live_orders_log + recent_orders, deduplicated per order). "
            f"Reconstructed end cash from fills: **{_money(recon_cash)}** vs official snapshot "
            f"**{_money(cf.get('cash_available'))}** — any gap is quantified in the "
            f"🧾 Reconciliation section."
        )

    series = bundle.cash_flow_series.copy()
    if series.empty:
        st.caption("Cash balance timeline unavailable.")
        return

    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("Cash Balance Over Time")
        if "cash" in series.columns:
            if not _plotly_line(series, "date", "cash", "Cash Balance"):
                st.line_chart(series.set_index("date")[["cash"]])
    with col_r:
        st.caption("Equity Over Time")
        if "equity" in series.columns:
            if not _plotly_line(series, "date", "equity", "Portfolio Equity"):
                st.line_chart(series.set_index("date")[["equity"]])

    if "capital_deployed_cum" in series.columns:
        st.caption("Capital Deployment Timeline")
        st.line_chart(series.set_index("date")[["capital_deployed_cum", "capital_returned_cum"]])


def _render_attribution_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 🎯 Trade Attribution")
    if not bundle.has_realized_trades:
        st.info("No realized trades yet.", icon="ℹ️")
        return

    attr = bundle.attribution
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Sector", str(attr.get("best_sector", "—")))
    c2.metric("Worst Sector", str(attr.get("worst_sector", "—")))
    c3.metric("Best Asset Class", str(attr.get("best_asset_class", "—")))
    c4.metric("Worst Asset Class", str(attr.get("worst_asset_class", "—")))

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Top 10 Winners**")
        tw = attr.get("top_winners")
        if isinstance(tw, pd.DataFrame) and not tw.empty:
            st.dataframe(tw, use_container_width=True, hide_index=True)
        else:
            st.caption("—")
    with col_r:
        st.markdown("**Top 10 Losers**")
        tl = attr.get("top_losers")
        if isinstance(tl, pd.DataFrame) and not tl.empty:
            st.dataframe(tl, use_container_width=True, hide_index=True)
        else:
            st.caption("—")

    st.markdown("**Contribution By Symbol**")
    by_sym = attr.get("by_symbol")
    if isinstance(by_sym, pd.DataFrame) and not by_sym.empty:
        if not _plotly_bar(by_sym.head(20), "ticker", "net_pnl", "P&L by Symbol"):
            st.bar_chart(by_sym.head(20).set_index("ticker"))

    st.markdown("**Contribution By Strategy (Lifecycle Action)**")
    by_strat = attr.get("by_strategy")
    if isinstance(by_strat, pd.DataFrame) and not by_strat.empty:
        st.dataframe(by_strat, use_container_width=True, hide_index=True)
    else:
        st.caption("No strategy attribution available.")

    st.markdown("**Contribution By Exit Reason**")
    by_exit = attr.get("by_exit_reason")
    if isinstance(by_exit, pd.DataFrame) and not by_exit.empty:
        st.dataframe(by_exit, use_container_width=True, hide_index=True)
    else:
        st.caption("No exit reason attribution available.")


def _render_exit_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 🚪 Exit Analysis")
    if bundle.exit_analysis.empty:
        st.info("No exits available.", icon="ℹ️")
        return

    em = bundle.exit_metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Exit Grade", str(em.get("average_exit_grade", "—")))
    c2.metric("Exit Accuracy", _pct(em.get("exit_accuracy_pct")))
    c3.metric("Capital Saved", _money(em.get("capital_saved")))
    c4.metric("Lost to Premature Exits", _money(em.get("capital_lost_premature")))

    view = bundle.exit_analysis.copy()
    if "exit_date" in view.columns:
        view["exit_date"] = pd.to_datetime(view["exit_date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption(
        "Grades: **A** = protected capital · **B** = reasonable · **C** = neutral · "
        "**D** = premature · **F** = harmful. Post-exit performance uses processed price history (+10 sessions)."
    )


def _render_equity_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 📉 Equity Curve")
    if bundle.equity_curve.empty:
        st.info("No P&L history available.", icon="ℹ️")
        return

    em = bundle.equity_metrics
    src = str(em.get("equity_source") or "")
    if src == "realized_pnl_anchored":
        st.caption(
            "Source: **initial equity + cumulative realized P&L** (reconciled with "
            "`trade_outcomes.csv`). `portfolio_history.csv` has too few snapshot days "
            "to chart directly; this curve excludes intraday unrealized swings."
        )
    elif src == "portfolio_history":
        st.caption("Source: `portfolio_history.csv` daily totals.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Equity", _money(em.get("current_equity")))
    c2.metric("Peak Equity", _money(em.get("peak_equity")))
    c3.metric("Current Drawdown", _pct(em.get("current_drawdown_pct")))
    c4.metric("Max Drawdown", _pct(em.get("max_drawdown_pct")))
    c5.metric("Recovery Factor", _num(em.get("recovery_factor")))

    eq = bundle.equity_curve.copy()
    if not _plotly_line(eq, "date", "equity", "Portfolio Equity Over Time"):
        st.line_chart(eq.set_index("date")[["equity"]])

    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("Drawdown Curve")
        if "drawdown_pct" in eq.columns:
            if not _plotly_line(eq, "date", "drawdown_pct", "Drawdown (%)"):
                st.line_chart(eq.set_index("date")[["drawdown_pct"]])
    with col_r:
        st.caption("Rolling Returns (30d)")
        if "rolling_return_30d" in eq.columns:
            if not _plotly_line(eq, "date", "rolling_return_30d", "Rolling 30-Day Return (%)"):
                st.line_chart(eq.set_index("date")[["rolling_return_30d"]])

    dd = eq[eq["drawdown_pct"] < -0.01].copy() if "drawdown_pct" in eq.columns else pd.DataFrame()
    if not dd.empty:
        st.markdown("**Peak-to-Trough Drawdowns (sample)**")
        st.dataframe(dd.tail(20), use_container_width=True, hide_index=True)


def _render_reconciliation_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 🧾 Reconciliation")
    st.caption(
        "Every-dollar audit: fill-ledger flows vs realized P&L vs broker positions. "
        "Identity: **Deployed − Returned = Open Cost Basis − Realized P&L**."
    )
    r = bundle.reconciliation
    if not r:
        st.info("No reconciliation data available.", icon="ℹ️")
        return

    # 1) Cross-artifact realized P&L agreement
    st.markdown("**Realized P&L — cross-artifact agreement**")
    sources = r.get("realized_sources") or []
    if sources:
        src_df = pd.DataFrame(sources)
        src_df["realized_pl"] = pd.to_numeric(src_df["realized_pl"], errors="coerce").map(
            lambda x: f"{x:,.2f}" if np.isfinite(x) else "—"
        )
        st.dataframe(src_df, use_container_width=True, hide_index=True)
    if r.get("realized_match"):
        st.success("All P&L artifacts agree on total realized P&L (within $1).", icon="✅")
    else:
        st.error(
            "P&L artifacts DISAGREE on realized P&L — investigate before trusting totals.",
            icon="🚨",
        )

    # 2) Cash identity check
    st.markdown("**Cash identity check (fill ledger)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital Deployed", _money(r.get("capital_deployed")))
    c2.metric("Capital Returned", _money(r.get("capital_returned")))
    c3.metric("Net Invested", _money(r.get("net_invested")))
    c4.metric("Expected (Open − Realized)", _money(r.get("expected_net_invested")))

    residual = _safe_float(r.get("residual"))
    if r.get("identity_ok"):
        st.success(f"Identity holds — residual {_money(residual)}.", icon="✅")
    else:
        st.warning(
            f"Identity residual: **{_money(residual)}** unaccounted. "
            + str(r.get("residual_explanation") or ""),
            icon="⚠️",
        )

    # 3) Coverage
    st.markdown("**Fill-ledger coverage**")
    c5, c6, c7 = st.columns(3)
    c5.metric("Buy fills", f"{int(r.get('fill_rows_buy', 0)):,}")
    c6.metric("Sell fills", f"{int(r.get('fill_rows_sell', 0)):,}")
    match_pct = _safe_float(r.get("exit_match_pct"))
    c7.metric(
        "Exits matched to fills",
        f"{int(r.get('exits_matched_to_fills', 0)):,} / {int(r.get('exits_total', 0)):,}"
        + (f" ({match_pct:.0f}%)" if np.isfinite(match_pct) else ""),
    )
    st.caption(
        "Sources: `live_orders_log.csv`, `recent_orders.csv`, `trade_outcomes.csv`, "
        "`positions_snapshot.csv`, `pnl_diagnostics.json`, `performance_intelligence.csv`. "
        "Note: `portfolio_history.csv` currently holds a single snapshot day and "
        "`trade_log.csv` contains simulation BUY rows only — neither is used for dollar flows."
    )


def _render_scorecard_tab(bundle: PnlLedgerBundle) -> None:
    st.markdown("#### 🏆 Operator Scorecard")
    sc = bundle.scorecard
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Realized P&L", _money(sc.get("total_realized_pnl")))
    c2.metric("Total Unrealized P&L", _money(sc.get("total_unrealized_pnl")))
    c3.metric("Combined P&L", _money(sc.get("combined_pnl")))
    c4.metric("Win Rate", _pct(sc.get("win_rate_pct")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Profit Factor", _num(sc.get("profit_factor")))
    c6.metric("Sharpe Ratio", _num(sc.get("sharpe_ratio")))
    c7.metric("Capital Efficiency", _num(sc.get("capital_efficiency")))
    c8.metric("Current CPS", _num(sc.get("current_cps")))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Best Trade", str(sc.get("best_trade", "—")))
    c10.metric("Worst Trade", str(sc.get("worst_trade", "—")))
    c11.metric("Most Profitable Symbol", str(sc.get("most_profitable_symbol", "—")))
    c12.metric("Least Profitable Symbol", str(sc.get("least_profitable_symbol", "—")))

    st.metric("Portfolio Health", str(sc.get("portfolio_health", "—")))


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _read_json_safe(path: Path) -> Dict[str, Any]:
    import json

    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def render_ledger_reconciliation_center(results_dir: Path) -> None:
    """💳 Ledger Reconciliation Center — Phase 165B broker truth vs local records."""
    st.markdown("### 💳 Ledger Reconciliation Center")
    st.caption(
        "Broker history recovery and ledger healing (Phase 165B). Read-only audit — "
        "no records are rewritten. Refresh artifacts with "
        "`python -m services.broker_history_recovery`."
    )

    integrity = _read_json_safe(results_dir / "ledger_integrity_report.json")
    healing = _read_csv_safe(results_dir / "ledger_healing_report.csv")
    residual = _read_csv_safe(results_dir / "residual_breakdown.csv")
    pos_recon = _read_json_safe(results_dir / "position_reconciliation_summary.json")
    recovered = _read_csv_safe(results_dir / "recovered_fills.csv")
    recovered_pos = _read_csv_safe(results_dir / "recovered_positions.csv")

    if not integrity and healing.empty and recovered.empty:
        st.info(
            "No reconciliation artifacts yet. Run "
            "`python -m services.broker_history_recovery` to recover broker history.",
            icon="ℹ️",
        )
        return

    # ── Headline metrics ──
    status = str(integrity.get("status") or "—")
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        status, "⚪"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Integrity Score", _num(integrity.get("ledger_integrity_score")))
    c2.metric("Certification", f"{badge} {status}")
    c3.metric("Residual Difference", _money(integrity.get("residual_difference")))
    c4.metric(
        "Identity",
        "✅ HOLDS" if integrity.get("identity_ok") else "⚠️ OPEN",
    )

    c5, c6, c7, c8, c9 = st.columns(5)
    c5.metric("Fill Match", _pct(integrity.get("fill_match")))
    c6.metric("Position Match", _pct(integrity.get("position_match")))
    c7.metric("Cash Match", _pct(integrity.get("cash_match")))
    c8.metric("P&L Match", _pct(integrity.get("pnl_match")))
    c9.metric("Trade Match", _pct(integrity.get("trade_match")))

    prev_epoch = _safe_float(integrity.get("previous_epoch_net_flow"))
    if np.isfinite(prev_epoch) and abs(prev_epoch) > 1:
        st.warning(
            f"**Previous account epoch detected** — the current paper account was created "
            f"{str(integrity.get('account_created') or '')[:10]}. Net "
            f"**{_money(prev_epoch)}** of local fill flow predates it and cannot be "
            f"certified against current broker history.",
            icon="🕰️",
        )

    reasons = integrity.get("reasons") or []
    if reasons:
        with st.expander("Certification reasons", expanded=False):
            for r in reasons:
                st.markdown(f"- {r}")

    # ── Residual sources ──
    st.markdown("#### Residual Sources")
    if residual.empty:
        st.info("No residual breakdown available.", icon="ℹ️")
    else:
        res = residual.copy()
        res["dollar_impact"] = pd.to_numeric(res["dollar_impact"], errors="coerce")
        if "is_root_cause" in res.columns:
            causes = res[res["is_root_cause"] == True]  # noqa: E712
            st.caption(
                f"Root-cause dollars: **{_money(causes['dollar_impact'].sum())}** "
                f"(phantom-lot rows are downstream symptoms of the same dollars and are "
                f"excluded from the sum)."
            )
        st.markdown("**Top 10 discrepancy sources**")
        top10 = res.reindex(res["dollar_impact"].abs().sort_values(ascending=False).index).head(10)
        st.dataframe(top10, use_container_width=True, hide_index=True)

        col_l, col_r = st.columns(2)
        with col_l:
            by_sym = res.groupby("symbol", as_index=False)["dollar_impact"].sum()
            by_sym = by_sym.reindex(
                by_sym["dollar_impact"].abs().sort_values(ascending=False).index
            ).head(15)
            if not by_sym.empty:
                st.caption("Residual by Symbol")
                if not _plotly_bar(by_sym, "symbol", "dollar_impact", "Residual by Symbol"):
                    st.bar_chart(by_sym.set_index("symbol"))
        with col_r:
            if "date" in res.columns and res["date"].astype(str).str.len().gt(0).any():
                by_date = res.groupby("date", as_index=False)["dollar_impact"].sum()
                st.caption("Residual by Date")
                if not _plotly_bar(by_date, "date", "dollar_impact", "Residual by Date"):
                    st.bar_chart(by_date.set_index("date"))
        if "session" in res.columns and res["session"].astype(str).str.len().gt(0).any():
            by_sess = res.groupby("session", as_index=False)["dollar_impact"].sum()
            by_sess = by_sess.reindex(
                by_sess["dollar_impact"].abs().sort_values(ascending=False).index
            ).head(15)
            st.caption("Residual by Session")
            if not _plotly_bar(by_sess, "session", "dollar_impact", "Residual by Session"):
                st.bar_chart(by_sess.set_index("session"))

    # ── Discrepancies & healing recommendations ──
    st.markdown("#### Discrepancy Table & Healing Recommendations")
    if healing.empty:
        st.info("No discrepancies recorded — ledger healing report is empty.", icon="ℹ️")
    else:
        sev_opts = [
            s for s in ("CRITICAL", "HIGH", "WARNING", "INFO") if s in set(healing["severity"])
        ]
        pick = st.multiselect(
            "Severity filter", options=sev_opts, default=sev_opts, key="lrc_severity"
        )
        view = healing[healing["severity"].isin(pick)] if pick else healing
        st.caption(f"{len(view):,} issues shown")
        st.dataframe(view, use_container_width=True, hide_index=True)
        fix_counts = healing.groupby("recommended_fix", as_index=False).agg(
            issues=("recommended_fix", "count"),
            dollars=("dollar_impact", "sum"),
        )
        st.markdown("**Recommended fixes summary**")
        st.dataframe(fix_counts, use_container_width=True, hide_index=True)

    # ── Position reconciliation ──
    st.markdown("#### Position Reconciliation")
    if not pos_recon:
        st.info("No position reconciliation summary available.", icon="ℹ️")
    else:
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Checked", str(pos_recon.get("symbols_checked", 0)))
        p2.metric("Perfect", str(pos_recon.get("perfect_matches", 0)))
        p3.metric("Qty Mismatch", str(pos_recon.get("quantity_mismatches", 0)))
        p4.metric("Cost Mismatch", str(pos_recon.get("cost_basis_mismatches", 0)))
        p5.metric("Missing", str(pos_recon.get("missing_positions", 0)))
        p6.metric("Phantom", str(pos_recon.get("phantom_positions", 0)))
        details = pos_recon.get("details") or []
        if details:
            with st.expander("Position differences", expanded=False):
                st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

    if not recovered_pos.empty:
        with st.expander("Recovered positions (reconstructed from broker fills)", expanded=False):
            st.dataframe(recovered_pos, use_container_width=True, hide_index=True)

    # ── Recovered fills preview ──
    st.markdown("#### Recovered Fills Preview")
    if recovered.empty:
        st.info("No recovered fills available.", icon="ℹ️")
    else:
        st.caption(
            f"{len(recovered):,} canonical fills · source: "
            f"{recovered['source'].iloc[0] if 'source' in recovered.columns else '—'}"
        )
        st.dataframe(recovered.tail(200), use_container_width=True, hide_index=True)


def render_pnl_ledger_dashboard(results_dir: Path) -> None:
    st.markdown("### 💰 Profit & Loss Ledger")
    st.caption(
        "Institutional-grade P&L and cash-flow accountability from **actual trade logs, fills, "
        "positions, and portfolio history**. Read-only — does not modify trading or execution."
    )

    try:
        bundle = _load_bundle(results_dir)
    except Exception as e:
        st.error(f"P&L Ledger failed to load: {e}")
        return

    st.caption(f"Data fingerprint `{bundle.fingerprint[:8]}` · Built {bundle.generated_at} (UTC)")

    sections = [
        "💰 P&L Ledger",
        "📊 Realized Performance",
        "📈 Open Positions",
        "💵 Cash Flow",
        "🎯 Trade Attribution",
        "🚪 Exit Analysis",
        "📉 Equity Curve",
        "🏆 Operator Scorecard",
        "🧾 Reconciliation",
    ]
    choice = st.radio(
        "Section",
        sections,
        horizontal=True,
        key="pnl_ledger_section",
        label_visibility="collapsed",
    )

    if choice == sections[0]:
        _render_ledger_tab(bundle)
    elif choice == sections[1]:
        _render_realized_tab(bundle)
    elif choice == sections[2]:
        _render_open_tab(bundle)
    elif choice == sections[3]:
        _render_cash_flow_tab(bundle)
    elif choice == sections[4]:
        _render_attribution_tab(bundle)
    elif choice == sections[5]:
        _render_exit_tab(bundle)
    elif choice == sections[6]:
        _render_equity_tab(bundle)
    elif choice == sections[7]:
        _render_scorecard_tab(bundle)
    else:
        _render_reconciliation_tab(bundle)
