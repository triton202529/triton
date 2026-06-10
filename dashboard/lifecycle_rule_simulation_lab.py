"""
TRITON Phase 169 — 🧪 Lifecycle Rule Simulation Lab (read-only dashboard).

Displays the replay / what-if results for the Phase 168 recommendations:
baseline replay, six simulated scenarios, exit-timing detail, scenario
rankings, certification, and the Phase 170 recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _money, _num, _pct, _plotly_bar


def _read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _table(records: Any) -> pd.DataFrame:
    try:
        return pd.DataFrame(records or [])
    except Exception:
        return pd.DataFrame()


_CLASS_BADGE = {
    "PROMISING": "🟢 PROMISING",
    "NEEDS_MORE_DATA": "🟡 NEEDS_MORE_DATA",
    "REJECT": "🟠 REJECT",
    "DANGEROUS": "🔴 DANGEROUS",
}


def _scenario_by_name(report: Dict[str, Any], name: str) -> Dict[str, Any]:
    for s in report.get("scenarios") or []:
        if s.get("scenario") == name:
            return s
    return {}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    baseline = report.get("baseline") or {}
    rankings = _table(report.get("scenario_rankings"))
    cert = report.get("certification") or {}
    rec = report.get("phase_170_recommendation") or {}

    best_delta = np.nan
    best_name = "—"
    if not rankings.empty:
        scored = rankings.dropna(subset=["simulated_pnl_delta"])
        if not scored.empty:
            best = scored.loc[
                pd.to_numeric(scored["simulated_pnl_delta"], errors="coerce").idxmax()
            ]
            best_delta = float(best["simulated_pnl_delta"])
            best_name = str(best["scenario"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Realized P&L", _money(baseline.get("realized_pnl")))
    c2.metric("Best Scenario", best_name)
    c3.metric("Best Scenario Gain", _money(best_delta))
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        str(cert.get("status")), "⚪"
    )
    c4.metric("Certification", f"{badge} {cert.get('status', '—')}")
    st.markdown(f"**Phase 170:** {rec.get('headline') or '—'}")
    st.caption(str(report.get("method_notes") or ""))


def _render_baseline(baseline: Dict[str, Any]) -> None:
    st.markdown("#### 📐 Baseline Replay")
    if not baseline:
        st.info("No baseline available.", icon="ℹ️")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Realized P&L", _money(baseline.get("realized_pnl")))
    c2.metric("Win Rate", _pct(baseline.get("win_rate")))
    c3.metric("Trades", f"{baseline.get('trade_count', 0):,}")
    c4.metric("Decision Alpha", _num(baseline.get("decision_alpha")))
    c5, c6, c7 = st.columns(3)
    c5.metric("Capital Deployed", _money(baseline.get("capital_deployed")))
    c6.metric("Drawdown Proxy", _pct(baseline.get("drawdown_proxy_pct")))
    c7.metric("Events Replayed", f"{baseline.get('events_replayed', 0):,}")
    outcomes = baseline.get("exit_outcomes") or {}
    if outcomes:
        st.markdown("**Actual exit outcomes**")
        st.dataframe(
            pd.DataFrame([{"outcome": k, "count": v} for k, v in outcomes.items()]),
            use_container_width=True,
            hide_index=True,
        )
    st.caption("Baseline matches Phase 167/168 metrics (validated in certification).")


def _render_scenarios(report: Dict[str, Any]) -> None:
    st.markdown("#### 🧪 Scenario Results")
    df = _table(report.get("scenarios"))
    if df.empty:
        st.info("No scenarios simulated.", icon="ℹ️")
        return
    keep = [
        c
        for c in (
            "scenario",
            "description",
            "simulated_pnl_delta",
            "impacted_trades",
            "false_positive_risk",
            "capital_protected",
            "capital_lost",
            "rule_confidence",
        )
        if c in df.columns
    ]
    st.dataframe(df[keep], use_container_width=True, hide_index=True)
    chart = df.dropna(subset=["simulated_pnl_delta"])
    if not chart.empty and not _plotly_bar(
        chart, "scenario", "simulated_pnl_delta", "Simulated P&L Delta vs Baseline"
    ):
        st.bar_chart(chart.set_index("scenario")["simulated_pnl_delta"])


def _render_exit_timing(report: Dict[str, Any]) -> None:
    st.markdown("#### 🚪 Exit Timing Simulation")
    block = report.get("exit_timing_simulation") or {}
    table = _table(block.get("table"))
    if table.empty:
        st.info("No exits to simulate.", icon="ℹ️")
        return
    s1 = _scenario_by_name(report, "EARLIER_EXIT_5D")
    s2 = _scenario_by_name(report, "EARLIER_EXIT_PEAK_DRAWDOWN")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Late Exits", str(block.get("late_exit_count", 0)))
    c2.metric("Exit 5d Earlier Gain", _money(s1.get("simulated_pnl_delta")))
    c3.metric("Peak-Drawdown Gain", _money(s2.get("simulated_pnl_delta")))
    c4.metric(
        "Improved / Premature",
        f"{int((table['verdict'] == 'IMPROVED').sum())} / {int((table['verdict'] == 'PREMATURE').sum())}",
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    benefits = _table(block.get("symbols_benefiting_most"))
    if not benefits.empty:
        st.markdown("**Symbols benefiting most**")
        if not _plotly_bar(benefits, "symbol", "best_delta", "Best Exit-Rule Benefit by Symbol"):
            st.bar_chart(benefits.set_index("symbol")["best_delta"])


def _render_confidence(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Confidence Filter Simulation (0.60–0.80)")
    s = _scenario_by_name(report, "MID_CONFIDENCE_FILTER")
    if not s:
        st.info("Scenario not simulated.", icon="ℹ️")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisions In Zone", f"{s.get('impacted_trades', 0):,}")
    c2.metric("Missed Winners", str(s.get("missed_winners", "—")))
    c3.metric("Avoided Losers", str(s.get("avoided_losers", "—")))
    c4.metric("Net P&L Impact", _money(s.get("simulated_pnl_delta")))
    st.metric("Net Alpha Impact (per decision)", _pct(s.get("net_alpha_impact_pct")))
    st.caption(str(s.get("method") or ""))
    fp = s.get("false_positive_risk")
    if fp is not None and fp > 50:
        st.warning(
            f"Blocking this zone outright misfires {fp}% of the time — the data favors "
            "extra confirmation or reduced sizing over a hard block.",
            icon="⚠️",
        )


def _render_hold_add(report: Dict[str, Any]) -> None:
    st.markdown("#### ➕ HOLD → ADD Reconfirmation Simulation")
    s = _scenario_by_name(report, "HOLD_ADD_RECONFIRMATION")
    if not s:
        st.info("Scenario not simulated.", icon="ℹ️")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HOLD → ADD Total", str(s.get("total_hold_add", "—")))
    c2.metric("Blocked Adds", str(s.get("blocked_adds", "—")))
    c3.metric("Avoided Losses", str(s.get("avoided_losses", "—")))
    c4.metric("Missed Gains", str(s.get("missed_gains", "—")))
    st.metric("Net Impact", _money(s.get("simulated_pnl_delta")))
    st.caption(str(s.get("method") or ""))


def _render_risk_flags(report: Dict[str, Any]) -> None:
    st.markdown("#### 🚩 Risk Flag Acceleration Simulation")
    s = _scenario_by_name(report, "RISK_FLAG_ACCELERATION")
    if not s:
        st.info("Scenario not simulated.", icon="ℹ️")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flagged Events", f"{s.get('impacted_trades', 0):,}")
    c2.metric("Capital Protected", _money(s.get("capital_protected")))
    c3.metric("Missed Recovery", _money(s.get("capital_lost")))
    c4.metric("Net Effect", _money(s.get("simulated_pnl_delta")))
    c5, c6 = st.columns(2)
    c5.metric("Correct Predictions", str(s.get("correct_predictions", "—")))
    c6.metric("False Exits", str(s.get("false_exits", "—")))
    note = str(s.get("note") or "")
    if note:
        st.warning(note, icon="⚠️")


def _render_rankings(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏁 Scenario Rankings")
    df = _table(report.get("scenario_rankings"))
    if df.empty:
        st.info("No rankings available.", icon="ℹ️")
        return
    view = df.copy()
    view["classification"] = view["classification"].map(lambda c: _CLASS_BADGE.get(str(c), str(c)))
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption(
        "Rank score = 40% P&L improvement + 25% low false-positive risk + "
        "20% rule confidence + 15% simplicity."
    )


def _render_phase170(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Recommendation For Phase 170")
    rec = report.get("phase_170_recommendation") or {}
    cert = report.get("certification") or {}
    if not rec:
        st.info("No recommendation generated.", icon="ℹ️")
        return
    st.markdown(f"**{rec.get('headline') or '—'}**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**✅ Implement**")
        for s in rec.get("implement") or ["—"]:
            st.markdown(f"- {s}")
    with col_b:
        st.markdown("**🟡 Collect more data**")
        for s in rec.get("collect_more_data") or ["—"]:
            st.markdown(f"- {s}")
    with col_c:
        st.markdown("**🚫 Do not implement**")
        for s in rec.get("do_not_implement") or ["—"]:
            st.markdown(f"- {s}")
    st.caption(str(rec.get("caveat") or ""))

    st.markdown("**Certification**")
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        str(cert.get("status")), "⚪"
    )
    st.metric("Status", f"{badge} {cert.get('status', '—')}")
    for r in cert.get("reasons") or []:
        st.warning(r, icon="⚠️")
    st.success(str(cert.get("safety") or "Simulation-only phase."), icon="🔒")


def render_lifecycle_rule_simulation_lab(results_dir: Path) -> None:
    st.markdown("### 🧪 Lifecycle Rule Simulation Lab")
    st.caption(
        "Replay / what-if testing of Phase 168 recommendations against historical "
        "prices — simulation only, nothing operational changes. Refresh with "
        "`python -m services.lifecycle_rule_simulation_lab`."
    )

    report = _read_json_safe(results_dir / "lifecycle_rule_simulation_results.json")
    if not report:
        st.info(
            "No simulation results yet. Run "
            "`python -m services.lifecycle_rule_simulation_lab` to generate them.",
            icon="ℹ️",
        )
        return

    _render_executive(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "📐 Baseline Replay",
            "🧪 Scenario Results",
            "🚪 Exit Timing",
            "🎯 Confidence Filter",
            "➕ HOLD → ADD",
            "🚩 Risk Flags",
            "🏁 Rankings",
            "🎯 Phase 170 Recommendation",
        ],
        horizontal=True,
        key="lrsl_section",
    )
    if section == "📐 Baseline Replay":
        _render_baseline(report.get("baseline") or {})
    elif section == "🧪 Scenario Results":
        _render_scenarios(report)
    elif section == "🚪 Exit Timing":
        _render_exit_timing(report)
    elif section == "🎯 Confidence Filter":
        _render_confidence(report)
    elif section == "➕ HOLD → ADD":
        _render_hold_add(report)
    elif section == "🚩 Risk Flags":
        _render_risk_flags(report)
    elif section == "🏁 Rankings":
        _render_rankings(report)
    else:
        _render_phase170(report)
