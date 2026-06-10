"""
TRITON Phase 168 — ⚙️ Adaptive Lifecycle Optimization (read-only dashboard).

Displays the recommendation-only optimization roadmap built from Phase 167
intelligence: exit-timing root causes, confidence calibration, transition
weaknesses, risk-flag value rankings, the prioritized improvement backlog,
and historical-approximation impact estimates.
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


_PRIORITY_BADGE = {"P1": "🔴 P1", "P2": "🟠 P2", "P3": "🟡 P3", "P4": "🟢 P4"}


def _render_executive(executive: Dict[str, Any], impact: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Optimization Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimization Grade", str(executive.get("overall_optimization_grade") or "—"))
    c2.metric("Current Lifecycle Grade", str(executive.get("current_lifecycle_grade") or "—"))
    c3.metric(
        "Total Improvement Potential",
        _money(executive.get("estimated_total_improvement_potential")),
    )
    c4.metric("Estimated Alpha Gain", _num(impact.get("estimated_alpha_gain")))

    rows = [
        ("Top Problem", "top_problem"),
        ("Top Opportunity", "top_opportunity"),
        ("Highest Alpha Improvement", "highest_alpha_improvement"),
        ("Lowest Risk Improvement", "lowest_risk_improvement"),
        ("Most Expensive Decision Pattern", "most_expensive_decision_pattern"),
        ("Most Valuable Decision Pattern", "most_valuable_decision_pattern"),
        ("Phase 169 Focus", "phase_169_focus"),
    ]
    for label, key in rows:
        st.markdown(f"- **{label}:** {executive.get(key) or '—'}")


def _render_exit_optimization(data: Dict[str, Any]) -> None:
    st.markdown("#### 🚪 Exit Optimization")
    table = _table(data.get("table"))
    if table.empty:
        st.info("No exit-timing data available.", icon="ℹ️")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Late Exits", str(data.get("late_exit_count", 0)))
    c2.metric("Post-Exit Capital Lost", _money(data.get("late_exit_capital_lost")))
    c3.metric("Peak Giveback (pre-exit)", _money(data.get("peak_giveback_estimate")))
    delay = data.get("avg_late_exit_delay_days")
    c4.metric("Avg Delay After Peak", f"{delay} d" if delay is not None else "—")

    st.dataframe(table, use_container_width=True, hide_index=True)

    causes = data.get("root_causes") or []
    if causes:
        st.markdown("**Root causes of late exits**")
        for c in causes:
            st.markdown(f"- {c}")
    recs = data.get("recommendations") or []
    if recs:
        st.markdown("**Recommendations** *(no rules changed — operator review required)*")
        for r in recs:
            st.markdown(f"- 💡 {r}")


def _render_calibration(data: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Confidence Calibration")
    table = _table(data.get("table"))
    if table.empty:
        st.info("No calibration data available.", icon="ℹ️")
        return
    st.dataframe(table, use_container_width=True, hide_index=True)

    chart = (
        table.dropna(subset=["calibration_error"])
        if "calibration_error" in table.columns
        else pd.DataFrame()
    )
    if not chart.empty and not _plotly_bar(
        chart, "bucket", "calibration_error", "Calibration Error (actual − expected win rate)"
    ):
        st.bar_chart(chart.set_index("bucket")["calibration_error"])

    over = data.get("overconfident_zones") or []
    under = data.get("underconfident_zones") or []
    c1, c2 = st.columns(2)
    c1.metric("Overconfident Zones", ", ".join(over) if over else "none")
    c2.metric("Underconfident Zones", ", ".join(under) if under else "none")
    st.caption(
        "Expected win rate treats stated confidence as an implied probability "
        "(bucket midpoint). Positive error = outcomes beat the model's stated confidence."
    )
    for r in data.get("recommendations") or []:
        st.markdown(f"- 💡 {r}")


def _render_transitions(data: Dict[str, Any]) -> None:
    st.markdown("#### 🔁 Transition Analysis")
    table = _table(data.get("table"))
    if table.empty:
        st.info("No transition data available.", icon="ℹ️")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Transition", str(data.get("best_transition") or "—"))
    c2.metric("Worst Transition", str(data.get("worst_transition") or "—"))
    c3.metric("Most Expensive", str(data.get("most_expensive_transition") or "—"))
    st.dataframe(table, use_container_width=True, hide_index=True)

    explanation = str(data.get("hold_add_explanation") or "").strip()
    if explanation:
        st.markdown("**Why HOLD → ADD underperforms**")
        st.info(explanation, icon="🔍")
    for r in data.get("recommendations") or []:
        st.markdown(f"- 💡 {r}")


def _render_flags(records: Any) -> None:
    st.markdown("#### 🚩 Risk Flag Rankings")
    df = _table(records)
    if df.empty:
        st.info("No risk-flag rankings available.", icon="ℹ️")
        return
    badge = {
        "HIGH_VALUE_FLAG": "🟢 HIGH_VALUE_FLAG",
        "MEDIUM_VALUE_FLAG": "🟡 MEDIUM_VALUE_FLAG",
        "LOW_VALUE_FLAG": "🔴 LOW_VALUE_FLAG",
        "UNOBSERVED": "⚪ UNOBSERVED",
    }
    view = df.copy()
    if "value_class" in view.columns:
        view["value_class"] = view["value_class"].map(lambda v: badge.get(str(v), str(v)))
    st.dataframe(view, use_container_width=True, hide_index=True)
    unobserved = df[df.get("value_class", pd.Series(dtype=str)) == "UNOBSERVED"]
    if not unobserved.empty:
        st.warning(
            f"{len(unobserved)} canonical risk flags have never appeared in decision "
            "records — the flag pipeline is not reaching the timeline. This is itself "
            "a P2 finding (see backlog).",
            icon="⚠️",
        )


def _render_backlog(records: Any) -> None:
    st.markdown("#### 🗂️ Optimization Backlog")
    df = _table(records)
    if df.empty:
        st.info("No backlog generated.", icon="ℹ️")
        return
    view = df.copy()
    view["priority"] = view["priority"].map(lambda p: _PRIORITY_BADGE.get(str(p), str(p)))
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption(
        "Recommendation-only: nothing in this backlog has been applied. Each item "
        "requires operator review before any rule change in a future phase."
    )


def _render_impact(impact: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Estimated Impact (historical approximation)")
    if not impact:
        st.info("No impact estimates available.", icon="ℹ️")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Profit Factor",
        _num(impact.get("estimated_profit_factor")),
        delta=(
            f"{(impact.get('estimated_profit_factor') or 0) - (impact.get('current_profit_factor') or 0):+.2f} vs current"
            if impact.get("current_profit_factor") is not None
            else None
        ),
    )
    c2.metric(
        "Overall Alpha Score",
        _num(impact.get("estimated_overall_alpha")),
        delta=(
            f"{impact.get('estimated_alpha_gain'):+.2f}"
            if impact.get("estimated_alpha_gain") is not None
            else None
        ),
    )
    c3.metric("Realized P&L Gain", _money(impact.get("estimated_realized_pnl_gain")))
    c4, c5 = st.columns(2)
    c4.metric("Current Profit Factor", _num(impact.get("current_profit_factor")))
    c5.metric(
        "Win Rate Gain",
        (
            f"+{impact.get('estimated_win_rate_gain_pts', 0)} pts"
            if impact.get("estimated_win_rate_gain_pts")
            else "—"
        ),
    )
    st.caption(str(impact.get("method") or ""))


def _render_certification(cert: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification Status")
    if not cert:
        st.info("No certification data.", icon="ℹ️")
        return
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        str(cert.get("status")), "⚪"
    )
    c1, c2 = st.columns(2)
    c1.metric("Status", f"{badge} {cert.get('status', '—')}")
    c2.metric("Upstream Intelligence", str(cert.get("upstream_status", "—")))
    for r in cert.get("reasons") or []:
        st.warning(r, icon="⚠️")
    st.success(str(cert.get("safety") or "Recommendation-only phase."), icon="🔒")


def render_adaptive_lifecycle_optimization(results_dir: Path) -> None:
    st.markdown("### ⚙️ Adaptive Lifecycle Optimization")
    st.caption(
        "Data-driven optimization roadmap from Phase 167 intelligence — "
        "recommendation only, no thresholds or rules modified. Refresh with "
        "`python -m services.adaptive_lifecycle_optimization`."
    )

    report = _read_json_safe(results_dir / "adaptive_lifecycle_optimization.json")
    if not report:
        st.info(
            "No optimization report yet. Run "
            "`python -m services.adaptive_lifecycle_optimization` to generate it.",
            icon="ℹ️",
        )
        return

    _render_executive(report.get("executive_report") or {}, report.get("impact_estimates") or {})
    st.divider()

    section = st.radio(
        "Section",
        [
            "🚪 Exit Optimization",
            "🎯 Confidence Calibration",
            "🔁 Transitions",
            "🚩 Risk Flags",
            "🗂️ Backlog",
            "📈 Estimated Impact",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="alo_section",
    )
    if section == "🚪 Exit Optimization":
        _render_exit_optimization(report.get("exit_optimization") or {})
    elif section == "🎯 Confidence Calibration":
        _render_calibration(report.get("confidence_calibration") or {})
    elif section == "🔁 Transitions":
        _render_transitions(report.get("transition_optimization") or {})
    elif section == "🚩 Risk Flags":
        _render_flags(report.get("risk_flag_rankings"))
    elif section == "🗂️ Backlog":
        _render_backlog(report.get("improvement_backlog"))
    elif section == "📈 Estimated Impact":
        _render_impact(report.get("impact_estimates") or {})
    else:
        _render_certification(report.get("certification") or {})
