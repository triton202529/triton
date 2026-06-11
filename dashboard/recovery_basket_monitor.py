"""
TRITON Phase 177 — 📈 Recovery Basket Monitor dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _num, _plotly_bar


def _read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


_FEAS = {
    "IMMEDIATELY_FEASIBLE": "🟢 IMMEDIATELY_FEASIBLE",
    "FEASIBLE_WITH_TIME": "🟡 FEASIBLE_WITH_TIME",
    "CURRENTLY_UNAVAILABLE": "🔴 CURRENTLY_UNAVAILABLE",
}
_TREND = {
    "IMPROVING": "📈 IMPROVING",
    "UNCHANGED": "➡️ UNCHANGED",
    "DETERIORATING": "📉 DETERIORATING",
}
_PROB = {"VERY_HIGH": "🟢 VERY_HIGH", "HIGH": "🟢 HIGH", "MODERATE": "🟡 MODERATE", "LOW": "🔴 LOW"}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    ex = report.get("executive_report") or {}
    dist = report.get("distance_to_recovery") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current CPS", str(ex.get("current_cps", "—")))
    c2.metric("Distance to YELLOW", str(dist.get("distance_to_yellow", "—")))
    c3.metric("Distance to GREEN", str(dist.get("distance_to_green", "—")))
    prob = str(ex.get("recovery_probability", "—"))
    c4.metric("Recovery Probability", _PROB.get(prob, prob))
    st.markdown(
        f"- **Governance:** {ex.get('current_governance_state')} ({ex.get('current_escalation_state')})"
    )
    st.markdown(f"- **Progress:** {ex.get('recovery_progress') or '—'}")
    st.markdown(f"- **Readiness:** {ex.get('estimated_recovery_readiness') or '—'}")
    st.markdown(f"- **Blocking:** {', '.join(ex.get('blocking_candidates') or []) or 'none'}")


def _render_status(report: Dict[str, Any]) -> None:
    st.markdown("#### 🧺 Recovery Basket Status")
    rows = pd.DataFrame(report.get("recovery_basket_status") or [])
    if rows.empty:
        st.info("No status data.", icon="ℹ️")
        return
    view = rows.copy()
    view["feasibility_status"] = view["feasibility_status"].map(lambda x: _FEAS.get(str(x), str(x)))
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_promotions(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🔔 Promotion Tracking")
    events = report.get("promotion_events") or []
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    else:
        st.caption("No promotion events detected this run.")
    hist = _read_csv_safe(results_dir / "recovery_basket_promotion_history.csv")
    if not hist.empty:
        st.markdown("**Historical promotions**")
        st.dataframe(hist.tail(20), use_container_width=True, hide_index=True)


def _render_distance(report: Dict[str, Any]) -> None:
    st.markdown("#### 📏 Distance To Recovery")
    dist = report.get("distance_to_recovery") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Current CPS", str(dist.get("current_cps", "—")))
    c2.metric("Target YELLOW", str(dist.get("target_cps_yellow", 75)))
    c3.metric("Target GREEN", str(dist.get("target_cps_green", 90)))
    c4, c5 = st.columns(2)
    c4.metric("Distance To YELLOW", str(dist.get("distance_to_yellow", "—")))
    c5.metric("Distance To GREEN", str(dist.get("distance_to_green", "—")))


def _render_governance(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Governance Recovery")
    gt = report.get("governance_tracker") or {}
    trend = str(gt.get("recovery_trend", "—"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", _TREND.get(trend, trend))
    c2.metric("Projected CPS (partial)", str(gt.get("projected_cps_partial", "—")))
    c3.metric("Projected CPS (full)", str(gt.get("projected_cps_full", "—")))
    c4.metric("Projected Escalation", str(gt.get("projected_escalation_full", "—")))
    st.json(gt)


def _render_probability(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Recovery Probability")
    rp = report.get("recovery_probability") or {}
    label = str(rp.get("classification", "—"))
    st.metric("Classification", _PROB.get(label, label))
    st.metric("Score", f"{_num(rp.get('score_pct'))}%")
    st.json(rp.get("factors") or {})


def _render_historical(results_dir: Path) -> None:
    st.markdown("#### 📜 Historical Changes")
    hist = _read_csv_safe(results_dir / "recovery_basket_status.csv")
    if hist.empty:
        st.info("History builds as the monitor runs over time.", icon="ℹ️")
        return
    if "generated_at" in hist.columns and "confidence" in hist.columns:
        tickers = sorted(hist["ticker"].unique().tolist())
        pick = st.selectbox("Ticker", tickers, key="rbm_hist_ticker")
        sub = hist[hist["ticker"] == pick].copy()
        if not sub.empty:
            sub["generated_at"] = pd.to_datetime(sub["generated_at"], errors="coerce")
            sub = sub.sort_values("generated_at")
            if len(sub) >= 2:
                st.line_chart(sub.set_index("generated_at")[["confidence", "edge_score"]])
        st.dataframe(sub, use_container_width=True, hide_index=True)


def _render_alerts(report: Dict[str, Any]) -> None:
    alerts = report.get("alerts") or []
    if not alerts:
        st.success("No new recovery alerts this run.", icon="✅")
        return
    for a in alerts:
        st.warning(a, icon="🔔")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.caption(f"Components tracked: {cert.get('components_tracked', '—')}")
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_recovery_basket_monitor(results_dir: Path) -> None:
    st.markdown("### 📈 Recovery Basket Monitor")
    st.caption(
        "Continuous recovery basket monitoring — observability only. "
        "Refresh with `python -m services.recovery_basket_monitor`."
    )

    report = _read_json_safe(results_dir / "recovery_basket_monitor.json")
    if not report:
        st.info(
            "No monitor report yet. Run `python -m services.recovery_basket_monitor`.",
            icon="ℹ️",
        )
        return

    _render_executive(report)
    _render_alerts(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "🧺 Basket Status",
            "🔔 Promotions",
            "📏 Distance",
            "⚖️ Governance",
            "🎯 Probability",
            "📜 Historical",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="rbm_section",
    )
    if section == "🧺 Basket Status":
        _render_status(report)
    elif section == "🔔 Promotions":
        _render_promotions(results_dir, report)
    elif section == "📏 Distance":
        _render_distance(report)
    elif section == "⚖️ Governance":
        _render_governance(report)
    elif section == "🎯 Probability":
        _render_probability(report)
    elif section == "📜 Historical":
        _render_historical(results_dir)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
