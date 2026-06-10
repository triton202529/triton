"""
TRITON Phase 170 — 🚪 Exit Timing Framework (read-only dashboard).

Displays the certified exit-timing design framework: late exit catalog,
opportunity-cost analysis, peak drawdown escalation bands, exit health
scores, and Phase 171 readiness recommendations.
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
    "NOT_LATE": "🟢 NOT_LATE",
    "MILD_DELAY": "🟡 MILD_DELAY",
    "LATE_EXIT": "🟠 LATE_EXIT",
    "SEVERE_LATE_EXIT": "🔴 SEVERE_LATE_EXIT",
}
_HEALTH_BADGE = {
    "Excellent": "🟣 Excellent",
    "Good": "🟢 Good",
    "Average": "🟡 Average",
    "Weak": "🟠 Weak",
    "Poor": "🔴 Poor",
}
_READY_BADGE = {
    "READY_FOR_IMPLEMENTATION": "🟢 READY_FOR_IMPLEMENTATION",
    "NEEDS_MORE_DATA": "🟡 NEEDS_MORE_DATA",
    "DO_NOT_IMPLEMENT": "🔴 DO_NOT_IMPLEMENT",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    ex = report.get("executive_report") or {}
    cert = report.get("certification") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Late Exits", str(ex.get("late_exit_count", "—")))
    c2.metric("Severe Late Exits", str(ex.get("severe_late_exit_count", "—")))
    c3.metric("Avg Delay", f"{ex.get('average_delay_days', '—')} d")
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        str(cert.get("status")), "⚪"
    )
    c4.metric("Framework Readiness", f"{badge} {ex.get('framework_readiness', '—')}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Opportunity Cost", _money(ex.get("average_opportunity_cost")))
    c6.metric("Estimated Alpha Recovery", _money(ex.get("estimated_alpha_recovery")))
    c7.metric("Avg Exit Health", _num(ex.get("average_exit_health")))
    c8.metric("Exits Analyzed", str(cert.get("exits_analyzed", "—")))
    st.markdown(f"- **Largest Missed Exit:** {ex.get('largest_missed_exit') or '—'}")
    st.markdown(f"- **Best Exit:** {ex.get('best_exit') or '—'}")
    st.markdown(f"- **Worst Exit:** {ex.get('worst_exit') or '—'}")


def _render_catalog(report: Dict[str, Any]) -> None:
    st.markdown("#### 📒 Late Exit Catalog")
    df = _table(report.get("late_exit_catalog"))
    if df.empty:
        st.info("No exits in the catalog yet.", icon="ℹ️")
        return
    opts = [
        c
        for c in ("SEVERE_LATE_EXIT", "LATE_EXIT", "MILD_DELAY", "NOT_LATE")
        if c in set(df["classification"])
    ]
    pick = st.multiselect("Classification filter", options=opts, default=opts, key="etf_class")
    view = df[df["classification"].isin(pick)] if pick else df
    view = view.copy()
    view["classification"] = view["classification"].map(lambda c: _CLASS_BADGE.get(str(c), str(c)))
    if "health_class" in view.columns:
        view["health_class"] = view["health_class"].map(lambda c: _HEALTH_BADGE.get(str(c), str(c)))
    st.caption(
        f"{len(view):,} exits shown · detector thresholds: 2d/2% mild · 5d/5% late · 10d/8% severe"
    )
    st.dataframe(view, use_container_width=True, hide_index=True)

    late = df[df["classification"].isin(["LATE_EXIT", "SEVERE_LATE_EXIT"])]
    if not late.empty:
        agg = late.groupby("ticker", as_index=False)["opportunity_cost"].sum()
        if not _plotly_bar(
            agg, "ticker", "opportunity_cost", "Late-Exit Opportunity Cost by Ticker"
        ):
            st.bar_chart(agg.set_index("ticker")["opportunity_cost"])


def _render_opportunity(report: Dict[str, Any]) -> None:
    st.markdown("#### 💸 Opportunity Cost Analysis")
    metrics = report.get("opportunity_cost_metrics") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Opportunity Cost", _money(metrics.get("total_opportunity_cost")))
    c2.metric("Average", _money(metrics.get("average_opportunity_cost")))
    c3.metric("Largest", _money(metrics.get("largest_opportunity_cost")))
    c4.metric("Recoverable (Phase 169)", _money(metrics.get("capital_recovered_if_optimized")))
    st.caption(str(metrics.get("method") or ""))

    df = _table(report.get("opportunity_cost_report"))
    if df.empty:
        st.info("No opportunity-cost rows available.", icon="ℹ️")
        return
    view = df.copy()
    view["classification"] = view["classification"].map(lambda c: _CLASS_BADGE.get(str(c), str(c)))
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_escalation(report: Dict[str, Any]) -> None:
    st.markdown("#### 📶 Peak Drawdown Escalation (design)")
    spec = report.get("peak_drawdown_escalation") or {}
    st.markdown(str(spec.get("design") or ""))
    bands = _table(spec.get("bands"))
    if not bands.empty:
        bands = bands.rename(columns={"giveback_below": "giveback_fraction_below"})
        st.dataframe(bands, use_container_width=True, hide_index=True)

    catalog = _table(report.get("late_exit_catalog"))
    if not catalog.empty and "escalation_band_at_exit" in catalog.columns:
        counts = catalog["escalation_band_at_exit"].value_counts().reset_index()
        counts.columns = ["band", "exits"]
        st.markdown("**Retrospective validation — band each historical exit had reached**")
        if not _plotly_bar(counts, "band", "exits", "Escalation Band at Actual Exit"):
            st.bar_chart(counts.set_index("band")["exits"])
        late = catalog[catalog["classification"].isin(["LATE_EXIT", "SEVERE_LATE_EXIT"])]
        if not late.empty:
            hit = (
                late["escalation_band_at_exit"]
                .isin(["REVIEW", "ESCALATE", "EXIT_CANDIDATE"])
                .mean()
                * 100
            )
            st.metric("Late exits flagged by bands before exit", _pct(hit))
    st.warning("Design only — no escalation rule is active.", icon="🔒")


def _render_health(report: Dict[str, Any]) -> None:
    st.markdown("#### ❤️ Exit Health Scores")
    df = _table(report.get("late_exit_catalog"))
    if df.empty or "exit_health_score" not in df.columns:
        st.info("No health scores available.", icon="ℹ️")
        return
    scores = pd.to_numeric(df["exit_health_score"], errors="coerce").dropna()
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Health", _num(scores.mean()))
    c2.metric("Best", _num(scores.max()))
    c3.metric("Worst", _num(scores.min()))
    dist = df["health_class"].value_counts().reset_index()
    dist.columns = ["class", "exits"]
    if not _plotly_bar(dist, "class", "exits", "Exit Health Distribution"):
        st.bar_chart(dist.set_index("class")["exits"])
    st.caption(
        "Health = 25% timeliness + 25% drawdown control + 20% capital preservation + "
        "15% decision quality + 15% post-exit outcome. Bands: 90+ Excellent · 75+ Good · "
        "60+ Average · 40+ Weak · <40 Poor."
    )
    view = df[
        [
            "ticker",
            "exit_date",
            "days_late",
            "giveback_pct",
            "exit_health_score",
            "health_class",
        ]
    ].copy()
    view["health_class"] = view["health_class"].map(lambda c: _HEALTH_BADGE.get(str(c), str(c)))
    st.dataframe(
        view.sort_values("exit_health_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def _render_quality(report: Dict[str, Any]) -> None:
    st.markdown("#### 🕰️ Historical Exit Quality")
    df = _table(report.get("late_exit_catalog"))
    if df.empty:
        st.info("No historical exits available.", icon="ℹ️")
        return
    df = df.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["exit_date"]).sort_values("exit_date")
    if df.empty:
        st.info("No dated exits available.", icon="ℹ️")
        return
    monthly = (
        df.set_index("exit_date").resample("ME")["exit_health_score"].mean().dropna().reset_index()
    )
    if len(monthly) >= 2:
        monthly["month"] = monthly["exit_date"].dt.strftime("%Y-%m")
        if not _plotly_bar(monthly, "month", "exit_health_score", "Average Exit Health by Month"):
            st.bar_chart(monthly.set_index("month")["exit_health_score"])
    detector = report.get("late_exit_detector") or {}
    st.markdown("**Late Exit Detector design**")
    st.info(str(detector.get("design") or "—"), icon="🛠️")
    thresholds = detector.get("thresholds") or {}
    if thresholds:
        st.dataframe(
            pd.DataFrame([{"classification": k, "threshold": v} for k, v in thresholds.items()]),
            use_container_width=True,
            hide_index=True,
        )


def _render_recommendations(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Recommendations — Phase 171 Readiness")
    df = _table(report.get("phase_171_readiness"))
    if df.empty:
        st.info("No recommendations generated.", icon="ℹ️")
        return
    view = df.copy()
    view["readiness"] = view["readiness"].map(lambda r: _READY_BADGE.get(str(r), str(r)))
    st.dataframe(view, use_container_width=True, hide_index=True)
    ready = df[df["readiness"] == "READY_FOR_IMPLEMENTATION"]
    if not ready.empty:
        st.success(
            "Ready for Phase 171: " + ", ".join(ready["enhancement"].astype(str)),
            icon="✅",
        )
    blocked = df[df["readiness"] == "DO_NOT_IMPLEMENT"]
    if not blocked.empty:
        st.error(
            "Do not implement: " + ", ".join(blocked["enhancement"].astype(str)),
            icon="🚫",
        )


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification Status")
    cert = report.get("certification") or {}
    if not cert:
        st.info("No certification data.", icon="ℹ️")
        return
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        str(cert.get("status")), "⚪"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", f"{badge} {cert.get('status', '—')}")
    c2.metric("Exits Analyzed", str(cert.get("exits_analyzed", "—")))
    c3.metric("FP Within 5% Bound", "✅" if cert.get("false_positive_within_bound") else "❌")
    c4.metric("Band Consistency", "✅" if cert.get("escalation_band_consistency") else "❌")
    st.caption(f"Phase 169 simulation certification: {cert.get('simulation_certification', '—')}")
    for r in cert.get("reasons") or []:
        st.warning(r, icon="⚠️")
    st.success(str(cert.get("safety") or "Design-only phase."), icon="🔒")


def render_exit_timing_framework(results_dir: Path) -> None:
    st.markdown("### 🚪 Exit Timing Framework")
    st.caption(
        "Certified design framework for detecting late exits — design and "
        "observability only, nothing operational changes. Refresh with "
        "`python -m services.exit_timing_framework`."
    )

    report = _read_json_safe(results_dir / "exit_timing_framework.json")
    if not report:
        st.info(
            "No framework report yet. Run "
            "`python -m services.exit_timing_framework` to generate it.",
            icon="ℹ️",
        )
        return

    _render_executive(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "📒 Late Exit Catalog",
            "💸 Opportunity Cost",
            "📶 Peak Drawdown Escalation",
            "❤️ Exit Health Scores",
            "🕰️ Historical Exit Quality",
            "🎯 Recommendations",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="etf_section",
    )
    if section == "📒 Late Exit Catalog":
        _render_catalog(report)
    elif section == "💸 Opportunity Cost":
        _render_opportunity(report)
    elif section == "📶 Peak Drawdown Escalation":
        _render_escalation(report)
    elif section == "❤️ Exit Health Scores":
        _render_health(report)
    elif section == "🕰️ Historical Exit Quality":
        _render_quality(report)
    elif section == "🎯 Recommendations":
        _render_recommendations(report)
    else:
        _render_certification(report)
