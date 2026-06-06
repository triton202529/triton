"""
TRITON Institutional Intelligence dashboards — Phases 31–33.
Read-only / simulation only. No live execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _quality_badge(band: str) -> str:
    return {
        "LOW": "🔴 LOW",
        "MODERATE": "🟠 MODERATE",
        "HIGH": "🟡 HIGH",
        "EXCELLENT": "🟢 EXCELLENT",
    }.get(str(band).upper(), band)


def _institutional_badge(band: str) -> str:
    return {
        "FOUNDATIONAL": "🔴 FOUNDATIONAL",
        "DEVELOPING": "🟠 DEVELOPING",
        "ADVANCED": "🟡 ADVANCED",
        "INSTITUTIONAL": "🟢 INSTITUTIONAL",
    }.get(str(band).upper(), band)


def render_decision_quality_center(results_dir: Path) -> None:
    st.markdown("### 🧩 Decision Quality Center")
    st.caption(
        "Decision quality scoring across advisory, escalation, preservation, governance, "
        "and recommendation stability. **Advisory only — no execution.**"
    )

    doc = _load_json(results_dir / "decision_quality_assessment.json")
    if not doc:
        st.warning("Decision quality data not found. Run `python -m services.risk_watchdog`.")
        return

    band = str(doc.get("quality_band") or "MODERATE")
    st.metric("Quality Band", _quality_badge(band))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision Quality Score", f"{doc.get('decision_quality_score', 0)}/100")
    c2.metric("Strongest Area", doc.get("strongest_area", "—"))
    c3.metric("Weakest Area", doc.get("weakest_area", "—"))

    metrics: List[Dict[str, Any]] = doc.get("metrics") or []
    if metrics:
        st.markdown("#### Quality Metrics")
        df = pd.DataFrame([{"Metric": m.get("metric"), "Score": m.get("score")} for m in metrics])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Metric")[["Score"]])

    history_path = results_dir / "decision_quality_assessment_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "decision_quality_score" in hist.columns:
                st.markdown("#### Decision Quality Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["decision_quality_score"])
        except Exception:
            pass

    st.info("Decision quality assessment informs human review only. No broker actions are taken.")


def render_institutional_intelligence(results_dir: Path) -> None:
    st.markdown("### 🏛 Institutional Intelligence")
    st.caption(
        "Cross-layer institutional intelligence across monitoring, governance, certification, "
        "oversight, accountability, and preservation. **Diagnostic only.**"
    )

    doc = _load_json(results_dir / "institutional_intelligence.json")
    if not doc:
        st.warning("Institutional intelligence data not found. Run the watchdog first.")
        return

    band = str(doc.get("institutional_band") or "DEVELOPING")
    st.metric("Institutional Band", _institutional_badge(band))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intelligence Score", f"{doc.get('institutional_intelligence_score', 0)}/100")
    c2.metric("Coordination Score", f"{doc.get('coordination_score', 0)}/100")
    c3.metric("Strongest Area", doc.get("strongest_area", "—"))
    c4.metric("Weakest Area", doc.get("weakest_area", "—"))

    areas: List[Dict[str, Any]] = doc.get("areas") or []
    if areas:
        st.markdown("#### Area Scores")
        df = pd.DataFrame(
            [
                {
                    "Area": a.get("area"),
                    "Score": a.get("score"),
                    "Band": a.get("band"),
                }
                for a in areas
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Area")[["Score"]])

    history_path = results_dir / "institutional_intelligence_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty:
                st.markdown("#### Intelligence Trends")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                chart_cols = [
                    c
                    for c in ("institutional_intelligence_score", "coordination_score")
                    if c in trend.columns
                ]
                if chart_cols:
                    st.line_chart(trend.set_index("timestamp")[chart_cols])
        except Exception:
            pass

    st.info("Institutional intelligence is read-only. No automated actions are triggered.")


def render_strategic_self_improvement(results_dir: Path) -> None:
    st.markdown("### 🚀 Strategic Self-Improvement")
    st.caption(
        "Prioritized improvement opportunities from decision quality, maturity, certification, "
        "and readiness gaps. **Advisory prioritization only.**"
    )

    doc = _load_json(results_dir / "strategic_self_improvement.json")
    if not doc:
        st.warning("Strategic self-improvement data not found. Run the watchdog first.")
        return

    st.metric("Top Priority", doc.get("top_priority", "—"))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Improvement Confidence", f"{doc.get('improvement_score', 0)}%")
    c2.metric("Decision Quality", doc.get("decision_quality_score", "—"))
    c3.metric("Institutional Intelligence", doc.get("institutional_intelligence_score", "—"))

    focus = doc.get("recommended_focus") or []
    if focus:
        st.markdown("#### Recommended Focus")
        st.markdown(", ".join(f"**{f}**" for f in focus))

    opportunities = doc.get("improvement_opportunities") or []
    if opportunities:
        st.markdown("#### Improvement Opportunities")
        for opp in opportunities:
            st.markdown(f"- {opp}")

    weakest = doc.get("weakest_systems") or []
    if weakest:
        st.markdown("#### Weakest Systems")
        for w in weakest:
            st.markdown(f"- **{w}**")

    leverage = doc.get("highest_leverage_enhancements") or []
    if leverage:
        st.markdown("#### Highest-Leverage Enhancements")
        for item in leverage:
            st.markdown(f"- {item}")

    debt = doc.get("technical_debt_areas") or []
    if debt:
        st.markdown("#### Technical Debt Areas")
        for item in debt:
            st.markdown(f"- {item}")

    st.warning(
        "Strategic self-improvement does not execute trades or modify portfolios. "
        "All recommendations require human review."
    )
