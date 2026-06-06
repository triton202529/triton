"""
TRITON Institutional Operations dashboards — Phases 28–30.
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


def _recommendation_badge(rec: str) -> str:
    return {
        "CLEAR": "🟢 CLEAR",
        "MONITOR": "🟡 MONITOR",
        "REVIEW_REQUIRED": "🟠 REVIEW REQUIRED",
        "ESCALATE": "🔴 ESCALATE",
    }.get(str(rec).upper(), rec)


def _area_badge(status: str) -> str:
    return {
        "CLEAR": "🟢",
        "MONITOR": "🟡",
        "CONCERN": "🟠",
        "CRITICAL": "🔴",
    }.get(str(status).upper(), status)


def _maturity_badge(band: str) -> str:
    return {
        "FOUNDATIONAL": "🔴 FOUNDATIONAL",
        "DEVELOPING": "🟠 DEVELOPING",
        "ADVANCED": "🟡 ADVANCED",
        "INSTITUTIONAL": "🟢 INSTITUTIONAL",
    }.get(str(band).upper(), band)


def _oversight_badge(status: str) -> str:
    return {
        "ACTIVE": "🟢 ACTIVE",
        "REVIEW": "🟠 REVIEW",
        "SUSPENDED": "🔴 SUSPENDED",
    }.get(str(status).upper(), status)


def render_investment_committee_review(results_dir: Path) -> None:
    st.markdown("### 🏛 Investment Committee Review")
    st.caption(
        "Investment committee assessment across portfolio, risk, governance, "
        "certification, and readiness. **Advisory only — no execution.**"
    )

    doc = _load_json(results_dir / "investment_committee_review.json")
    if not doc:
        st.warning("Investment committee data not found. Run `python -m services.risk_watchdog`.")
        return

    rec = str(doc.get("committee_recommendation") or "MONITOR")
    st.metric("Committee Recommendation", _recommendation_badge(rec))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{doc.get('confidence', 0)}%")
    c2.metric("Avg Area Score", doc.get("average_area_score", 0))
    c3.metric("CPS", doc.get("capital_preservation_score", "—"))
    c4.metric("Escalation", doc.get("escalation_state", "—"))

    concerns = doc.get("top_concerns") or []
    if concerns:
        st.markdown("#### Top Concerns")
        for c in concerns:
            st.markdown(f"- **{c}**")

    areas: List[Dict[str, Any]] = doc.get("review_areas") or []
    if areas:
        st.markdown("#### Review Areas")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Area": a.get("area"),
                        "Score": a.get("score"),
                        "Status": a.get("status"),
                        "Concerns": ", ".join(a.get("concerns") or []) or "—",
                    }
                    for a in areas
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        for a in areas:
            with st.expander(
                f"{_area_badge(a.get('status', ''))} {a.get('area')} — {a.get('score')}/100",
                expanded=False,
            ):
                for c in a.get("concerns") or []:
                    st.markdown(f"- {c}")
                if not a.get("concerns"):
                    st.caption("No active concerns in this area.")

    st.info("Investment committee review informs human review only. No broker actions are taken.")


def render_triton_maturity_assessment(results_dir: Path) -> None:
    st.markdown("### 📈 Triton Maturity Assessment")
    st.caption(
        "Institutional maturity scoring across monitoring, governance, preservation, "
        "certification, readiness, and oversight. **Diagnostic only.**"
    )

    doc = _load_json(results_dir / "triton_maturity_assessment.json")
    if not doc:
        st.warning("Maturity assessment not found. Run the watchdog first.")
        return

    band = str(doc.get("maturity_band") or "DEVELOPING")
    st.metric("Maturity Band", _maturity_badge(band))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Maturity", f"{doc.get('overall_maturity', 0)}/100")
    c2.metric("Strongest Area", doc.get("strongest_area", "—"))
    c3.metric("Weakest Area", doc.get("weakest_area", "—"))

    categories: List[Dict[str, Any]] = doc.get("categories") or []
    if categories:
        st.markdown("#### Category Scores")
        df = pd.DataFrame(
            [
                {
                    "Category": c.get("category"),
                    "Score": c.get("score"),
                    "Band": c.get("maturity_band"),
                }
                for c in categories
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        chart_df = df.set_index("Category")[["Score"]]
        st.bar_chart(chart_df)

    history_path = results_dir / "triton_maturity_assessment_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "overall_maturity" in hist.columns:
                st.markdown("#### Maturity Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["overall_maturity"])
        except Exception:
            pass

    st.info("Maturity assessment is read-only. No automated actions are triggered.")


def render_strategic_oversight_center(results_dir: Path) -> None:
    st.markdown("### 🎯 Strategic Oversight Center")
    st.caption(
        "Unified strategic view aggregating governor, certification, investment committee, "
        "maturity, executive risk, and governance board. **Advisory only.**"
    )

    doc = _load_json(results_dir / "strategic_oversight.json")
    if not doc:
        st.warning("Strategic oversight data not found. Run the watchdog first.")
        return

    status = str(doc.get("oversight_status") or "ACTIVE")
    st.metric("Oversight Status", _oversight_badge(status))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategic Confidence", f"{doc.get('strategic_confidence', 0)}%")
    c2.metric("Institutional Readiness", doc.get("institutional_readiness", "—"))
    c3.metric("Strategic Readiness", doc.get("strategic_readiness", "—"))
    automation = str(doc.get("automation_status") or "NOT_AUTHORIZED")
    c4.metric("Automation Status", automation)

    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Committee Recommendation", doc.get("committee_recommendation", "—"))
    c_b.metric("Preservation Posture", doc.get("preservation_posture", "—"))
    c_c.metric("Board Status", doc.get("board_status", "—"))

    concerns = doc.get("top_strategic_concerns") or []
    if concerns:
        st.markdown("#### Top Strategic Concerns")
        for c in concerns:
            st.markdown(f"- **{c}**")

    recommendations = doc.get("strategic_recommendations") or []
    if recommendations:
        st.markdown("#### Strategic Recommendations (Advisory Only)")
        for rec in recommendations:
            st.markdown(f"- {rec}")

    c_x, c_y = st.columns(2)
    c_x.metric("Overall Maturity", doc.get("overall_maturity", "—"))
    c_y.metric(
        "Live Execution Permitted",
        "No" if not doc.get("live_execution_permitted") else "Yes",
    )

    history_path = results_dir / "strategic_oversight_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "strategic_confidence" in hist.columns:
                st.markdown("#### Strategic Confidence Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["strategic_confidence"])
        except Exception:
            pass

    st.warning(
        "automation_status defaults to NOT_AUTHORIZED. "
        "Strategic oversight does not execute trades or modify portfolios."
    )
