"""
TRITON Institutional Reasoning dashboards — Phases 37–39.
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


def render_causal_reasoning_center(results_dir: Path) -> None:
    st.markdown("### 🔍 Causal Reasoning Center")
    st.caption(
        "Cause-effect analysis for escalations, certification blocks, readiness failures, "
        "and persistent risk patterns. **Advisory diagnostic only — no execution.**"
    )

    doc = _load_json(results_dir / "causal_reasoning.json")
    if not doc:
        st.warning("Causal reasoning data not found. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    st.metric("Analyses", doc.get("reasoning_count", 0))

    analyses: List[Dict[str, Any]] = doc.get("analyses") or []
    if not analyses:
        st.info("No active causal analyses — institutional posture may be nominal.")
        return

    for analysis in analyses:
        issue = analysis.get("issue", "Unknown issue")
        confidence = analysis.get("confidence", 0)
        causes = analysis.get("likely_causes") or []
        evidence = analysis.get("evidence") or []

        with st.expander(f"**{issue}** — confidence {confidence}%", expanded=True):
            if causes:
                st.markdown("**Likely causes**")
                for cause in causes:
                    st.markdown(f"- {cause}")
            if evidence:
                st.markdown("**Evidence**")
                for item in evidence:
                    st.markdown(f"- `{item}`")

    st.info("Causal reasoning produces advisory text only. No broker actions are taken.")


def render_explainability_center(results_dir: Path) -> None:
    st.markdown("### 📖 Explainability Center")
    st.caption(
        "Plain-language explanations for governor posture, certification status, "
        "readiness gates, and automation authorization. **Human-readable advisory only.**"
    )

    doc = _load_json(results_dir / "institutional_explanations.json")
    if not doc:
        st.warning("Explainability data not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    st.metric("Explanations", doc.get("explanation_count", 0))

    explanations: List[Dict[str, Any]] = doc.get("explanations") or []
    if not explanations:
        st.info("No explanations generated for current institutional state.")
        return

    for exp in explanations:
        topic = exp.get("topic", "—")
        subject = exp.get("subject", "—")
        st.markdown(f"#### {topic}: `{subject}`")
        st.write(exp.get("explanation", ""))

        facts = exp.get("supporting_facts") or []
        if facts:
            with st.expander("Supporting facts"):
                for fact in facts:
                    st.markdown(f"- {fact}")
        st.markdown("---")

    st.warning(
        "Explainability output does not execute trades or modify portfolios. "
        "All conclusions require human review."
    )


def render_institutional_insights_center(results_dir: Path) -> None:
    st.markdown("### 💡 Institutional Insights")
    st.caption(
        "Strategic observations synthesized from causal reasoning, explainability, "
        "organizational learning, and executive risk signals. **Advisory only.**"
    )

    doc = _load_json(results_dir / "institutional_insights.json")
    if not doc:
        st.warning("Institutional insights not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    st.metric("Top Insight", doc.get("top_insight", "—"))
    st.metric("Insight Confidence", f"{doc.get('insight_confidence', 0)}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Risk", doc.get("most_important_risk", "—"))
    c2.metric("Top Weakness", doc.get("most_important_weakness", "—"))
    c3.metric("Top Opportunity", (doc.get("most_important_opportunity") or "—")[:40])
    c4.metric("Governance Concern", (doc.get("most_important_governance_concern") or "—")[:40])

    insights: List[Dict[str, Any]] = doc.get("insights") or []
    if insights:
        st.markdown("#### Insight Breakdown")
        df = pd.DataFrame(
            [
                {
                    "Priority": i.get("priority"),
                    "Category": i.get("category"),
                    "Insight": i.get("insight"),
                }
                for i in sorted(insights, key=lambda x: x.get("priority") or 99)
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        cat_counts = df["Category"].value_counts()
        if not cat_counts.empty:
            st.bar_chart(cat_counts)

    history_path = results_dir / "institutional_insights_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty:
                st.markdown("#### Insights Trend")
                st.caption(f"History rows: **{len(hist)}**")
                if "insight_confidence" in hist.columns:
                    chart_df = hist.tail(50).set_index("timestamp")["insight_confidence"]
                    st.line_chart(chart_df)
        except Exception:
            pass

    st.info("Institutional insights are observational. No automated trading or intervention.")
