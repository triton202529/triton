"""
TRITON Institutional Planning dashboards — Phases 43–45.
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


def render_scenario_planning_center(results_dir: Path) -> None:
    st.markdown("### 🗺 Scenario Planning Center")
    st.caption(
        "Six institutional scenario lenses derived from CPI, escalation, readiness, "
        "certification, and strategic forecasts. **Advisory simulation only.**"
    )

    doc = _load_json(results_dir / "scenario_planning.json")
    if not doc:
        st.warning("Scenario planning data not found. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Scenarios", doc.get("scenario_count", 0))
    c2.metric("Probability Sum", doc.get("probability_sum", 0))
    inputs = doc.get("inputs") or {}
    c3.metric("CPI Score", inputs.get("cpi_score", "—"))

    st.caption(doc.get("probability_model", ""))

    scenarios: List[Dict[str, Any]] = doc.get("scenarios") or []
    if not scenarios:
        st.info("No scenarios generated for current institutional posture.")
        return

    for scenario in scenarios:
        name = scenario.get("scenario", "—")
        prob = scenario.get("probability", 0)
        outcome = scenario.get("primary_outcome", "")
        with st.expander(f"**{name}** — {prob}%", expanded=name in ("BASE_CASE", "BEST_CASE")):
            st.write(outcome)
            assumptions = scenario.get("assumptions") or []
            if assumptions:
                st.markdown("**Assumptions**")
                for a in assumptions:
                    st.markdown(f"- {a}")
            st.caption(f"Horizon: {scenario.get('time_horizon_days', 90)} days")

    df = pd.DataFrame(
        [
            {
                "Scenario": s.get("scenario"),
                "Probability": s.get("probability"),
                "Primary Outcome": s.get("primary_outcome"),
                "Horizon (days)": s.get("time_horizon_days"),
            }
            for s in scenarios
        ]
    )
    st.markdown("#### Scenario Summary")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if "Probability" in df.columns and not df.empty:
        chart_df = df.set_index("Scenario")["Probability"]
        st.bar_chart(chart_df)

    st.info("Scenario planning does not trigger trades or portfolio modifications.")


def render_future_path_analysis_center(results_dir: Path) -> None:
    st.markdown("### 🛣 Future Path Analysis")
    st.caption(
        "Trajectory evaluation across current, accelerated, stalled, and governance paths. "
        "**Simulation-only — no automated intervention.**"
    )

    doc = _load_json(results_dir / "future_path_analysis.json")
    if not doc:
        st.warning("Future path analysis not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    st.metric("Recommended Path", doc.get("recommended_path", "—"))

    paths: List[Dict[str, Any]] = doc.get("paths") or []
    if not paths:
        st.info("No paths evaluated for current conditions.")
        return

    for path in paths:
        name = path.get("path", "—")
        benefit = path.get("expected_benefit", 0)
        confidence = path.get("confidence", 0)
        with st.expander(f"**{name}** — benefit {benefit}%, confidence {confidence}%"):
            st.write(path.get("description", ""))
            milestones = path.get("milestones") or []
            if milestones:
                st.markdown("**Milestones**")
                for m in milestones:
                    st.markdown(f"- {m}")
            risks = path.get("risks") or []
            if risks:
                st.markdown("**Risks**")
                for r in risks:
                    st.markdown(f"- {r}")

    df = pd.DataFrame(
        [
            {
                "Path": p.get("path"),
                "Expected Benefit": p.get("expected_benefit"),
                "Confidence": p.get("confidence"),
            }
            for p in paths
        ]
    )
    st.markdown("#### Path Comparison")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if not df.empty:
        benefit_chart = df.set_index("Path")["Expected Benefit"]
        st.bar_chart(benefit_chart)

    st.warning(
        "Future path analysis is advisory trajectory modeling. "
        "No broker actions or live execution."
    )


def render_strategic_priorities_center(results_dir: Path) -> None:
    st.markdown("### 🎯 Strategic Priorities Center")
    st.caption(
        "Ranked institutional objectives synthesized from strategic reasoning, "
        "self-improvement, wisdom, scenarios, and future paths. **Advisory only.**"
    )

    doc = _load_json(results_dir / "strategic_priorities.json")
    if not doc:
        st.warning("Strategic priorities not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Priority", doc.get("top_priority", "—"))
    c2.metric("Recommended Path", doc.get("recommended_path", "—"))
    c3.metric("Dominant Scenario", doc.get("dominant_scenario", "—"))

    st.markdown("#### Four Priority Lenses")
    l1, l2 = st.columns(2)
    l1.metric("Highest Priority Issue", doc.get("highest_priority_issue", "—"))
    l2.metric("Highest Leverage Improvement", doc.get("highest_leverage_improvement", "—")[:60])
    l3, l4 = st.columns(2)
    l3.metric("Most Important Bottleneck", doc.get("most_important_bottleneck", "—"))
    l4.metric("Most Important Opportunity", doc.get("most_important_opportunity", "—")[:60])

    priorities: List[Dict[str, Any]] = doc.get("priorities") or []
    if priorities:
        st.markdown("#### Ranked Priorities")
        df = pd.DataFrame(
            [
                {
                    "Rank": p.get("priority_rank"),
                    "Focus Area": p.get("focus_area"),
                    "Expected Impact": p.get("expected_impact"),
                    "Category": p.get("category"),
                    "Rationale": p.get("rationale"),
                }
                for p in priorities
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        if "Expected Impact" in df.columns and not df.empty:
            impact_chart = df.set_index("Focus Area")["Expected Impact"]
            st.bar_chart(impact_chart)

    if doc.get("wisdom_alignment"):
        st.caption(f"Wisdom alignment: {doc.get('wisdom_alignment')}")

    history_path = results_dir / "strategic_priorities_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "top_impact" in hist.columns:
                st.markdown("#### Priority Impact Trend")
                chart_df = hist.tail(50).set_index("timestamp")["top_impact"]
                st.line_chart(chart_df)
        except Exception:
            pass

    st.info(
        "Strategic priorities are observational rankings only. "
        "Distinct from Strategic Oversight Center (phase 30)."
    )
