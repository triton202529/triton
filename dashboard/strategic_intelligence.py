"""
TRITON Strategic Intelligence dashboards — Phases 40–42.
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


def render_strategic_reasoning_center(results_dir: Path) -> None:
    st.markdown("### ♟ Strategic Reasoning Center")
    st.caption(
        "Strategic importance ranking across risks, weaknesses, governance issues, "
        "and institutional bottlenecks. **Advisory analysis only — no execution.**"
    )

    doc = _load_json(results_dir / "strategic_reasoning.json")
    if not doc:
        st.warning("Strategic reasoning data not found. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Strategic Issue", doc.get("top_strategic_issue", "—"))
    c2.metric("Strategic Importance", f"{doc.get('strategic_importance', 0)}%")
    c3.metric("Impact Scope", doc.get("impact_scope", "—"))

    issues: List[Dict[str, Any]] = doc.get("strategic_issues") or []
    if not issues:
        st.info("No strategic issues identified for current institutional posture.")
        return

    df = pd.DataFrame(
        [
            {
                "Issue": i.get("issue"),
                "Importance": i.get("importance"),
                "Scope": i.get("scope"),
                "Category": i.get("category"),
            }
            for i in issues
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    if "Importance" in df.columns and not df.empty:
        chart_df = df.set_index("Issue")["Importance"]
        st.bar_chart(chart_df)

    scope_counts = df["Scope"].value_counts() if "Scope" in df.columns else pd.Series(dtype=int)
    if not scope_counts.empty:
        st.markdown("**Impact scope distribution**")
        st.bar_chart(scope_counts)

    st.info("Strategic reasoning produces advisory rankings only. No broker actions are taken.")


def render_consequence_forecast_center(results_dir: Path) -> None:
    st.markdown("### 🔮 Consequence Forecast Center")
    st.caption(
        "90-day consequence projections if current institutional conditions persist. "
        "**Simulation-only forecasts — no automated intervention.**"
    )

    doc = _load_json(results_dir / "consequence_forecasts.json")
    if not doc:
        st.warning("Consequence forecast data not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    st.metric("Forecast Horizon (days)", doc.get("forecast_horizon_days", 90))

    forecasts: List[Dict[str, Any]] = doc.get("forecasts") or []
    if not forecasts:
        st.info("No consequence forecasts generated for current conditions.")
        return

    for forecast in forecasts:
        issue = forecast.get("issue", "—")
        severity = forecast.get("severity", "—")
        confidence = forecast.get("confidence", 0)
        horizon = forecast.get("forecast_horizon_days", 90)
        consequence = forecast.get("likely_consequence", "")

        with st.expander(f"**{issue}** — {severity} ({confidence}% confidence)", expanded=True):
            st.write(consequence)
            st.caption(f"Horizon: {horizon} days")

    df = pd.DataFrame(
        [
            {
                "Issue": f.get("issue"),
                "Severity": f.get("severity"),
                "Confidence": f.get("confidence"),
                "Horizon (days)": f.get("forecast_horizon_days"),
                "Likely Consequence": f.get("likely_consequence"),
            }
            for f in forecasts
        ]
    )
    st.markdown("#### Forecast Summary")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if "Confidence" in df.columns and not df.empty:
        conf_chart = df.set_index("Issue")["Confidence"]
        st.line_chart(conf_chart)

    st.warning(
        "Consequence forecasts are advisory projections. "
        "They do not trigger trades or portfolio modifications."
    )


def render_institutional_wisdom_center(results_dir: Path) -> None:
    st.markdown("### 📜 Institutional Wisdom Center")
    st.caption(
        "Long-term institutional guidance synthesized from memory, learning, "
        "strategic reasoning, and consequence forecasts. **Advisory only.**"
    )

    doc = _load_json(results_dir / "institutional_wisdom.json")
    if not doc:
        st.warning("Institutional wisdom data not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{doc.get('confidence', 0)}%")
    c2.metric("Supporting Systems", doc.get("supporting_systems", 0))

    st.markdown("#### Wisdom Statement")
    st.info(doc.get("wisdom_statement", "—"))

    themes: List[Dict[str, Any]] = doc.get("wisdom_themes") or []
    if themes:
        st.markdown("#### Wisdom Themes")
        theme_df = pd.DataFrame(
            [
                {
                    "Theme": t.get("theme"),
                    "Source": t.get("source"),
                    "Weight": t.get("weight"),
                }
                for t in themes
            ]
        )
        st.dataframe(theme_df, use_container_width=True, hide_index=True)

    guidance: List[Dict[str, Any]] = doc.get("guidance_items") or []
    if guidance:
        st.markdown("#### Guidance Items")
        for item in sorted(guidance, key=lambda x: x.get("priority") or 99):
            horizon = item.get("horizon_days")
            horizon_txt = f" ({horizon}d)" if horizon else ""
            st.markdown(
                f"**{item.get('priority')}.** [{item.get('category', 'general')}] "
                f"{item.get('guidance', '')}{horizon_txt}"
            )

    history_path = results_dir / "institutional_wisdom_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "confidence" in hist.columns:
                st.markdown("#### Wisdom Confidence Trend")
                chart_df = hist.tail(50).set_index("timestamp")["confidence"]
                st.line_chart(chart_df)
        except Exception:
            pass

    st.info("Institutional wisdom is observational guidance. No automated trading or intervention.")
