"""
TRITON Institutional Optimization dashboards — Phases 46–48.
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


def render_attention_allocation_center(results_dir: Path) -> None:
    st.markdown("### 🎯 Attention Allocation Center")
    st.caption(
        "Attention scoring across governance, readiness, certification, risk, oversight, "
        "and preservation. **Advisory simulation only — distinct from Strategic Priorities "
        "and Strategic Oversight centers.**"
    )

    doc = _load_json(results_dir / "attention_allocation.json")
    if not doc:
        st.warning("Attention allocation data not found. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest Attention Area", doc.get("highest_attention_area", "—"))
    c2.metric("Attention Score", doc.get("attention_score", "—"))
    c3.metric("Recommended Focus %", doc.get("recommended_focus_percent", "—"))

    allocations: List[Dict[str, Any]] = doc.get("allocations") or []
    if not allocations:
        st.info("No attention allocations generated for current posture.")
        return

    df = pd.DataFrame(
        [
            {
                "Area": a.get("area"),
                "Attention Score": a.get("attention_score"),
                "Focus %": a.get("recommended_focus_percent"),
                "Rationale": a.get("rationale"),
            }
            for a in allocations
        ]
    )
    st.markdown("#### Attention Allocations")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if "Attention Score" in df.columns and not df.empty:
        score_chart = df.set_index("Area")["Attention Score"]
        st.bar_chart(score_chart)
        focus_chart = df.set_index("Area")["Focus %"]
        st.bar_chart(focus_chart)

    st.info(
        "Attention allocation guides advisory focus only. "
        "No broker actions, live execution, or portfolio modifications."
    )


def render_system_coordination_center(results_dir: Path) -> None:
    st.markdown("### 🔗 System Coordination Center")
    st.caption(
        "Cross-system coordination across governance, oversight, preservation, "
        "certification, and planning dimensions. **Simulation-only analysis.**"
    )

    doc = _load_json(results_dir / "system_coordination.json")
    if not doc:
        st.warning("System coordination data not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Coordination Score", doc.get("coordination_score", "—"))
    c2.metric("Strongest Connection", doc.get("strongest_connection", "—"))
    c3.metric("Weakest Connection", doc.get("weakest_connection", "—"))

    dimensions = doc.get("dimensions") or {}
    if dimensions:
        st.markdown("#### Coordination Dimensions")
        dim_df = pd.DataFrame(
            [{"Dimension": k.replace("_", " ").title(), "Score": v} for k, v in dimensions.items()]
        )
        st.dataframe(dim_df, use_container_width=True, hide_index=True)
        if "Score" in dim_df.columns and not dim_df.empty:
            st.bar_chart(dim_df.set_index("Dimension")["Score"])

    connections: List[Dict[str, Any]] = doc.get("connections") or []
    if connections:
        st.markdown("#### Cross-System Connections")
        conn_df = pd.DataFrame(
            [
                {
                    "From": c.get("from"),
                    "To": c.get("to"),
                    "Strength": c.get("strength"),
                    "Edges": c.get("edge_count"),
                }
                for c in connections
            ]
        )
        st.dataframe(conn_df, use_container_width=True, hide_index=True)

    st.warning(
        "System coordination is observational alignment analysis. "
        "No automated intervention or trading."
    )


def render_institutional_optimization_center(results_dir: Path) -> None:
    st.markdown("### ⚡ Institutional Optimization Center")
    st.caption(
        "Optimization opportunities synthesized from priorities, self-improvement, "
        "maturity, future paths, and attention allocation. **Advisory only.**"
    )

    doc = _load_json(results_dir / "institutional_optimization.json")
    if not doc:
        st.warning("Institutional optimization data not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Optimization", doc.get("top_optimization", "—"))
    c2.metric("Optimization Score", doc.get("optimization_score", "—"))
    c3.metric("Expected System Benefit", doc.get("expected_system_benefit", "—"))

    st.markdown("#### Optimization Lenses")
    l1, l2 = st.columns(2)
    l1.metric("Highest ROI", doc.get("highest_roi_improvement", "—"))
    l2.metric("Highest Leverage", (doc.get("highest_leverage_improvement") or "—")[:60])
    l3, l4 = st.columns(2)
    l3.metric("Fastest Improvement", doc.get("fastest_improvement", "—"))
    l4.metric("Longest Term", doc.get("longest_term_improvement", "—"))

    optimizations: List[Dict[str, Any]] = doc.get("optimizations") or []
    if optimizations:
        st.markdown("#### Ranked Optimizations")
        opt_df = pd.DataFrame(
            [
                {
                    "Type": o.get("type"),
                    "Focus": o.get("focus"),
                    "Score": o.get("score"),
                    "Expected Benefit": o.get("expected_benefit"),
                    "Timeframe": o.get("timeframe"),
                    "Rationale": o.get("rationale"),
                }
                for o in optimizations
            ]
        )
        st.dataframe(opt_df, use_container_width=True, hide_index=True)
        if "Score" in opt_df.columns and not opt_df.empty:
            st.bar_chart(opt_df.set_index("Focus")["Score"])

    history_path = results_dir / "institutional_optimization_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "optimization_score" in hist.columns:
                st.markdown("#### Optimization Score Trend")
                chart_df = hist.tail(50).set_index("timestamp")["optimization_score"]
                st.line_chart(chart_df)
        except Exception:
            pass

    st.info(
        "Institutional optimization is observational prioritization only. "
        "No trades, orders, or automated intervention."
    )
