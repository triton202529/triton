"""
TRITON Capital Preservation Evaluation dashboards — Phases 19–21.
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


def _eval_badge(state: str) -> str:
    return {
        "BENEFICIAL": "🟢 BENEFICIAL",
        "NEUTRAL": "🟡 NEUTRAL",
        "NEGATIVE": "🔴 NEGATIVE",
    }.get(str(state).upper(), state)


def _posture_badge(posture: str) -> str:
    return {
        "GREEN": "🟢 GREEN",
        "YELLOW": "🟡 YELLOW",
        "ORANGE": "🟠 ORANGE",
        "RED": "🔴 RED",
        "CRITICAL": "⛔ CRITICAL",
    }.get(str(posture).upper(), posture)


def render_protective_action_evaluation(results_dir: Path) -> None:
    st.markdown("### 📊 Protective Action Evaluation")
    st.caption(
        "Effectiveness scoring for paper-mode protective trials. "
        "**Evaluation only — no execution.**"
    )

    doc = _load_json(results_dir / "protective_action_evaluation.json")
    if not doc:
        st.warning("Evaluation data not found. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluations", doc.get("evaluation_count", 0))
    c2.metric("Avg Effectiveness", doc.get("average_effectiveness", 0))
    c3.metric("Beneficial", doc.get("beneficial_count", 0))
    c4.metric("Negative", doc.get("negative_count", 0))

    evaluations: List[Dict[str, Any]] = doc.get("evaluations") or []
    if evaluations:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Trial": e.get("trial_name"),
                        "Score": e.get("effectiveness_score"),
                        "CPS Δ": e.get("cps_improvement"),
                        "Risk Reduction": e.get("risk_reduction"),
                        "Concentration Δ": e.get("concentration_reduction"),
                        "Drawdown Δ": e.get("drawdown_reduction"),
                        "Stability Δ": e.get("stability_improvement"),
                        "Evaluation": e.get("evaluation"),
                    }
                    for e in evaluations
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        for ev in evaluations:
            with st.expander(
                f"{ev.get('trial_name')} — {ev.get('effectiveness_score')} ({ev.get('evaluation')})",
                expanded=False,
            ):
                st.markdown(f"**Evaluation:** {_eval_badge(ev.get('evaluation', ''))}")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Effectiveness", ev.get("effectiveness_score"))
                m2.metric("CPS Improvement", ev.get("cps_improvement"))
                m3.metric("Risk Reduction", ev.get("risk_reduction"))
                m4.metric("Concentration Δ", ev.get("concentration_reduction"))
                m5.metric("Drawdown Δ", ev.get("drawdown_reduction"))

    history_path = results_dir / "protective_action_evaluation_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "average_effectiveness" in hist.columns:
                st.markdown("#### Effectiveness Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["average_effectiveness"])
        except Exception:
            pass

    st.info("All evaluations are based on paper simulations. No broker actions are performed.")


def render_adaptive_capital_preservation(results_dir: Path) -> None:
    st.markdown("### 🧠 Adaptive Capital Preservation")
    st.caption(
        "Learning from protective trial simulations to identify effective protections. "
        "**No automated actions.**"
    )

    doc = _load_json(results_dir / "adaptive_capital_preservation.json")
    if not doc:
        st.warning("Adaptive preservation data not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Protection", doc.get("best_protection", "—"))
    c2.metric("Avg Effectiveness", doc.get("average_effectiveness", 0))
    c3.metric("Confidence", f"{doc.get('confidence', 0)}%")

    if doc.get("learning_summary"):
        st.markdown(f"**Summary:** {doc.get('learning_summary')}")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### Most Effective")
        rows = doc.get("most_effective_protections") or []
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No data")
    with col_b:
        st.markdown("#### Lowest Risk")
        rows = doc.get("lowest_risk_protections") or []
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No data")
    with col_c:
        st.markdown("#### Highest CPS Gain")
        rows = doc.get("highest_cps_improvements") or []
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No data")

    st.warning("Adaptive insights inform review only. No portfolio or order changes are made.")


def render_capital_preservation_governor(results_dir: Path) -> None:
    st.markdown("### 👑 Capital Preservation Governor")
    st.caption(
        "Unified preservation posture from CPI, escalation, trials, and readiness. "
        "**Advisory only — paper mode.**"
    )

    doc = _load_json(results_dir / "capital_preservation_governor.json")
    if not doc:
        st.warning("Governor data not found. Run the watchdog first.")
        return

    posture = str(doc.get("preservation_posture") or "UNKNOWN")
    st.metric("Preservation Posture", _posture_badge(posture))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governor Confidence", f"{doc.get('governor_confidence', 0)}%")
    c2.metric("CPS", doc.get("capital_preservation_score"))
    c3.metric("Escalation", doc.get("escalation_state"))
    c4.metric("Readiness", doc.get("readiness_status"))

    drivers = doc.get("top_drivers") or []
    if drivers:
        st.markdown("#### Top Drivers")
        for d in drivers:
            st.markdown(f"- **{d}**")

    actions = doc.get("recommended_review_actions") or []
    if actions:
        st.markdown("#### Recommended Review Actions")
        for a in actions:
            st.markdown(f"- {a}")

    c_a, c_b = st.columns(2)
    c_a.metric("Best Protection (Adaptive)", doc.get("best_protection", "—"))
    c_b.metric(
        "Live Execution Permitted",
        "No" if not doc.get("live_execution_permitted") else "Yes",
    )

    history_path = results_dir / "capital_preservation_governor_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "governor_confidence" in hist.columns:
                st.markdown("#### Governor Confidence Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["governor_confidence"])
        except Exception:
            pass

    st.info("Governor posture is advisory. No trades or portfolio modifications are executed.")
