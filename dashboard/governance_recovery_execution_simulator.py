"""
TRITON Phase 204 — Governance Recovery Execution Simulation dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


def _read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 [GOVERNANCE_RECOVERY_EXECUTION_SIMULATION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    pos = ex.get("current_position") or {}
    timeline = ex.get("most_likely_recovery_timeline") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Day 0 CPS", str(pos.get("cps", "—")))
    c2.metric("Expected YELLOW", str(ex.get("expected_yellow_date", "—")))
    c3.metric("Expected GREEN", str(ex.get("expected_green_date", "—")))
    c4.metric("Confidence", str(ex.get("certification_readiness", "—")))
    st.info(
        f"Primary path: **{timeline.get('scenario', '—')}** — YELLOW day **{timeline.get('yellow_day', '—')}**, "
        f"GREEN day **{timeline.get('green_day', '—')}**",
        icon="💡",
    )


def _render_timeline(results_dir: Path) -> None:
    st.markdown("#### 📅 Day-by-Day Recovery Timeline")
    df = _read_csv_safe(results_dir / "governance_recovery_daily_projection.csv")
    if df.empty:
        st.info("No daily projection.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "day",
            "date",
            "cps",
            "cps_momentum",
            "qqq_concentration_pct",
            "escalation_state",
            "authorization_probability_pct",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_cps(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 CPS Projection")
    daily = report.get("daily_projection") or []
    if daily:
        df = pd.DataFrame([{"day": d["day"], "CPS": d["cps"]} for d in daily])
        st.line_chart(df.set_index("day"))


def _render_concentration(report: Dict[str, Any]) -> None:
    st.markdown("#### 📉 Concentration Reduction Curve")
    daily = report.get("daily_projection") or []
    if daily:
        df = pd.DataFrame(
            [
                {
                    "day": d["day"],
                    "QQQ %": d.get("qqq_concentration_pct"),
                    "HD %": d.get("hd_weight_pct"),
                    "XLK %": d.get("xlk_weight_pct"),
                }
                for d in daily
            ]
        )
        st.line_chart(df.set_index("day"))


def _render_transitions(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Governance Transition Forecast")
    transitions = report.get("governance_transitions") or []
    if transitions:
        st.dataframe(pd.DataFrame(transitions), use_container_width=True, hide_index=True)


def _render_checkpoints(results_dir: Path) -> None:
    st.markdown("#### ✅ Certification Checkpoints")
    df = _read_csv_safe(results_dir / "governance_recovery_certification_checkpoints.csv")
    if df.empty:
        st.info("No checkpoints.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_failures(results_dir: Path) -> None:
    st.markdown("#### ⚠️ Failure Scenarios")
    df = _read_csv_safe(results_dir / "governance_recovery_failure_branches.csv")
    if df.empty:
        st.info("No failure branches.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_confidence(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎓 Recovery Confidence")
    conf = report.get("confidence_model") or {}
    if conf:
        df = pd.DataFrame(
            [
                {"metric": k, "score": v}
                for k, v in conf.items()
                if k.endswith("_score") or k == "classification"
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_recommendations(report: Dict[str, Any]) -> None:
    st.markdown("#### 💡 Executive Recommendation")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    st.markdown(f"**30-day outlook:** {qs.get('next_30_days_summary', '—')}")
    for p in ex.get("recommended_monitoring_focus") or []:
        st.markdown(f"- {p}")


def render_governance_recovery_execution_simulator(results_dir: Path) -> None:
    st.markdown("### 🎯 Governance Recovery Execution Simulation & Certification Rehearsal")
    st.caption(
        "Day-by-day ORANGE→YELLOW→GREEN rehearsal. "
        "Refresh with `python -m services.governance_recovery_execution_simulator`."
    )
    report = _read_json_safe(results_dir / "governance_recovery_simulation.json")
    if not report:
        st.warning(
            "No simulation report. Run `python -m services.governance_recovery_execution_simulator`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_timeline(results_dir)
    _render_cps(report)
    _render_concentration(report)
    _render_transitions(report)
    _render_checkpoints(results_dir)
    _render_failures(results_dir)
    _render_confidence(report)
    _render_recommendations(report)
