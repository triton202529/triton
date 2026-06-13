"""
TRITON Phase 202 — Governance Recovery Acceleration dashboard.
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
    st.markdown("#### 🚀 [GOVERNANCE_RECOVERY_ACCELERATION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    pos = ex.get("current_recovery_position") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS", str(pos.get("cps", "—")))
    c2.metric("Recovery Score", str(pos.get("recovery_score", "—")))
    c3.metric("QQQ %", str(pos.get("qqq_concentration_pct", "—")))
    c4.metric("Δ YELLOW", str(pos.get("distance_to_yellow", "—")))
    fastest = ex.get("fastest_path_to_yellow") or {}
    st.info(
        f"Fastest YELLOW: **{fastest.get('scenario', '—')}** in **{fastest.get('days', '—')} days** "
        f"({fastest.get('date', '—')})",
        icon="💡",
    )


def _render_bottlenecks(results_dir: Path) -> None:
    st.markdown("#### 🚧 Bottleneck Rankings")
    df = _read_csv_safe(results_dir / "recovery_bottleneck_rankings.csv")
    if df.empty:
        st.info("No bottleneck data.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "rank",
            "bottleneck",
            "classification",
            "current_impact_score",
            "recovery_delay_days",
            "acceleration_potential",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_impact_matrix(results_dir: Path) -> None:
    st.markdown("#### 📊 Evidence Impact Matrix")
    df = _read_csv_safe(results_dir / "evidence_impact_matrix.csv")
    if df.empty:
        st.info("No impact matrix.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_scenarios(results_dir: Path) -> None:
    st.markdown("#### 🛤 Recovery Acceleration Scenarios")
    df = _read_csv_safe(results_dir / "recovery_acceleration_scenarios.csv")
    if df.empty:
        st.info("No scenarios.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_yellow_paths(report: Dict[str, Any]) -> None:
    st.markdown("#### 🟡 YELLOW Acceleration Paths")
    scenarios = report.get("acceleration_scenarios") or []
    yellow = sorted(scenarios, key=lambda s: s.get("days_to_yellow", 999))[:4]
    for s in yellow:
        st.markdown(
            f"- **{s.get('scenario_name')}** — {s.get('days_to_yellow')} days → "
            f"{s.get('new_yellow_date')} (Δ CPS {s.get('change_in_cps')})"
        )


def _render_green_paths(report: Dict[str, Any]) -> None:
    st.markdown("#### 🟢 GREEN Acceleration Paths")
    scenarios = report.get("acceleration_scenarios") or []
    green = sorted(scenarios, key=lambda s: s.get("days_to_green", 999))[:4]
    for s in green:
        st.markdown(
            f"- **{s.get('scenario_name')}** — {s.get('days_to_green')} days → "
            f"{s.get('new_green_date')} (CPS {s.get('projected_cps')})"
        )


def _render_probability(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Recovery Probability Changes")
    scenarios = report.get("acceleration_scenarios") or []
    if scenarios:
        df = pd.DataFrame(
            [
                {
                    "scenario": s.get("scenario_name"),
                    "auth_prob_Δ": s.get("authorization_probability_change"),
                    "confidence": s.get("confidence_level"),
                }
                for s in scenarios
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_recommendations(report: Dict[str, Any]) -> None:
    st.markdown("#### 💡 Executive Recommendations")
    ex = report.get("executive_recommendation") or {}
    lev = report.get("evidence_leverage") or {}
    for p in ex.get("recommended_evidence_priorities") or []:
        st.markdown(f"- {p}")
    st.markdown(
        f"**Cost of inaction:** {lev.get('highest_cost_of_inaction', '—')} · "
        f"**Single best lever:** {(lev.get('single_largest_benefit') or {}).get('improvement', '—')}"
    )


def render_governance_recovery_acceleration_engine(results_dir: Path) -> None:
    st.markdown("### 🚀 Governance Recovery Acceleration & Bottleneck Intelligence")
    st.caption(
        "Ranks bottlenecks and simulates evidence acceleration paths. "
        "Refresh with `python -m services.governance_recovery_acceleration_engine`."
    )
    report = _read_json_safe(results_dir / "governance_recovery_acceleration.json")
    if not report:
        st.warning(
            "No acceleration report. Run `python -m services.governance_recovery_acceleration_engine`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_bottlenecks(results_dir)
    _render_impact_matrix(results_dir)
    _render_scenarios(results_dir)
    _render_yellow_paths(report)
    _render_green_paths(report)
    _render_probability(report)
    _render_recommendations(report)
