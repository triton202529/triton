"""
TRITON Phase 205 — GREEN Authorization Strategy dashboard.
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
    st.markdown("#### 🟢 [GREEN_AUTHORIZATION_STRATEGY_REPORT]")
    ex = report.get("executive_recommendation") or {}
    dist = ex.get("current_distance_to_green") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS Gap to GREEN", str(dist.get("cps_gap", "—")))
    c2.metric("YELLOW Date", str(dist.get("yellow_date", "—")))
    c3.metric(
        "Expected GREEN",
        str(ex.get("expected_green_timeline", "—")),
    )
    c4.metric(
        "Sustainability",
        str(ex.get("sustainability_classification", "—")),
    )
    bito = ex.get("bito_dependency_analysis") or {}
    st.info(bito.get("explanation", "—"), icon="💡")


def _render_readiness(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 GREEN Readiness Score")
    maint = report.get("green_maintenance_framework") or {}
    c1, c2 = st.columns(2)
    c1.metric("Sustainability Score", str(maint.get("green_sustainability_score", "—")))
    c2.metric("Classification", str(maint.get("classification", "—")))
    res = report.get("green_resilience_testing") or {}
    st.caption(
        f"Resilience: **{res.get('classification', '—')}** "
        f"(score {res.get('overall_resilience_score', '—')})"
    )


def _render_distance(report: Dict[str, Any]) -> None:
    st.markdown("#### 📏 Distance to GREEN")
    base = report.get("baseline") or {}
    st.markdown(
        f"- Current CPS: **{base.get('current_cps')}** (ORANGE)\n"
        f"- YELLOW baseline CPS: **{base.get('yellow_cps')}** (gap **{base.get('cps_gap')}** pts)\n"
        f"- GREEN threshold: **{base.get('green_target_cps')}**"
    )


def _render_requirements(results_dir: Path) -> None:
    st.markdown("#### ✅ Certification Requirements")
    df = _read_csv_safe(results_dir / "green_certification_requirements.csv")
    if df.empty:
        st.info("No requirements data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_paths(results_dir: Path) -> None:
    st.markdown("#### 🛤 Portfolio Evolution Paths")
    df = _read_csv_safe(results_dir / "green_portfolio_evolution_paths.csv")
    if df.empty:
        st.info("No path data.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "rank",
            "path_id",
            "path_name",
            "probability_pct",
            "days_from_yellow_to_green",
            "projected_cps",
            "reaches_green",
            "expected_green_date",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_bito(report: Dict[str, Any]) -> None:
    st.markdown("#### 🪙 BITO Dependency Tracker")
    bito = report.get("bito_dependency_analysis") or {}
    st.markdown(
        f"- **Mandatory for GREEN?** {bito.get('bito_mandatory_for_green')}\n"
        f"- **Preferred?** {bito.get('bito_preferred_for_green')}\n"
        f"- **Status:** {bito.get('current_status')} · "
        f"Promotion prob (30d): {bito.get('promotion_probability_30d_pct')}%\n"
        f"- **If never promotes:** {bito.get('path_if_never_promotes')}"
    )


def _render_alternatives(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Alternative Candidate Analysis")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    st.markdown(f"**Alternatives:** {qs.get('alternative_candidates', '—')}")
    st.markdown(ex.get("alternative_candidate_strategy", "—"))


def _render_stress(report: Dict[str, Any]) -> None:
    st.markdown("#### 🧪 GREEN Stress Testing")
    tests = (report.get("green_resilience_testing") or {}).get("stress_tests") or []
    if tests:
        st.dataframe(pd.DataFrame(tests), use_container_width=True, hide_index=True)


def _render_sustainability(report: Dict[str, Any]) -> None:
    st.markdown("#### 🌿 Sustainability Framework")
    maint = report.get("green_maintenance_framework") or {}
    for action in maint.get("maintenance_actions") or []:
        st.markdown(f"- {action}")


def _render_failures(results_dir: Path) -> None:
    st.markdown("#### ⚠️ Failure Modes")
    df = _read_csv_safe(results_dir / "green_failure_modes.csv")
    if df.empty:
        st.info("No failure mode data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_recommendations(report: Dict[str, Any]) -> None:
    st.markdown("#### 💡 Executive Recommendation")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    st.markdown(
        f"**BITO mandatory?** {qs.get('bito_mandatory')} · "
        f"**Realistic GREEN:** {qs.get('realistic_green_timeline')} · "
        f"**Regression risk:** {qs.get('yellow_orange_regression_risk')}"
    )
    for p in ex.get("recommended_evidence_priorities") or []:
        st.markdown(f"- {p}")


def render_green_authorization_strategy_engine(results_dir: Path) -> None:
    st.markdown(
        "### 🟢 GREEN Authorization Strategy & Long-Term Portfolio Certification Intelligence"
    )
    st.caption(
        "Pathway from YELLOW to GREEN and long-term sustainability planning. "
        "Refresh with `python -m services.green_authorization_strategy_engine`."
    )
    report = _read_json_safe(results_dir / "green_authorization_strategy.json")
    if not report:
        st.warning(
            "No GREEN strategy report. Run `python -m services.green_authorization_strategy_engine`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_readiness(report)
    _render_distance(report)
    _render_requirements(results_dir)
    _render_paths(results_dir)
    _render_bito(report)
    _render_alternatives(report)
    _render_stress(report)
    _render_sustainability(report)
    _render_failures(results_dir)
    _render_recommendations(report)
