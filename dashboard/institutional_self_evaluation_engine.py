"""
TRITON Phase 208 — Institutional Self-Evaluation & Strategic Evolution dashboard.
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
    st.markdown("#### 🏛 [TRITON_INSTITUTIONAL_SELF_EVALUATION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    evo = report.get("triton_evolution") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evolution Score", str(evo.get("triton_evolution_score", "—")))
    c2.metric("Classification", str(evo.get("classification", "—")))
    c3.metric("Current Level", str(evo.get("current_institutional_level", "—")))
    c4.metric("Critical Gaps", str(len(ex.get("critical_strategic_gaps") or [])))


def _render_maturity(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Institutional Maturity Score")
    evo = report.get("triton_evolution") or {}
    comps = evo.get("components") or {}
    if comps:
        st.dataframe(
            pd.DataFrame([{"component": k, "score": v} for k, v in comps.items()]),
            use_container_width=True,
            hide_index=True,
        )


def _render_report_card(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Capability Report Card")
    maturity = report.get("institutional_maturity") or {}
    divisions = maturity.get("divisions") or []
    if not divisions:
        st.info("No maturity data.", icon="ℹ️")
        return
    rows = []
    for d in divisions:
        rows.append(
            {
                "division": d.get("division"),
                "grade": d.get("grade"),
                "score": d.get("composite_score"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for d in divisions:
        measures = d.get("measures") or {}
        if measures:
            with st.expander(f"{d.get('division')} — measures"):
                st.dataframe(
                    pd.DataFrame([{"measure": k, "score": v} for k, v in measures.items()]),
                    use_container_width=True,
                    hide_index=True,
                )


def _render_strengths(results_dir: Path) -> None:
    st.markdown("#### 💪 Strategic Strengths")
    df = _read_csv_safe(results_dir / "institutional_strengths.csv")
    if df.empty:
        st.info("No strengths data.", icon="ℹ️")
        return
    cols = [c for c in ["strength_id", "category", "title", "impact", "score"] if c in df.columns]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_weaknesses(results_dir: Path) -> None:
    st.markdown("#### ⚠️ Strategic Weaknesses")
    df = _read_csv_safe(results_dir / "institutional_weaknesses.csv")
    if df.empty:
        st.info("No weaknesses data.", icon="ℹ️")
        return
    cols = [
        c for c in ["weakness_id", "category", "title", "severity", "impact"] if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_roadmap(results_dir: Path) -> None:
    st.markdown("#### 🗺 Evolution Roadmap")
    df = _read_csv_safe(results_dir / "institutional_growth_roadmap.csv")
    if df.empty:
        st.info("No roadmap data.", icon="ℹ️")
        return
    horizon = st.selectbox("Horizon filter", ["All", "1yr", "3yr", "5yr"], key="p208_horizon")
    if horizon != "All" and "horizon" in df.columns:
        df = df[df["horizon"] == horizon]
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_capability_matrix(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔮 Future Capability Matrix")
    visions = (report.get("executive_recommendation") or {}).get("evolution_plans") or {}
    for label, key in [("1-Year", "1_year"), ("3-Year", "3_year"), ("5-Year", "5_year")]:
        plan = visions.get(key) or {}
        st.markdown(f"**{label}:** {plan.get('vision', '—')}")
        for cap in plan.get("capabilities") or []:
            st.markdown(f"- {cap}")


def _render_stagnation(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛑 Risk of Stagnation")
    critique = report.get("self_critique") or []
    stagnation = [c for c in critique if c.get("category") == "Stagnation Risk"]
    if stagnation:
        s = stagnation[0]
        st.warning(
            f"**{s.get('issue')}** — {s.get('institutional_impact')}. "
            f"Mitigation: {s.get('recommended_mitigation')}"
        )
    human = [c for c in critique if c.get("category") == "Human Oversight Required"]
    if human:
        st.markdown("**Human oversight must remain:**")
        for h in human:
            st.markdown(f"- {h.get('issue')}: {h.get('recommended_mitigation')}")


def _render_recommendations(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 💡 Executive Strategic Recommendations")
    df = _read_csv_safe(results_dir / "strategic_recommendations.csv")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    st.markdown(
        f"**Today:** {qs.get('what_is_triton_today', '—')} · "
        f"**Build next:** {qs.get('what_to_build_next', '—')} · "
        f"**Never change:** {qs.get('what_never_change', '—')}"
    )
    st.markdown(f"**Long-term vision:** {ex.get('long_term_vision', '—')}")


def render_institutional_self_evaluation_engine(results_dir: Path) -> None:
    st.markdown("### 🏛 TRITON Institutional Self-Evaluation & Strategic Evolution Engine")
    st.caption(
        "Institutional self-assessment and strategic evolution planning. "
        "Refresh with `python -m services.institutional_self_evaluation_engine`."
    )
    report = _read_json_safe(results_dir / "institutional_self_evaluation.json")
    if not report:
        st.warning(
            "No self-evaluation report. Run `python -m services.institutional_self_evaluation_engine`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_maturity(report)
    _render_report_card(report)
    _render_strengths(results_dir)
    _render_weaknesses(results_dir)
    _render_roadmap(results_dir)
    _render_capability_matrix(report)
    _render_stagnation(report)
    _render_recommendations(results_dir, report)
