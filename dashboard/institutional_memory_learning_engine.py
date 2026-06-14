"""
TRITON Phase 207 — Institutional Memory & Governance Learning dashboard.
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
    st.markdown("#### 🧠 [INSTITUTIONAL_MEMORY_AND_LEARNING_REPORT]")
    ex = report.get("executive_recommendation") or {}
    wisdom = report.get("institutional_wisdom") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wisdom Score", str(wisdom.get("institutional_wisdom_score", "—")))
    c2.metric("Classification", str(wisdom.get("classification", "—")))
    c3.metric("Events", str(ex.get("total_events_catalogued", "—")))
    c4.metric("Failures Prevented", str(ex.get("historical_failures_prevented", "—")))


def _render_wisdom(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Institutional Wisdom Score")
    wisdom = report.get("institutional_wisdom") or {}
    comps = wisdom.get("components") or {}
    if comps:
        st.dataframe(
            pd.DataFrame([{"component": k, "score": v} for k, v in comps.items()]),
            use_container_width=True,
            hide_index=True,
        )


def _render_timeline(results_dir: Path) -> None:
    st.markdown("#### 📅 Historical Timeline")
    df = _read_csv_safe(results_dir / "institutional_event_history.csv")
    if df.empty:
        st.info("No event history.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "rank",
            "event_type",
            "title",
            "severity",
            "recovery_success",
            "phase_source",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_lessons(results_dir: Path) -> None:
    st.markdown("#### 💡 Major Lessons Learned")
    df = _read_csv_safe(results_dir / "governance_lessons_learned.csv")
    if df.empty:
        st.info("No lessons data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_patterns(results_dir: Path) -> None:
    st.markdown("#### 🔁 Repeated Failure Patterns")
    df = _read_csv_safe(results_dir / "repeated_failure_patterns.csv")
    if df.empty:
        st.info("No pattern data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_decisions(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Decision Quality Review")
    reviews = report.get("decision_reviews") or []
    if reviews:
        st.dataframe(pd.DataFrame(reviews), use_container_width=True, hide_index=True)


def _render_knowledge(results_dir: Path) -> None:
    st.markdown("#### 📚 Knowledge Base")
    df = _read_csv_safe(results_dir / "institutional_knowledge_base.csv")
    if df.empty:
        st.info("No knowledge base.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_prevention(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡 Prevention Strategies")
    patterns = report.get("pattern_recognition") or []
    for p in patterns[:5]:
        st.markdown(
            f"- **{p.get('pattern_name')}** ({p.get('pattern_category')}): "
            f"{p.get('prevention_strategy')}"
        )


def _render_recommendations(report: Dict[str, Any]) -> None:
    st.markdown("#### 💡 Executive Recommendation")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    st.markdown(
        f"**Learned:** {qs.get('what_has_triton_learned', '—')} · "
        f"**Recurring risk:** {qs.get('recurring_risks', '—')} · "
        f"**Preserved:** {qs.get('lessons_preserved', '—')}"
    )
    for p in ex.get("recommendations_for_future_operations") or []:
        st.markdown(f"- {p}")


def render_institutional_memory_learning_engine(results_dir: Path) -> None:
    st.markdown("### 🧠 Institutional Memory & Governance Learning Engine")
    st.caption(
        "Governance history, pattern recognition, and institutional wisdom. "
        "Refresh with `python -m services.institutional_memory_learning_engine`."
    )
    report = _read_json_safe(results_dir / "institutional_memory_report.json")
    if not report:
        st.warning(
            "No memory report. Run `python -m services.institutional_memory_learning_engine`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_wisdom(report)
    _render_timeline(results_dir)
    _render_lessons(results_dir)
    _render_patterns(results_dir)
    _render_decisions(report)
    _render_knowledge(results_dir)
    _render_prevention(report)
    _render_recommendations(report)
