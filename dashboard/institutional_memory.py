"""
TRITON Institutional Memory dashboards — Phases 34–36.
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


def render_institutional_memory(results_dir: Path) -> None:
    st.markdown("### 🧠 Institutional Memory")
    st.caption(
        "Persistent organizational memory from risk events, escalations, governance decisions, "
        "and certification outcomes. **Observational only — no execution.**"
    )

    doc = _load_json(results_dir / "institutional_memory.json")
    if not doc:
        st.warning("Institutional memory data not found. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Memory Entries", doc.get("memory_entries", 0))
    c2.metric("Last Major Event", doc.get("last_major_event", "—"))
    c3.metric("Retention Status", doc.get("retention_status", "—"))
    c4.metric("Displayed", len(doc.get("entries") or []))

    entries: List[Dict[str, Any]] = doc.get("entries") or []
    if entries:
        st.markdown("#### Recent Memory Entries")
        df = pd.DataFrame(
            [
                {
                    "Timestamp": e.get("timestamp"),
                    "Type": e.get("event_type"),
                    "Source": e.get("event_source"),
                    "Summary": e.get("summary"),
                    "Severity": e.get("severity") or "—",
                }
                for e in entries[:100]
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        type_counts = df["Type"].value_counts()
        if not type_counts.empty:
            st.markdown("#### Event Type Distribution")
            st.bar_chart(type_counts)

    history_path = results_dir / "institutional_memory_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty:
                st.markdown("#### Memory Retention History")
                st.caption(f"Total retained rows: **{len(hist)}**")
        except Exception:
            pass

    st.info(
        "Institutional memory records events for human review only. No broker actions are taken."
    )


def render_institutional_knowledge_graph(results_dir: Path) -> None:
    st.markdown("### 🕸 Institutional Knowledge Graph")
    st.caption(
        "Relationship map between alerts, escalations, governance, certification, and oversight "
        "components. **Diagnostic only.**"
    )

    doc = _load_json(results_dir / "institutional_knowledge_graph.json")
    if not doc:
        st.warning("Knowledge graph data not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", doc.get("nodes", 0))
    c2.metric("Relationships", doc.get("relationships", 0))
    c3.metric("Most Connected Area", doc.get("most_connected_area", "—"))

    if doc.get("truncated"):
        st.caption("Graph display is capped; counts reflect full graph size.")

    graph = doc.get("graph") or {}
    nodes: List[Dict[str, Any]] = graph.get("nodes") or []
    edges: List[Dict[str, Any]] = graph.get("edges") or []

    if nodes:
        st.markdown("#### Sample Nodes")
        node_df = pd.DataFrame(
            [
                {
                    "ID": n.get("id"),
                    "Type": n.get("type"),
                    "Label": n.get("label"),
                    "Area": n.get("area"),
                }
                for n in nodes[:50]
            ]
        )
        st.dataframe(node_df, use_container_width=True, hide_index=True)

        area_counts = node_df["Area"].value_counts()
        if not area_counts.empty:
            st.bar_chart(area_counts)

    if edges:
        st.markdown("#### Sample Relationships")
        edge_df = pd.DataFrame(
            [
                {
                    "Source": e.get("source"),
                    "Type": e.get("type"),
                    "Target": e.get("target"),
                }
                for e in edges[:50]
            ]
        )
        st.dataframe(edge_df, use_container_width=True, hide_index=True)

        rel_counts = edge_df["Type"].value_counts()
        if not rel_counts.empty:
            st.markdown("#### Relationship Types")
            st.bar_chart(rel_counts)

    st.info("Knowledge graph is read-only. No automated actions are triggered.")


def render_organizational_learning_center(results_dir: Path) -> None:
    st.markdown("### 📚 Organizational Learning Center")
    st.caption(
        "Pattern analysis from institutional memory, audit history, and recurring governance "
        "themes. **Advisory learning only.**"
    )

    doc = _load_json(results_dir / "organizational_learning.json")
    if not doc:
        st.warning("Organizational learning data not found. Run the watchdog first.")
        return

    st.metric("Top Lesson", doc.get("top_lesson", "—"))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{doc.get('confidence', 0)}%")
    c2.metric("Learning Status", doc.get("learning_status", "—"))
    c3.metric("Top Priority", doc.get("top_priority", "—"))

    patterns: Dict[str, Any] = doc.get("patterns") or {}

    col_a, col_b = st.columns(2)
    with col_a:
        failures = patterns.get("repeated_failures") or []
        if failures:
            st.markdown("#### Repeated Failures")
            for item in failures:
                st.markdown(f"- **{item}**")

        concerns = patterns.get("most_common_governance_concerns") or []
        if concerns:
            st.markdown("#### Common Governance Concerns")
            for item in concerns:
                st.markdown(f"- {item}")

    with col_b:
        strengths = patterns.get("repeated_strengths") or []
        if strengths:
            st.markdown("#### Repeated Strengths")
            for item in strengths:
                st.markdown(f"- {item}")

        blockers = patterns.get("most_common_certification_blockers") or []
        if blockers:
            st.markdown("#### Certification Blockers")
            for item in blockers:
                st.markdown(f"- {item}")

    lessons = patterns.get("highest_value_lessons") or []
    if lessons:
        st.markdown("#### Highest-Value Lessons")
        for lesson in lessons:
            st.markdown(f"- {lesson}")

    st.warning(
        "Organizational learning does not execute trades or modify portfolios. "
        "All insights require human review."
    )
