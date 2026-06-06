"""
TRITON Institutional Autonomy dashboards — Phases 22–24.
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


def _result_badge(result: str) -> str:
    return {
        "PASS": "🟢 PASS",
        "WARN": "🟡 WARN",
        "FAIL": "🔴 FAIL",
    }.get(str(result).upper(), result)


def _cert_badge(status: str) -> str:
    return {
        "NOT_CERTIFIED": "⛔ NOT CERTIFIED",
        "PARTIALLY_CERTIFIED": "🟡 PARTIALLY CERTIFIED",
        "CERTIFIED_FOR_PAPER_PROTECTION": "🟢 CERTIFIED (PAPER)",
    }.get(str(status).upper(), status)


def render_capital_preservation_audit_center(results_dir: Path) -> None:
    st.markdown("### 📋 Capital Preservation Audit Center")
    st.caption(
        "Unified audit trail across alerts, escalations, advisories, simulations, and governor. "
        "**Read-only — no execution.**"
    )

    doc = _load_json(results_dir / "capital_preservation_audit.json")
    if not doc:
        st.warning("Audit data not found. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Event Count", doc.get("event_count", 0))
    c2.metric("Latest Type", doc.get("latest_event_type", "—"))
    c3.metric("Latest Result", doc.get("latest_event_result", "—"))

    summary = doc.get("summary_by_event_type") or {}
    if summary:
        st.markdown("#### Events by Type")
        st.dataframe(
            pd.DataFrame([{"Event Type": k, "Count": v} for k, v in sorted(summary.items())]),
            use_container_width=True,
            hide_index=True,
        )

    events: List[Dict[str, Any]] = doc.get("events") or []
    if events:
        st.markdown("#### Audit Events")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Timestamp": e.get("audit_timestamp"),
                        "Type": e.get("event_type"),
                        "Source": e.get("event_source"),
                        "Result": e.get("event_result"),
                        "Details": (e.get("details") or "")[:80],
                    }
                    for e in events[:100]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    history_path = results_dir / "capital_preservation_audit_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "event_count" in hist.columns:
                st.markdown("#### Audit Volume Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["event_count"])
        except Exception:
            pass

    st.info("Audit events are derived from preservation stack artifacts. No broker actions.")


def render_preservation_stress_lab(results_dir: Path) -> None:
    st.markdown("### 🧪 Preservation Stress Lab")
    st.caption(
        "Counterfactual stress scenarios against current preservation state. "
        "**Simulation only — no portfolio changes.**"
    )

    doc = _load_json(results_dir / "stress_test_results.json")
    if not doc:
        st.warning("Stress test data not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenarios", doc.get("scenario_count", 0))
    c2.metric("Pass", doc.get("pass_count", 0))
    c3.metric("Warn", doc.get("warn_count", 0))
    c4.metric("Fail", doc.get("fail_count", 0))

    c_a, c_b = st.columns(2)
    c_a.metric("Avg Survivability", doc.get("average_survivability", 0))
    c_b.metric("Baseline CPS", doc.get("baseline_cps", "—"))

    scenarios: List[Dict[str, Any]] = doc.get("scenarios") or []
    if scenarios:
        st.markdown("#### Scenario Results")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Scenario": s.get("scenario_name"),
                        "Score": s.get("survivability_score"),
                        "Result": s.get("result"),
                        "Details": s.get("details"),
                    }
                    for s in scenarios
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        for s in scenarios:
            with st.expander(
                f"{s.get('scenario_name')} — {_result_badge(s.get('result', ''))} "
                f"({s.get('survivability_score')}/100)",
                expanded=False,
            ):
                st.markdown(f"**Details:** {s.get('details')}")

    st.warning("Stress tests evaluate resilience only. No trades or orders are placed.")


def render_preservation_certification_center(results_dir: Path) -> None:
    st.markdown("### 🏅 Preservation Certification Center")
    st.caption(
        "Certification scoring for paper-mode capital preservation capabilities. "
        "**Live execution remains blocked by default.**"
    )

    doc = _load_json(results_dir / "capital_preservation_certification.json")
    if not doc:
        st.warning("Certification data not found. Run the watchdog first.")
        return

    status = str(doc.get("certification_status") or "NOT_CERTIFIED")
    st.metric("Certification Status", _cert_badge(status))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Certification Score", doc.get("certification_score", 0))
    c2.metric(
        "Certified Areas", f"{doc.get('certified_area_count', 0)}/{doc.get('total_areas', 8)}"
    )
    c3.metric("Escalation", doc.get("escalation_state", "—"))

    areas: Dict[str, Any] = doc.get("areas") or {}
    if areas:
        st.markdown("#### Certification Areas")
        rows = []
        for name, area in areas.items():
            if not isinstance(area, dict):
                continue
            rows.append(
                {
                    "Area": name,
                    "Certified": "Yes" if area.get("certified") else "No",
                    "Score": area.get("score"),
                    "Notes": area.get("notes"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    failed = doc.get("failed_requirements") or []
    if failed:
        st.markdown("#### Failed Requirements")
        for req in failed:
            st.markdown(f"- **{req}**")

    history_path = results_dir / "capital_preservation_certification_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "certification_score" in hist.columns:
                st.markdown("#### Certification Score Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["certification_score"])
        except Exception:
            pass

    st.info(
        "Full certification requires all areas plus blocked live execution gates. "
        "Default posture is NOT_CERTIFIED."
    )
