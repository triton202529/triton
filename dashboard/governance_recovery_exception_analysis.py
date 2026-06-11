"""
TRITON Phase 180 — Governance Recovery Exception Analysis dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _num, _plotly_bar


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


_DEADLOCK = {
    "NO_DEADLOCK": "🟢 NO_DEADLOCK",
    "PARTIAL_DEADLOCK": "🟡 PARTIAL_DEADLOCK",
    "FULL_RECOVERY_DEADLOCK": "🔴 FULL_RECOVERY_DEADLOCK",
}

_SAFETY = {
    "SAFE": "🟢 SAFE",
    "CONDITIONALLY_SAFE": "🟡 CONDITIONALLY_SAFE",
    "HIGH_RISK": "🟠 HIGH_RISK",
    "UNACCEPTABLE": "🔴 UNACCEPTABLE",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [GOVERNANCE_RECOVERY_EXCEPTION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Recovery Without Exception?",
        "❌ No" if not qs.get("can_recovery_occur_without_exceptions") else "✅ Yes",
    )
    c2.metric(
        "Exception Improves Recovery?",
        "✅ Yes" if qs.get("would_exception_materially_improve_recovery") else "❌ No",
    )
    c3.metric(
        "Governance Intact?",
        "✅ Yes" if qs.get("governance_remains_intact") else "❌ No",
    )
    st.markdown(
        f"**Safest path:** {qs.get('safest_exception_design') or '—'} (`{qs.get('safest_exception_path') or '—'}`)"
    )
    st.markdown(
        f"**Most effective path:** {qs.get('most_effective_exception_design') or '—'} "
        f"(`{qs.get('most_effective_exception_path') or '—'}`)"
    )
    st.markdown(f"**Minimum authority:** {qs.get('minimum_authority_required') or '—'}")
    st.info(str(ex.get("recommended_approach") or ""), icon="💡")


def _render_deadlock(report: Dict[str, Any], results_dir: Path) -> None:
    st.markdown("#### 🔒 Current Deadlock Analysis")
    dl = report.get("deadlock_analysis") or {}
    cls = str(dl.get("deadlock_classification") or "—")
    c1, c2, c3 = st.columns(3)
    c1.metric("Deadlock Status", _DEADLOCK.get(cls, cls))
    c2.metric("Paradox Detected", "Yes" if dl.get("governance_paradox_detected") else "No")
    c3.metric(
        "Recovery Under Current Rules",
        "Yes" if dl.get("recovery_possible_under_current_rules") else "No",
    )
    cycle = dl.get("self_reinforcing_cycle") or []
    if cycle:
        st.markdown("**Self-reinforcing cycle:**")
        st.markdown(" → ".join(cycle))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Controls blocking recovery**")
        for x in dl.get("controls_blocking_recovery") or []:
            st.markdown(f"- {x}")
    with c2:
        st.markdown("**Controls protecting recovery**")
        for x in dl.get("controls_protecting_recovery") or []:
            st.markdown(f"- {x}")
    df = _read_csv_safe(results_dir / "governance_recovery_deadlock_analysis.csv")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_paths(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🔀 Exception Path Comparison")
    df = _read_csv_safe(results_dir / "governance_recovery_exception_paths.csv")
    if df.empty:
        df = pd.DataFrame(report.get("exception_paths") or [])
    if df.empty:
        st.info("No exception paths modeled.", icon="ℹ️")
        return
    view = df.copy()
    if "safety_classification" in view.columns:
        view["safety_classification"] = view["safety_classification"].map(
            lambda x: _SAFETY.get(str(x), str(x))
        )
    st.dataframe(view, use_container_width=True, hide_index=True)
    if "recovery_probability_score" in df.columns:
        chart = df[["exception_path", "recovery_probability_score"]].copy()
        if not _plotly_bar(
            chart, "exception_path", "recovery_probability_score", "Recovery Probability by Path"
        ):
            st.bar_chart(chart.set_index("exception_path")["recovery_probability_score"])


def _render_safety(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Safety Analysis")
    sa = report.get("safety_analysis") or {}
    st.metric("Overall Safety", _SAFETY.get(str(sa.get("overall_classification")), "—"))
    for pc in sa.get("path_classifications") or []:
        st.markdown(
            f"- **{pc.get('exception_path')}:** "
            f"{_SAFETY.get(str(pc.get('safety_classification')), pc.get('safety_classification'))} — "
            f"{pc.get('notes') or ''}"
        )
    st.markdown("**Abuse vectors**")
    for v in sa.get("abuse_vectors") or []:
        with st.expander(str(v.get("vector"))):
            st.markdown(f"**Q:** {v.get('question')}")
            st.markdown(f"**Assessment:** {v.get('assessment')}")
            st.markdown(f"**Mitigation:** {v.get('mitigation')}")


def _render_recovery_impact(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Recovery Impact")
    ri = report.get("recovery_impact") or {}
    base = ri.get("baseline_without_exception") or {}
    st.markdown(
        f"**Without exception:** days_to_yellow={base.get('days_to_yellow')} · "
        f"probability={base.get('recovery_probability')}"
    )
    df = pd.DataFrame(ri.get("path_impacts") or [])
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_governance_impact(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏛 Governance Impact")
    cs = report.get("current_state") or {}
    gm = cs.get("governance_model") or {}
    auth = gm.get("authorization_rules") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS", str(cs.get("cps", "—")))
    c2.metric("Escalation", str(cs.get("escalation", "—")))
    c3.metric("Governance Auth", str(cs.get("governance_authorized")))
    c4.metric("Concentration", f"{cs.get('largest_concentration_pct', '—')}%")
    st.json(auth)


def _render_current_state(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current State")
    cs = report.get("current_state") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Open NEW", ", ".join(cs.get("open_new_candidates") or []) or "—")
    c2.metric("Available Slots", str(cs.get("available_slots", "—")))
    c3.metric("Recovery Basket", ", ".join(cs.get("recovery_basket") or []) or "—")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_governance_recovery_exception_analysis(results_dir: Path) -> None:
    st.markdown("### 🧩 Governance Recovery Exception Analysis")
    st.caption(
        "Design-only analysis of governance-safe exception paths for ORANGE recovery deadlock. "
        "Refresh with `python -m services.governance_recovery_exception_analysis`."
    )
    report = _read_json_safe(results_dir / "governance_recovery_exception_analysis.json")
    if not report:
        st.warning(
            "No analysis found. Run `python -m services.governance_recovery_exception_analysis`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_current_state(report)
    _render_deadlock(report, results_dir)
    _render_paths(results_dir, report)
    _render_safety(report)
    _render_recovery_impact(report)
    _render_governance_impact(report)
    _render_certification(report)
