"""
TRITON Phase 199 — Governance Authorization Intelligence dashboard.
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


def _band_color(band: str) -> str:
    return {
        "INSTITUTIONAL_GRADE": "🟢",
        "MATURE": "🟡",
        "DEVELOPING": "🟠",
        "INSUFFICIENT": "🔴",
    }.get(band, "⚪")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 🧠 [GOVERNANCE_AUTHORIZATION_INTELLIGENCE_REPORT]")
    ex = report.get("executive_recommendation") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Authorization", str(ex.get("current_authorization_status", "—")))
    c2.metric("Block Intentional", "YES" if ex.get("block_intentional") else "NO")
    c3.metric("Auth Probability", str(ex.get("authorization_probability_pct", "—")))
    c4.metric(
        "Contradiction",
        str(report.get("contradiction_detection", {}).get("classification", "—")),
    )
    st.info(ex.get("institutional_learning", ""), icon="💡")
    eq = ex.get("executive_questions") or {}
    st.markdown(
        f"**Why blocked:** {eq.get('why_blocked', '—')} · "
        f"**Intentional?** {eq.get('intentional_or_system_problem', '—')}"
    )


def _render_status(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔐 Authorization Status")
    pkg = report.get("authorization_status") or {}
    st.markdown(f"**Status:** {pkg.get('current_status', '—')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Governance", "YES" if pkg.get("governance_authorized") else "NO")
    c2.metric("Execution", "YES" if pkg.get("execution_authorized") else "NO")
    c3.metric("Paper", "YES" if pkg.get("paper_execution_permitted") else "NO")
    st.caption(pkg.get("summary", ""))


def _render_chain(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔗 Decision Chain")
    chain = report.get("decision_chain") or []
    if not chain:
        st.info("No chain data.", icon="ℹ️")
        return
    df = pd.DataFrame(chain)
    cols = [
        c
        for c in [
            "layer",
            "status",
            "confidence",
            "blocking_reason",
            "downstream_impact",
            "evidence_source",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_blocking(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🚧 Blocking Factors")
    blocking = report.get("blocking_analysis") or {}
    primary = blocking.get("primary_blocker") or {}
    if primary:
        st.error(
            f"Primary: **{primary.get('layer')}** — {primary.get('blocking_reason')} "
            f"({primary.get('classification')})",
            icon="⚠️",
        )
    df = _read_csv_safe(results_dir / "governance_blocking_chain.csv")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_cps(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡 CPS & Risk Alignment")
    cps = report.get("cps_alignment") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS Score", str(cps.get("cps_score", "—")))
    c2.metric("Escalation", str(cps.get("escalation_state", "—")))
    c3.metric("Δ to YELLOW", str(cps.get("distance_to_yellow_cps_points", "—")))
    c4.metric("Δ to GREEN", str(cps.get("distance_to_green_cps_points", "—")))
    st.markdown(
        f"Expected YELLOW: **{cps.get('expected_yellow_date') or '—'}** "
        f"({cps.get('days_to_yellow', '—')} days) · "
        f"GREEN: **{cps.get('expected_green_date') or '—'}** "
        f"({cps.get('days_to_green', '—')} days)"
    )
    st.caption(cps.get("projected_governance_recovery") or "")


def _render_requirements(results_dir: Path) -> None:
    st.markdown("#### 📋 Recovery Requirements")
    df = _read_csv_safe(results_dir / "governance_recovery_requirements.csv")
    if df.empty:
        st.info("No requirements.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_contradiction(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Governance Contradictions")
    c = report.get("contradiction_detection") or {}
    st.markdown(f"**Classification:** {c.get('classification', '—')}")
    st.markdown(
        f"Execution posture: **{c.get('execution_posture')}** · Governance: **{c.get('governance_posture')}**"
    )
    st.info(c.get("explanation", ""), icon="ℹ️")


def _render_evidence(report: Dict[str, Any]) -> None:
    st.markdown("#### 📦 Evidence Package")
    pkg = report.get("authorization_status") or {}
    missing = (report.get("executive_recommendation") or {}).get("evidence_missing") or []
    if missing:
        for m in missing:
            st.markdown(f"- {m}")
    else:
        st.success("No critical evidence gaps identified beyond standard recovery path.", icon="✅")


def _render_probability(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Authorization Probability")
    scores = report.get("intelligence_scores") or {}
    band = str(scores.get("classification", "—"))
    st.markdown(
        f"{_band_color(band)} **{band}** — composite **{scores.get('composite_intelligence_score', '—')}**"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transparency", str(scores.get("governance_transparency_score", "—")))
    c2.metric("Auth Confidence", str(scores.get("authorization_confidence_score", "—")))
    c3.metric("Evidence", str(scores.get("evidence_completeness_score", "—")))
    c4.metric("Recovery Ready", str(scores.get("recovery_readiness_score", "—")))
    ex = report.get("executive_recommendation") or {}
    st.markdown(f"Recovery probability: **{ex.get('authorization_probability_pct', '—')}%**")


def render_governance_authorization_intelligence(results_dir: Path) -> None:
    st.markdown("### 🧠 Governance Authorization Intelligence & Alignment")
    st.caption(
        "Explains why governance is blocked and the evidence path to authorization. "
        "Refresh with `python -m services.governance_authorization_intelligence`."
    )
    report = _read_json_safe(results_dir / "governance_authorization_intelligence.json")
    if not report:
        st.warning(
            "No intelligence report. Run `python -m services.governance_authorization_intelligence`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_status(report)
    _render_chain(report)
    _render_blocking(results_dir, report)
    _render_cps(report)
    _render_requirements(results_dir)
    _render_contradiction(report)
    _render_evidence(report)
    _render_probability(report)
