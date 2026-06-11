"""
TRITON Phase 186 — ⚠️ Governance Risk Prediction Engine dashboard.
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
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


_CLASS = {
    "CRITICAL": "🔴 CRITICAL",
    "HIGH": "🟠 HIGH",
    "MODERATE": "🟡 MODERATE",
    "LOW": "🟢 LOW",
}


def _render_rankings(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏆 Risk Rankings")
    ranked = sorted(
        [r for r in (report.get("risk_predictions") or []) if r.get("time_horizon") == 30],
        key=lambda r: float(r.get("risk_score") or 0),
        reverse=True,
    )
    if not ranked:
        st.info("No rankings.", icon="ℹ️")
        return
    df = pd.DataFrame(ranked)
    if "classification" in df.columns:
        df["classification"] = df["classification"].map(lambda x: _CLASS.get(str(x), str(x)))
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_matrix(results_dir: Path) -> None:
    st.markdown("#### 🧮 Risk Matrix")
    mx = _read_csv_safe(results_dir / "governance_risk_matrix.csv")
    if mx.empty:
        st.info("No matrix.", icon="ℹ️")
        return
    sub = mx[mx["time_horizon"] == 30] if "time_horizon" in mx.columns else mx
    if "risk_type" in sub.columns and "matrix_score" in sub.columns:
        chart = sub.set_index("risk_type")[["matrix_score"]].sort_values("matrix_score")
        st.bar_chart(chart)
    st.dataframe(sub, use_container_width=True, hide_index=True)


def _render_warnings(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔔 Early Warnings")
    warnings = report.get("early_warnings") or []
    if not warnings:
        st.success("No active early warnings.", icon="✅")
        return
    for w in warnings:
        st.warning(w, icon="⚠️")


def _render_timeline_risks(report: Dict[str, Any]) -> None:
    st.markdown("#### 🗓️ Timeline Risks")
    impacts = report.get("risk_impacts") or {}
    st.json(impacts.get("impact_on_timeline") or {})
    st.markdown("**Impact on YELLOW**")
    st.json(impacts.get("impact_on_yellow") or {})
    st.markdown("**Impact on GREEN**")
    st.json(impacts.get("impact_on_green") or {})


def _render_recovery_threats(report: Dict[str, Any]) -> None:
    st.markdown("#### 🚨 Recovery Threats")
    ex = report.get("executive_report") or {}
    for t in ex.get("expected_recovery_threats") or []:
        st.markdown(f"- {t}")
    st.markdown("**Operator monitoring**")
    for item in ex.get("operator_monitoring_checklist") or []:
        st.markdown(f"- {item}")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Recommendation")
    ex = report.get("executive_report") or {}
    st.markdown(f"**[{ex.get('report_type', 'GOVERNANCE_RISK_PREDICTION_REPORT')}]**")
    st.markdown(f"**Top risks (30d):** {', '.join(ex.get('top_risks') or [])}")
    hp = ex.get("highest_probability_risk") or {}
    hs = ex.get("highest_severity_risk") or {}
    c1, c2 = st.columns(2)
    c1.metric("Highest Probability", f"{hp.get('type', '—')} ({hp.get('probability')}%)")
    c2.metric("Highest Severity", f"{hs.get('type', '—')} ({hs.get('severity')})")
    st.markdown(f"**Forecast stability:** {ex.get('forecast_stability')}")
    st.markdown(f"**Recommendation:** {ex.get('executive_recommendation')}")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_governance_risk_prediction_engine(results_dir: Path) -> None:
    st.markdown("### ⚠️ Governance Risk Prediction Engine")
    st.caption(
        "Forward-looking governance risk forecasting — analysis only. "
        "Refresh with `python -m services.governance_risk_prediction_engine`."
    )

    report = _read_json_safe(results_dir / "governance_risk_predictions.json")
    if not report:
        st.info(
            "No risk report yet. Run `python -m services.governance_risk_prediction_engine`.",
            icon="ℹ️",
        )
        return

    section = st.radio(
        "Section",
        [
            "🏆 Risk Rankings",
            "🧮 Risk Matrix",
            "🔔 Early Warnings",
            "🗓️ Timeline Risks",
            "🚨 Recovery Threats",
            "📋 Executive Recommendation",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="grpe_section",
    )
    if section == "🏆 Risk Rankings":
        _render_rankings(report)
    elif section == "🧮 Risk Matrix":
        _render_matrix(results_dir)
    elif section == "🔔 Early Warnings":
        _render_warnings(report)
    elif section == "🗓️ Timeline Risks":
        _render_timeline_risks(report)
    elif section == "🚨 Recovery Threats":
        _render_recovery_threats(report)
    elif section == "📋 Executive Recommendation":
        _render_executive(report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
