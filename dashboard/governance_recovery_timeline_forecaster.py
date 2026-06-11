"""
TRITON Phase 182 — 🗓️ Governance Recovery Timeline dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _num


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


_CONF = {
    "VERY_HIGH": "🟢 VERY_HIGH",
    "HIGH": "🟢 HIGH",
    "MODERATE": "🟡 MODERATE",
    "LOW": "🔴 LOW",
}


def _render_current_state(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current State")
    cs = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current CPS", str(cs.get("current_cps", "—")))
    c2.metric("Governance", str(cs.get("current_governance_status", "—")))
    c3.metric("Escalation", str(cs.get("current_escalation", "—")))
    c4.metric("QQQ Concentration %", str(cs.get("qqq_concentration_pct", "—")))


def _render_timeline(report: Dict[str, Any]) -> None:
    st.markdown("#### 🗓️ Timeline Forecast")
    ex = report.get("executive_report") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected YELLOW Date", str(ex.get("expected_yellow_date", "—")))
    c2.metric("Expected GREEN Date", str(ex.get("expected_green_date", "—")))
    c3.metric("Days To YELLOW", str(ex.get("days_to_yellow", "—")))
    c4.metric("Days To GREEN", str(ex.get("days_to_green", "—")))
    cm = report.get("confidence_model") or {}
    fc = str(cm.get("forecast_confidence", "—"))
    st.metric("Forecast Confidence", _CONF.get(fc, fc))


def _render_strategy_comparison(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Strategy Comparison")
    df = _read_csv_safe(results_dir / "governance_recovery_timeline.csv")
    if df.empty:
        st.info("No timeline data.", icon="ℹ️")
        return
    sf = report.get("strategy_forecasts") or {}
    rows = []
    for key, fc in sf.items():
        rows.append(
            {
                "Strategy": key,
                "Days To YELLOW": fc.get("days_to_yellow"),
                "Days To GREEN": fc.get("days_to_green") or "—",
                "Transition Confidence": fc.get("transition_confidence_label"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if "forecast_day" in df.columns and "projected_cps" in df.columns:
        pivot = df.pivot_table(
            index="forecast_day", columns="strategy", values="projected_cps", aggfunc="first"
        )
        st.line_chart(pivot)


def _render_scenarios(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Best / Expected / Worst Case")
    for key, title in [
        ("best_case", "Best Case"),
        ("expected_case", "Expected Case"),
        ("worst_case", "Worst Case"),
    ]:
        sc = report.get(key) or {}
        with st.expander(f"**{title}**", expanded=key == "expected_case"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Days To YELLOW", str(sc.get("days_to_yellow", "—")))
            c2.metric("Days To GREEN", str(sc.get("days_to_green", "—")))
            conf = str(sc.get("timeline_confidence_label", "—"))
            c3.metric("Confidence", _CONF.get(conf, conf))
            if sc.get("cps_path"):
                path_df = pd.DataFrame(sc["cps_path"])
                if not path_df.empty:
                    st.line_chart(path_df.set_index("day"))


def _render_risks(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Risks & Accelerators")
    tr = report.get("timeline_risks") or {}
    st.markdown("**Timeline Risks**")
    st.json(tr)
    st.markdown("**Recovery Accelerators**")
    for a in report.get("accelerators") or []:
        st.markdown(f"- **{a.get('factor')}:** {a.get('impact')}")
    st.markdown("**Delay Factors**")
    for d in report.get("delay_factors") or []:
        st.markdown(f"- **{d.get('factor')}** ({d.get('severity')}): {d.get('impact')}")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Recommendation")
    ex = report.get("executive_report") or {}
    st.markdown(f"**[{ex.get('report_type', 'GOVERNANCE_RECOVERY_TIMELINE_REPORT')}]**")
    st.markdown(f"**Recommended strategy:** {ex.get('recommended_recovery_path')}")
    st.markdown(f"**Recommendation:** {ex.get('executive_recommendation')}")
    st.markdown(f"**Projected recovery:** {ex.get('projected_governance_recovery')}")
    cm = report.get("confidence_model") or {}
    c1, c2 = st.columns(2)
    c1.metric("Forecast Confidence", _CONF.get(str(cm.get("forecast_confidence")), "—"))
    c2.metric("Recovery Confidence", _CONF.get(str(cm.get("recovery_confidence")), "—"))


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_governance_recovery_timeline_forecaster(results_dir: Path) -> None:
    st.markdown("### 🗓️ Governance Recovery Timeline")
    st.caption(
        "Time-based governance recovery forecasting — analysis only. "
        "Refresh with `python -m services.governance_recovery_timeline_forecaster`."
    )

    report = _read_json_safe(results_dir / "governance_recovery_timeline.json")
    if not report:
        st.info(
            "No timeline report yet. Run "
            "`python -m services.governance_recovery_timeline_forecaster`.",
            icon="ℹ️",
        )
        return

    _render_current_state(report)
    _render_timeline(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "📊 Strategy Comparison",
            "📈 Best / Expected / Worst",
            "⚠️ Risks & Accelerators",
            "📋 Executive Recommendation",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="grtf_section",
    )
    if section == "📊 Strategy Comparison":
        _render_strategy_comparison(results_dir, report)
    elif section == "📈 Best / Expected / Worst":
        _render_scenarios(report)
    elif section == "⚠️ Risks & Accelerators":
        _render_risks(report)
    elif section == "📋 Executive Recommendation":
        _render_executive(report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
