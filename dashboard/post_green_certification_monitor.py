"""
TRITON Phase 206 — Post-GREEN Institutional Stability dashboard.
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
    st.markdown("#### 🏛 [POST_GREEN_INSTITUTIONAL_STABILITY_REPORT]")
    ex = report.get("executive_recommendation") or {}
    stab = report.get("institutional_stability") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", str(ex.get("current_certification_status", "—")))
    c2.metric("Stability Score", str(stab.get("institutional_stability_score", "—")))
    c3.metric("Classification", str(stab.get("classification", "—")))
    c4.metric("Regression Prob %", str(stab.get("regression_probability_pct", "—")))


def _render_stability_score(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 GREEN Stability Score")
    stab = report.get("institutional_stability") or {}
    comps = stab.get("components") or {}
    if comps:
        df = pd.DataFrame([{"component": k, "score": v} for k, v in comps.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_timeline(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📅 Certification Timeline")
    mem = report.get("governance_memory") or {}
    cert = report.get("certification_status") or {}
    st.markdown(
        f"- **Mode:** {cert.get('certification_status', '—')}\n"
        f"- **Days GREEN:** {mem.get('days_continuously_green', 0)}\n"
        f"- **Projected GREEN date:** {mem.get('projected_certification_date', '—')}\n"
        f"- **Longevity score:** {mem.get('green_longevity_score', '—')}"
    )
    df = _read_csv_safe(results_dir / "post_green_stability_history.csv")
    if not df.empty:
        st.dataframe(df.tail(14), use_container_width=True, hide_index=True)


def _render_regression(results_dir: Path) -> None:
    st.markdown("#### ⚠️ Regression Warning Center")
    df = _read_csv_safe(results_dir / "post_green_regression_risks.csv")
    if df.empty:
        st.info("No regression data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_concentration(report: Dict[str, Any]) -> None:
    st.markdown("#### 📉 Concentration Drift Monitor")
    pi = (report.get("continuous_monitoring") or {}).get("portfolio_integrity") or {}
    cert = report.get("certification_status") or {}
    st.markdown(
        f"QQQ: **{pi.get('concentration_pct', cert.get('qqq_concentration_pct'))}%** "
        f"(limit 35%) · Positions: **{pi.get('position_count', '—')}**"
    )


def _render_cps(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 CPS Resilience Tracker")
    pi = (report.get("continuous_monitoring") or {}).get("portfolio_integrity") or {}
    cert = report.get("certification_status") or {}
    st.markdown(
        f"CPS: **{pi.get('cps', cert.get('cps'))}** · Trend: **{pi.get('cps_trend', '—')}** · "
        f"Buffer: **{cert.get('cps_buffer_above_green', '—')}** (need +5 at GREEN)"
    )


def _render_replacement(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Replacement Candidate Health")
    li = (report.get("continuous_monitoring") or {}).get("lifecycle_integrity") or {}
    st.markdown(
        f"OPEN_NEW: **{li.get('open_new_count', '—')}** · "
        f"BUY-effective: **{li.get('buy_effective_count', '—')}** · "
        f"BITO WATCH: **{li.get('bito_watch_status', '—')}** · "
        f"Deterioration risk: **{li.get('deterioration_risk', '—')}**"
    )


def _render_stress(report: Dict[str, Any]) -> None:
    st.markdown("#### 🧪 Stress Test Results")
    tests = report.get("stress_survival_testing") or []
    if tests:
        st.dataframe(pd.DataFrame(tests), use_container_width=True, hide_index=True)


def _render_memory(report: Dict[str, Any]) -> None:
    st.markdown("#### 📚 Institutional Memory")
    mem = report.get("governance_memory") or {}
    for lesson in mem.get("institutional_lessons") or []:
        st.markdown(f"- {lesson}")


def _render_recommendations(report: Dict[str, Any]) -> None:
    st.markdown("#### 💡 Executive Recommendation")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    st.markdown(
        f"**GREEN stable?** {qs.get('is_green_stable')} · "
        f"**Monitoring mode:** {qs.get('monitoring_mode')} · "
        f"**Shock resilience:** {qs.get('shock_resilience')}"
    )
    for p in ex.get("recommended_preventative_actions") or []:
        st.markdown(f"- {p}")


def render_post_green_certification_monitor(results_dir: Path) -> None:
    st.markdown("### 🏛 Post-GREEN Institutional Stability & Continuous Certification Intelligence")
    st.caption(
        "Permanent GREEN durability monitoring and regression early warning. "
        "Refresh with `python -m services.post_green_certification_monitor`."
    )
    report = _read_json_safe(results_dir / "post_green_certification_status.json")
    if not report:
        st.warning(
            "No post-GREEN report. Run `python -m services.post_green_certification_monitor`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_stability_score(report)
    _render_timeline(results_dir, report)
    _render_regression(results_dir)
    _render_concentration(report)
    _render_cps(report)
    _render_replacement(report)
    _render_stress(report)
    _render_memory(report)
    _render_recommendations(report)
