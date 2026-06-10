"""
TRITON Phase 175 — 🟢 Diversification Recovery Basket dashboard.
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


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    ex = report.get("executive_report") or {}
    gov = ex.get("current_governance_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Escalation", str(gov.get("escalation_state", "—")))
    c2.metric("Current CPS", str(ex.get("current_cps", "—")))
    c3.metric("Concentration", f"{_num(ex.get('current_concentration_pct'))}%")
    c4.metric("Projected CPS", str(ex.get("projected_cps", "—")))
    st.markdown(f"- **Best Candidate:** {ex.get('best_recovery_candidate') or '—'}")
    st.markdown(f"- **Best Basket:** {', '.join(ex.get('best_recovery_basket') or []) or '—'}")
    st.markdown(
        f"- **Projected Governance:** {ex.get('projected_governance_status')} "
        f"({ex.get('projected_escalation_state')})"
    )
    st.markdown(
        f"- **ORANGE→YELLOW min names:** {ex.get('orange_to_yellow_minimum_names')} · "
        f"**YELLOW→GREEN min names:** {ex.get('yellow_to_green_minimum_names')}"
    )


def _render_concentration(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current Concentration")
    ex = report.get("executive_report") or {}
    cause = ex.get("concentration_cause")
    st.markdown(f"**Cause:** {cause}")
    gov = ex.get("current_governance_state") or {}
    st.caption(f"Block: {gov.get('block_reason') or '—'}")


def _render_basket(report: Dict[str, Any]) -> None:
    st.markdown("#### 🧺 Recovery Basket V1")
    basket = report.get("recovery_basket_v1") or {}
    members = pd.DataFrame(basket.get("members") or [])
    if members.empty:
        st.info("No basket designed.", icon="ℹ️")
        return
    st.dataframe(members, use_container_width=True, hide_index=True)
    m = basket.get("metrics") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("CPS", str(m.get("capital_preservation_score", "—")))
    c2.metric("Largest Weight", f"{_num(m.get('largest_position_weight_pct'))}%")
    c3.metric("Governance OK", str(m.get("governance_authorized_estimate", "—")))


def _render_scenarios(results_dir: Path) -> None:
    st.markdown("#### 🔬 Scenario Analysis")
    df = _read_csv_safe(results_dir / "diversification_scenarios.csv")
    if df.empty:
        st.info("No scenarios.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "capital_preservation_score" in df.columns:
        chart = df[["scenario", "capital_preservation_score"]].copy()
        chart["scenario"] = chart["scenario"].str.replace("Scenario ", "S")
        if not _plotly_bar(chart, "scenario", "capital_preservation_score", "CPS by Scenario"):
            st.bar_chart(chart.set_index("scenario")["capital_preservation_score"])


def _render_governance(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Governance Impact")
    thresholds = report.get("governance_recovery_thresholds") or {}
    for label in ("orange_to_yellow", "yellow_to_green"):
        block = thresholds.get(label) or {}
        st.markdown(f"**{label.replace('_', ' ').title()}**")
        st.json(block)


def _render_cp(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Capital Preservation Impact")
    cp = report.get("capital_preservation_assessment") or {}
    verdict = str(cp.get("verdict", "—"))
    badge = {"POSITIVE": "🟢", "NEUTRAL": "🟡", "NEGATIVE": "🔴"}.get(verdict, "⚪")
    st.metric("Assessment", f"{badge} {verdict}")
    st.caption(str(cp.get("summary") or ""))
    factors = cp.get("factors") or {}
    if factors:
        st.dataframe(
            pd.DataFrame([{"factor": k, "value": v} for k, v in factors.items()]),
            use_container_width=True,
            hide_index=True,
        )


def _render_fastest(report: Dict[str, Any]) -> None:
    st.markdown("#### 🚀 Fastest Path To Green")
    path = report.get("fastest_path_to_green") or {}
    st.json(path)


def _render_candidates(results_dir: Path) -> None:
    st.markdown("#### 🎯 Best Diversification Candidates")
    df = _read_csv_safe(results_dir / "diversification_recovery_candidates.csv")
    if df.empty:
        st.info("No OPEN_NEW candidates ranked.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    status = str(cert.get("status", "—"))
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        status, "⚪"
    )
    st.metric("Status", f"{badge} {status}")
    for r in cert.get("reasons") or []:
        st.warning(r, icon="⚠️")
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_diversification_recovery_basket(results_dir: Path) -> None:
    st.markdown("### 🟢 Diversification Recovery Basket")
    st.caption(
        "Governance recovery basket design — analysis only. "
        "Refresh with `python -m services.diversification_recovery_basket`."
    )

    report = _read_json_safe(results_dir / "diversification_recovery_basket.json")
    if not report:
        st.info(
            "No report yet. Run `python -m services.diversification_recovery_basket` to generate it.",
            icon="ℹ️",
        )
        return

    _render_executive(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "📊 Current Concentration",
            "🎯 Candidates",
            "🧺 Recovery Basket",
            "🔬 Scenarios",
            "⚖️ Governance Impact",
            "🛡️ Capital Preservation",
            "🚀 Fastest Path",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="drb_section",
    )
    if section == "📊 Current Concentration":
        _render_concentration(report)
    elif section == "🎯 Candidates":
        _render_candidates(results_dir)
    elif section == "🧺 Recovery Basket":
        _render_basket(report)
    elif section == "🔬 Scenarios":
        _render_scenarios(results_dir)
    elif section == "⚖️ Governance Impact":
        _render_governance(report)
    elif section == "🛡️ Capital Preservation":
        _render_cp(report)
    elif section == "🚀 Fastest Path":
        _render_fastest(report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
