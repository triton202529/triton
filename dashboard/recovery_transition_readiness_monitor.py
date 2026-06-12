"""
TRITON Phase 193 — Recovery Transition Live Authorization Readiness Monitor dashboard.
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
    st.markdown("#### 🟢 [RECOVERY_TRANSITION_READINESS_REPORT]")
    ex = report.get("executive_recommendation") or {}
    st1 = ex.get("current_stage_1_status") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Classification", str(st1.get("classification", "—")))
    c2.metric("Readiness Score", str(st1.get("authorization_readiness_score", "—")))
    c3.metric("Stage 1 Eligible", "YES" if st1.get("stage_1_eligible") else "NO")
    st.info(str(ex.get("recommended_operator_action") or ""), icon="💡")
    st.warning("Monitor only — does NOT grant approval or activate exception.", icon="⚠️")
    qs = ex.get("executive_questions") or {}
    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_gates(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔐 Authorization Gate Matrix")
    gates = report.get("authorization_gates") or {}
    rows = []
    for key in [
        "operator_approval",
        "risk_office_approval",
        "market_session",
        "pep_lifecycle",
        "execution_infrastructure",
    ]:
        g = gates.get(key) or {}
        rows.append(
            {
                "gate": g.get("gate_name", key),
                "status": g.get("status"),
                "detail": g.get("detail"),
                "blocking_reason": g.get("blocking_reason"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_score(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Readiness Score")
    scores = report.get("readiness_scores") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall", str(scores.get("authorization_readiness_score", "—")))
    c2.metric("Governance", str(scores.get("governance_readiness", "—")))
    c3.metric("Market", str(scores.get("market_readiness", "—")))
    c4.metric("Lifecycle", str(scores.get("lifecycle_readiness", "—")))
    c5.metric("Execution", str(scores.get("execution_readiness", "—")))


def _render_pep(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 PEP Lifecycle Watch")
    pep = report.get("pep_lifecycle_watch") or {}
    st.markdown(
        f"**{pep.get('symbol')}** — {pep.get('effective_stance')} · signal {pep.get('signal')} · "
        f"conf {pep.get('confidence')} · valid={pep.get('lifecycle_valid')}"
    )


def _render_market(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Market Session Status")
    ex = report.get("executive_recommendation") or {}
    mkt = ex.get("market_status") or {}
    st.metric("Market Open", "YES" if mkt.get("open") else "NO")


def _render_execution(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚙️ Execution Health")
    ex = report.get("executive_recommendation") or {}
    eh = ex.get("execution_health") or {}
    c1, c2 = st.columns(2)
    c1.metric("Infrastructure Gate", str(eh.get("status", "—")))
    c2.metric("Stale Orders", str(eh.get("stale_orders", 0)))


def _render_blockers(report: Dict[str, Any]) -> None:
    st.markdown("#### ⛔ Current Blockers")
    blockers = report.get("blockers") or []
    if not blockers:
        st.success("No blockers — all gates pass.", icon="✅")
        return
    for b in blockers:
        st.markdown(f"- **{b.get('gate')}** ({b.get('severity')}): {b.get('reason')}")


def _render_alerts(results_dir: Path) -> None:
    st.markdown("#### 🔔 Transition Alerts")
    df = _read_csv_safe(results_dir / "recovery_transition_readiness_alerts.csv")
    if df.empty:
        st.info("No alerts yet.", icon="ℹ️")
        return
    show = df.tail(20).iloc[::-1]
    st.dataframe(show, use_container_width=True, hide_index=True)


def _render_history(results_dir: Path) -> None:
    st.markdown("#### 📈 Readiness History")
    df = _read_csv_safe(results_dir / "recovery_transition_readiness_history.csv")
    if df.empty:
        st.info("No history yet.", icon="ℹ️")
        return
    st.line_chart(df.set_index("timestamp")["readiness_score"].tail(50))


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))
    status = report.get("current_status") or {}
    if status.get("minutes_since_last_validation") is not None:
        st.caption(f"Last validation: {status.get('minutes_since_last_validation')} min ago")


def render_recovery_transition_readiness_monitor(results_dir: Path) -> None:
    st.markdown("### 🟢 Recovery Transition Live Authorization Monitor")
    st.caption(
        "Real-time Stage 1 authorization gate monitor. "
        "Refresh with `python -m services.recovery_transition_readiness_monitor`."
    )
    report = _read_json_safe(results_dir / "recovery_transition_readiness.json")
    if not report:
        st.warning(
            "No readiness data. Run `python -m services.recovery_transition_readiness_monitor`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_gates(report)
    _render_score(report)
    _render_pep(report)
    _render_market(report)
    _render_execution(report)
    _render_blockers(report)
    _render_alerts(results_dir)
    _render_history(results_dir)
    _render_certification(report)
