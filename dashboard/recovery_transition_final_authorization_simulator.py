"""
TRITON Phase 192 — Final Recovery Authorization Simulation dashboard (read-only).
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
    st.markdown("#### 🎬 [FINAL_RECOVERY_AUTHORIZATION_SIMULATION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    state = ex.get("current_state") or report.get("current_state") or {}
    go = ex.get("go_nogo_outcome") or {}
    st.markdown(
        f"**Portfolio:** {state.get('weights')} · CPS {state.get('current_cps')} · "
        f"{state.get('escalation')} · {state.get('governance_status')}"
    )
    c1, c2 = st.columns(2)
    c1.metric("Real-World Decision", str(go.get("real_world", "—")))
    c2.metric("Dress Rehearsal", str(go.get("dress_rehearsal", "—")))
    st.info(str(ex.get("recommended_decision") or ""), icon="💡")
    st.warning(
        "Simulation only — no authorization granted, exception not activated.",
        icon="⚠️",
    )
    qs = ex.get("executive_questions") or {}
    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_timeline(report: Dict[str, Any]) -> None:
    st.markdown("#### ⏱ Authorization Timeline")
    chain = (report.get("executive_recommendation") or {}).get("authorization_chain") or []
    sim = report.get("authorization_simulation") or {}
    steps = sim.get("steps") or chain
    df = pd.DataFrame(steps)
    if df.empty:
        st.info("No timeline.", icon="ℹ️")
        return
    cols = [
        c
        for c in ["step", "name", "outcome", "real_outcome", "simulated_outcome"]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_dual(report: Dict[str, Any]) -> None:
    st.markdown("#### ✍️ DUAL_APPROVAL Simulation")
    dual = (report.get("authorization_simulation") or {}).get("dual_approval") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Real Outcome", str(dual.get("real_outcome", "—")))
    c2.metric("Simulated Outcome", str(dual.get("simulated_outcome", "—")))
    c3.metric("Bypass Possible", "NO" if not dual.get("single_approval_bypass_possible") else "YES")
    tests = pd.DataFrame(dual.get("bypass_tests") or [])
    if not tests.empty:
        st.dataframe(tests, use_container_width=True, hide_index=True)


def _render_market(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Market Readiness")
    market = report.get("market_readiness") or {}
    st.metric("Outcome", str(market.get("outcome", "—")))
    checks = market.get("checks") or {}
    for k, v in checks.items():
        icon = "✅" if v else "❌"
        st.markdown(f"{icon} {k.replace('_', ' ')}")


def _render_lifecycle(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Lifecycle Confirmation")
    lc = report.get("lifecycle_validation") or {}
    st.markdown(
        f"**{lc.get('symbol')}** — {lc.get('effective_stance')} · conf {lc.get('confidence')} · "
        f"**{lc.get('outcome')}**"
    )


def _render_gonogo(report: Dict[str, Any]) -> None:
    st.markdown("#### 🚦 GO / NO-GO Decision")
    go = (report.get("authorization_simulation") or {}).get("go_nogo") or {}
    st.markdown(f"**Real-world:** {go.get('real_world_decision')}")
    st.markdown(f"**Dress rehearsal:** {go.get('dress_rehearsal_decision')}")
    if go.get("no_go_examples"):
        st.markdown("**NO-GO examples:**")
        for ex in go.get("no_go_examples") or []:
            st.markdown(f"- {ex}")


def _render_failures(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Failure Scenario Matrix")
    df = pd.DataFrame(report.get("failure_scenarios") or [])
    if df.empty:
        st.info("No scenarios.", icon="ℹ️")
        return
    show = [
        c
        for c in [
            "scenario",
            "failure_classification",
            "simulated_result",
            "currently_applicable",
            "required_response",
        ]
        if c in df.columns
    ]
    st.dataframe(df[show] if show else df, use_container_width=True, hide_index=True)


def _render_audit(results_dir: Path) -> None:
    st.markdown("#### 📎 Audit Trail")
    df = _read_csv_safe(results_dir / "final_authorization_audit_trail.csv")
    if df.empty:
        st.info("No audit trail.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_tree(results_dir: Path) -> None:
    st.markdown("#### 🌳 Decision Tree")
    df = _read_csv_safe(results_dir / "final_authorization_decision_tree.csv")
    if df.empty:
        st.info("No decision tree.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_autonomy(report: Dict[str, Any]) -> None:
    st.markdown("#### 🤖 Autonomy Boundaries")
    auto = report.get("autonomy_validation") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Triton CAN**")
        for item in auto.get("triton_can") or []:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Triton CANNOT**")
        for item in auto.get("triton_cannot") or []:
            st.markdown(f"- {item}")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_recovery_transition_final_authorization_simulator(results_dir: Path) -> None:
    st.markdown("### 🎬 Final Recovery Authorization Simulation")
    st.caption(
        "End-to-end dress rehearsal of Stage 1 authorization. "
        "Refresh with `python -m services.recovery_transition_final_authorization_simulator`."
    )
    report = _read_json_safe(results_dir / "final_authorization_simulation.json")
    if not report:
        st.warning(
            "No simulation found. Run "
            "`python -m services.recovery_transition_final_authorization_simulator`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_timeline(report)
    _render_dual(report)
    _render_market(report)
    _render_lifecycle(report)
    _render_gonogo(report)
    _render_tree(results_dir)
    _render_failures(report)
    _render_audit(results_dir)
    _render_autonomy(report)
    _render_certification(report)
