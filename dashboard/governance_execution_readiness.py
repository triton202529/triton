"""
TRITON Governance Authorization & Execution Readiness dashboards — Phases 16–18.
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


def _gate_badge(ok: bool) -> str:
    return "✅ PASS" if ok else "🔒 BLOCKED"


def render_governance_authorization_center(results_dir: Path) -> None:
    st.markdown("### 🏛 Governance Authorization Center")
    st.caption(
        "Four-layer authorization gate. Paper execution permitted only when policy gate "
        "and execution gate pass — live execution remains blocked."
    )

    doc = _load_json(results_dir / "governance_authorization.json")
    if not doc:
        st.warning("Authorization data not found. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    overall = doc.get("overall_authorization", False)
    st.metric("Overall Authorization", "AUTHORIZED" if overall else "NOT AUTHORIZED")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Gate", _gate_badge(doc.get("governance_authorized")))
    c2.metric("Operator Gate", _gate_badge(doc.get("operator_authorized")))
    c3.metric("Policy Gate", _gate_badge(doc.get("policy_authorized")))
    c4.metric("Execution Gate", _gate_badge(doc.get("execution_authorized")))

    if doc.get("paper_execution_permitted"):
        st.success("Paper execution authorized by policy gate (does not place orders).")
    if doc.get("live_execution_permitted"):
        st.error("Live execution must remain blocked.")

    questions = doc.get("authorization_questions") or {}
    st.markdown("#### Authorization Questions")
    q_rows = [
        {
            "Question": "Is governance permitting this?",
            "Answer": questions.get("is_governance_permitting"),
        },
        {
            "Question": "Is operator approval present?",
            "Answer": questions.get("is_operator_approval_present"),
        },
        {"Question": "Is policy enabled?", "Answer": questions.get("is_policy_enabled")},
        {"Question": "Is execution allowed?", "Answer": questions.get("is_execution_allowed")},
    ]
    st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)

    reasons = doc.get("gate_reasons") or {}
    if reasons:
        with st.expander("Gate details", expanded=False):
            st.json(reasons)

    auths: List[Dict[str, Any]] = doc.get("candidate_authorizations") or []
    if auths:
        st.markdown("#### Per-Candidate Authorization")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Action": a.get("candidate_action"),
                        "Governance": a.get("governance_authorized"),
                        "Operator": a.get("operator_authorized"),
                        "Policy": a.get("policy_authorized"),
                        "Execution": a.get("execution_authorized"),
                        "Overall": a.get("overall_authorization"),
                    }
                    for a in auths
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if not overall:
        st.info(
            "All authorization gates must pass before any future protective action is eligible."
        )


def render_execution_readiness_center(results_dir: Path) -> None:
    st.markdown("### ⚙️ Execution Readiness Center")
    st.caption(
        "Eligibility assessment for future protective actions. "
        "**Paper mode only — no live execution.**"
    )

    doc = _load_json(results_dir / "execution_readiness.json")
    if not doc:
        st.warning("Readiness data not found. Run the watchdog first.")
        return

    status = str(doc.get("readiness_status") or "NOT_READY")
    status_color = {"NOT_READY": "🔴", "PARTIALLY_READY": "🟡", "READY": "🟢"}.get(status, "⚪")
    st.metric("Readiness State", f"{status_color} {status}")
    st.caption(
        f"Mode: **{doc.get('mode', '—')}** | "
        f"Paper execution permitted: **{doc.get('paper_execution_permitted', False)}** | "
        f"Live execution permitted: **{doc.get('live_execution_permitted', False)}**"
    )

    c1, c2 = st.columns(2)
    c1.metric("Passing Checks", doc.get("checks_passing_count", 0))
    c2.metric("Total Checks", doc.get("checks_total", 0))

    passing = doc.get("passing_checks") or []
    failing = doc.get("failed_checks") or []
    col_p, col_f = st.columns(2)
    with col_p:
        st.markdown("#### Passing Checks")
        if passing:
            for name in passing:
                detail = (doc.get("check_details") or {}).get(name, "")
                st.success(f"**{name}** — {detail}")
        else:
            st.caption("None")
    with col_f:
        st.markdown("#### Failed Checks")
        if failing:
            for name in failing:
                detail = (doc.get("check_details") or {}).get(name, "")
                st.warning(f"**{name}** — {detail}")
        else:
            st.caption("None")

    checks = doc.get("checks") or {}
    if checks:
        st.dataframe(
            pd.DataFrame([{"Check": k, "Passing": v} for k, v in checks.items()]),
            use_container_width=True,
            hide_index=True,
        )

    history_path = results_dir / "execution_readiness_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "readiness_status" in hist.columns:
                st.markdown("#### Readiness Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(
                    trend.set_index("timestamp")["checks_passing"]
                    if "checks_passing" in trend.columns
                    else trend["readiness_status"].astype("category").cat.codes
                )
        except Exception:
            pass


def render_protective_action_trials(results_dir: Path) -> None:
    st.markdown("### 🧪 Protective Action Trials")
    st.caption(
        "Paper-mode protective action test environment. "
        "All trials are **SIMULATION_ONLY** — no live orders or portfolio changes."
    )

    doc = _load_json(results_dir / "protective_action_trials.json")
    if not doc:
        st.warning("Trial data not found. Run the watchdog first.")
        return

    st.caption(
        f"Generated: **{doc.get('generated_at', '—')}** | Baseline CPS: **{doc.get('baseline_cps')}**"
    )

    trials: List[Dict[str, Any]] = doc.get("trials") or []
    if not trials:
        st.info("No trials generated.")
        return

    summary_rows = []
    for t in trials:
        summary_rows.append(
            {
                "Trial": t.get("trial_name"),
                "CPS Δ": t.get("estimated_cps_improvement"),
                "Risk Reduction": t.get("estimated_risk_reduction"),
                "Concentration Δ": t.get("estimated_concentration_reduction"),
                "Drawdown Δ": t.get("estimated_drawdown_improvement"),
                "Status": t.get("status"),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    for trial in trials:
        with st.expander(trial.get("trial_name", "Trial"), expanded=False):
            st.markdown(f"**Status:** `{trial.get('status')}` | **Mode:** `{trial.get('mode')}`")
            m1, m2, m3 = st.columns(3)
            m1.metric("Est. CPS Improvement", trial.get("estimated_cps_improvement"))
            m2.metric("Est. Risk Reduction", trial.get("estimated_risk_reduction"))
            m3.metric("Est. Drawdown Improvement", trial.get("estimated_drawdown_improvement"))

            st.markdown("**Expected Benefits**")
            for b in trial.get("expected_benefits") or []:
                st.markdown(f"- {b}")
            st.markdown("**Expected Risks**")
            for r in trial.get("expected_risks") or []:
                st.markdown(f"- {r}")

    st.warning("Trials are counterfactual paper simulations only. No broker actions are performed.")
