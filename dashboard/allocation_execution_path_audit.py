"""
TRITON Phase 179 — Allocation-to-Execution Path Audit dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

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


_STATUS_BADGE = {
    "EXECUTABLE_NOW": "🟢 EXECUTABLE_NOW",
    "BLOCKED_BY_GOVERNANCE": "🔴 BLOCKED_BY_GOVERNANCE",
    "BLOCKED_BY_LIFECYCLE": "🟠 BLOCKED_BY_LIFECYCLE",
    "BLOCKED_BY_POSITION_MANAGER": "🟠 BLOCKED_BY_POSITION_MANAGER",
    "BLOCKED_BY_CAPITAL_ALLOCATION": "🟡 BLOCKED_BY_CAPITAL_ALLOCATION",
    "BLOCKED_BY_EXECUTION_LAYER": "🔴 BLOCKED_BY_EXECUTION_LAYER",
    "BLOCKED_BY_BROKER_RULES": "🔴 BLOCKED_BY_BROKER_RULES",
    "BLOCKED_BY_RISK_OVERLAY": "🟠 BLOCKED_BY_RISK_OVERLAY",
    "NO_ACTION_REQUIRED": "⚪ NO_ACTION_REQUIRED",
}


def _pass_badge(v: Any) -> str:
    if v is True:
        return "✅"
    if v is False:
        return "❌"
    return "—"


def _render_summary(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Required Summary")
    s = report.get("summary") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates", str(s.get("TOTAL_CANDIDATES", "—")))
    c2.metric("Executable Now", str(s.get("EXECUTABLE_NOW", "—")))
    c3.metric("Total Blocked", str(s.get("TOTAL_BLOCKED", "—")))
    c4.metric("Available Slots", str(s.get("available_slots", "—")))
    c5, c6, c7 = st.columns(3)
    c5.metric("Top Blocking Layer", str(s.get("TOP_BLOCKING_LAYER", "—")))
    c6.metric("Orders Planned (observed)", str(s.get("observed_orders_planned", "—")))
    c7.metric("Orders Executed (observed)", str(s.get("observed_orders_executed", "—")))
    st.markdown(f"**Top blocking reason:** {s.get('TOP_BLOCKING_REASON') or '—'}")
    st.markdown(
        f"**First symbol if blocker removed:** "
        f"{s.get('FIRST_SYMBOL_THAT_COULD_EXECUTE_IF_BLOCKER_REMOVED') or '—'}"
    )
    if s.get("paradox_detected"):
        st.warning(
            "Paradox detected: available_slots > 0 and open_new_candidates > 0 but orders_planned=0.",
            icon="⚠️",
        )


def _render_funnel(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔽 Execution Funnel")
    funnel = (report.get("summary") or {}).get("execution_funnel") or {}
    if not funnel:
        st.info("No funnel data.", icon="ℹ️")
        return
    stages = [
        ("Allocation", funnel.get("allocation", 0)),
        ("Governance", funnel.get("governance", 0)),
        ("Execution Auth", funnel.get("execution", 0)),
        ("Lifecycle", funnel.get("lifecycle", 0)),
        ("Risk", funnel.get("risk", 0)),
        ("Capital", funnel.get("capital", 0)),
        ("Placement", funnel.get("placement", 0)),
        ("Executable", funnel.get("executable", 0)),
    ]
    df = pd.DataFrame(stages, columns=["stage", "count"])
    st.markdown(
        " → ".join(f"**{name}** ({cnt})" for name, cnt in stages),
    )
    if not _plotly_bar(df, "stage", "count", "Candidates Surviving Each Layer"):
        st.bar_chart(df.set_index("stage")["count"])


def _render_blockers(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🚧 Blocker Summary")
    df = _read_csv_safe(results_dir / "allocation_execution_blockers.csv")
    if df.empty:
        df = pd.DataFrame(report.get("blocker_analysis") or [])
    if df.empty:
        st.info("No blockers recorded.", icon="ℹ️")
        return
    view = df.copy()
    if "blocker_category" in view.columns:
        view["blocker_category"] = view["blocker_category"].map(
            lambda x: _STATUS_BADGE.get(str(x), str(x))
        )
    st.dataframe(view, use_container_width=True, hide_index=True)
    if "count" in df.columns and "blocker_category" in df.columns:
        chart_df = df[["blocker_category", "count"]].copy()
        if not _plotly_bar(chart_df, "blocker_category", "count", "Blockers by Category"):
            st.bar_chart(chart_df.set_index("blocker_category")["count"])


def _render_executable(report: Dict[str, Any]) -> None:
    st.markdown("#### ✅ Executable Candidates")
    rows = report.get("candidate_traces") or []
    exe = [r for r in rows if r.get("final_status") == "EXECUTABLE_NOW"]
    if not exe:
        st.info("No candidates are fully executable through all layers.", icon="ℹ️")
        blocked = [r for r in rows if r.get("recommended_action") in ("OPEN_NEW", "ADD")]
        if blocked:
            st.markdown("**Nearest actionable candidates (blocked):**")
            near = sorted(
                blocked,
                key=lambda x: sum(
                    1
                    for k in (
                        "allocation_pass",
                        "governance_pass",
                        "execution_pass",
                        "lifecycle_pass",
                        "risk_pass",
                        "capital_pass",
                        "placement_pass",
                    )
                    if x.get(k)
                ),
                reverse=True,
            )
            st.dataframe(pd.DataFrame(near[:5]), use_container_width=True, hide_index=True)
        return
    st.dataframe(pd.DataFrame(exe), use_container_width=True, hide_index=True)


def _render_trace(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Decision Trace Table")
    df = _read_csv_safe(results_dir / "allocation_execution_decision_trace.csv")
    if df.empty:
        df = pd.DataFrame(report.get("candidate_traces") or [])
    if df.empty:
        st.info("No trace data.", icon="ℹ️")
        return
    view = df.copy()
    for col in (
        "allocation_pass",
        "governance_pass",
        "execution_pass",
        "lifecycle_pass",
        "risk_pass",
        "capital_pass",
        "placement_pass",
    ):
        if col in view.columns:
            view[col] = view[col].map(_pass_badge)
    if "final_status" in view.columns:
        view["final_status"] = view["final_status"].map(lambda x: _STATUS_BADGE.get(str(x), str(x)))
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_candidate_detail(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔍 Per-Candidate Detail")
    rows: List[Dict[str, Any]] = report.get("candidate_traces") or []
    if not rows:
        return
    symbols = [str(r.get("symbol")) for r in rows]
    sym = st.selectbox("Symbol", symbols, key="aepa_sym")
    row = next((r for r in rows if str(r.get("symbol")) == sym), {})
    if not row:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Allocation Score", _num(row.get("allocation_score")))
    c2.metric("Edge Score", _num(row.get("edge_score")))
    c3.metric("Confidence", _num(row.get("confidence")))
    st.markdown(
        f"**Action:** {row.get('recommended_action')} · **Lifecycle:** {row.get('lifecycle_action')} · "
        f"**Stance:** {row.get('effective_stance')}"
    )
    if row.get("lifecycle_mismatch"):
        st.warning(
            f"Lifecycle mismatch: allocation stance={row.get('allocation_effective_stance')} "
            f"vs effective={row.get('effective_stance')}",
            icon="⚠️",
        )
    st.markdown(f"**Blocking reason:** {row.get('blocking_reason') or '—'}")
    st.markdown(
        f"**Final decision:** {_STATUS_BADGE.get(str(row.get('final_decision')), row.get('final_decision'))}"
    )
    detail_cols = [
        "governance_authorized",
        "execution_authorized",
        "performance_risk_flags",
        "position_cap_status",
        "cash_available",
        "capital_allocation_result",
        "placement_eligible",
        "dry_run",
    ]
    st.json({k: row.get(k) for k in detail_cols})


def _render_root_cause(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Root Cause Analysis")
    rca = report.get("root_cause_analysis") or {}
    ctx = report.get("execution_context") or {}
    st.markdown(f"**Hypothesis:** {rca.get('hypothesis') or '—'}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Auth", str(ctx.get("governance_authorized")))
    c2.metric("Execution Auth", str(ctx.get("execution_authorized")))
    c3.metric("Escalation", str(ctx.get("escalation_state", "—")))
    c4.metric("Dry Run", str(ctx.get("dry_run")))
    st.markdown(
        f"Readiness: **{ctx.get('readiness_status')}** · "
        f"Final submit allowed: **{ctx.get('final_submit_allowed')}**"
    )


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or "read-only audit"), icon="🔒")
    st.caption(str(report.get("disclaimer") or ""))


def render_allocation_execution_path_audit(results_dir: Path) -> None:
    st.markdown("### 🔬 Allocation Execution Path Audit")
    st.caption(
        "Traces allocation candidates through governance → lifecycle → capital → placement. "
        "Refresh with `python -m services.allocation_execution_path_audit`."
    )
    report = _read_json_safe(results_dir / "allocation_execution_path_audit.json")
    if not report:
        st.warning(
            "No audit report found. Run `python -m services.allocation_execution_path_audit`.",
            icon="⚠️",
        )
        return
    _render_summary(report)
    _render_funnel(report)
    _render_blockers(results_dir, report)
    _render_executable(report)
    _render_trace(results_dir, report)
    _render_candidate_detail(report)
    _render_root_cause(report)
    _render_certification(report)
