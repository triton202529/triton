"""
TRITON THP-004 — Execution Gate Governance Dashboard (read-only).

Displays execution gate decisions, evidence, authorization chains, and authority matrix.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"

DECISIONS_JSON = GOVERNANCE_DIR / "execution_gate_decisions.json"
DECISIONS_CSV = GOVERNANCE_DIR / "execution_gate_decisions.csv"
EVIDENCE_JSON = GOVERNANCE_DIR / "execution_gate_evidence.json"
EVIDENCE_CSV = GOVERNANCE_DIR / "execution_gate_evidence.csv"
AUTHORITY_MATRIX_JSON = GOVERNANCE_DIR / "execution_gate_authority_matrix.json"
AUDIT_CSV = ROOT / "data" / "results" / "execution_gate_audit.csv"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _decision_badge(decision: str) -> str:
    return {
        "ALLOW": "🟢 ALLOW",
        "BLOCK": "🔴 BLOCK",
        "REVIEW_REQUIRED": "🟡 REVIEW",
        "SANDBOX_ONLY": "🔵 SANDBOX",
        "DISABLED": "⚫ DISABLED",
    }.get(str(decision).upper(), f"⚪ {decision}")


def _allowed_badge(v: Any) -> str:
    if v is True or str(v).lower() in ("true", "1", "yes"):
        return "🟢 Approved"
    return "🔴 Blocked"


def page_execution_gate_governance() -> None:
    st.title("🛡 Execution Gate Governance")
    st.info(
        "**Institutional execution gate evidence.** Read-only view of THP-004 decision "
        "registry, authorization chains, and audit linkages. No trading or broker changes."
    )

    decisions_doc = _read_json(DECISIONS_JSON)
    evidence_doc = _read_json(EVIDENCE_JSON)
    matrix = _read_json(AUTHORITY_MATRIX_JSON)
    decisions = decisions_doc.get("records") or []
    evidence = evidence_doc.get("records") or []
    stats = matrix.get("statistics") or {}

    total = len(decisions)
    approved = sum(1 for d in decisions if d.get("allowed"))
    blocked = total - approved
    review_req = sum(
        1 for d in decisions if str(d.get("execution_gate_decision")).upper() == "REVIEW_REQUIRED"
    )
    emergency = sum(
        1 for d in decisions if d.get("authority_classification") == "risk_emergency_action"
    )

    st.markdown("### Execution Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Requests", total)
    c2.metric("Approved", approved)
    c3.metric("Blocked", blocked)
    c4.metric("Review Required", review_req)
    c5.metric("Emergency Actions", emergency)

    st.markdown("### Execution Paths")
    path_counts = Counter(str(d.get("execution_path") or "unknown") for d in decisions)
    path_df = pd.DataFrame(
        [{"path": k, "count": v} for k, v in sorted(path_counts.items(), key=lambda x: -x[1])]
    )
    if not path_df.empty:
        st.bar_chart(path_df.set_index("path"))
    else:
        st.caption("No execution path activity recorded yet.")

    st.markdown("### Decision Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Most common block reasons**")
        block_reasons = stats.get("top_block_reasons") or {}
        if block_reasons:
            br_df = pd.DataFrame([{"reason": k, "count": v} for k, v in block_reasons.items()])
            st.dataframe(br_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No block reasons recorded.")
    with col_b:
        st.markdown("**Most active services**")
        svc = stats.get("by_service") or {}
        if svc:
            svc_df = pd.DataFrame([{"service": k, "count": v} for k, v in svc.items()])
            st.dataframe(svc_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No service activity recorded.")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Environment breakdown**")
        env_counts = Counter(str(d.get("environment") or "unknown") for d in decisions)
        if env_counts:
            st.dataframe(
                pd.DataFrame([{"environment": k, "count": v} for k, v in env_counts.items()]),
                use_container_width=True,
                hide_index=True,
            )
    with col_d:
        st.markdown("**Paper vs live activity**")
        pl_counts = Counter(str(d.get("paper_or_live") or "unknown") for d in decisions)
        if pl_counts:
            st.dataframe(
                pd.DataFrame([{"mode": k, "count": v} for k, v in pl_counts.items()]),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### Risk & Governance")
    rg1, rg2, rg3, rg4 = st.columns(4)
    rg1.metric("Override Attempts", stats.get("by_authority", {}).get("operator_override", 0))
    rg2.metric("Live Execution Requests", path_counts.get("live_execution", 0))
    rg3.metric(
        "Unauthorized Attempts",
        stats.get("by_authority", {}).get("system_rejection", 0),
    )
    rg4.metric("Evidence Records", len(evidence))

    st.markdown("### Decision Registry")
    df = _read_csv(DECISIONS_CSV)
    if df.empty and decisions:
        df = pd.DataFrame(decisions)
    if not df.empty:
        show = df.copy()
        if "execution_gate_decision" in show.columns:
            show["decision_badge"] = show["execution_gate_decision"].apply(_decision_badge)
        if "allowed" in show.columns:
            show["disposition"] = show["allowed"].apply(_allowed_badge)
        st.dataframe(show, use_container_width=True, hide_index=True)

        labels = [
            f"{r.get('decision_id', '')} — {r.get('execution_path', '')} — {r.get('symbol(s)', '')}"
            for r in decisions[:200]
        ]
        if labels:
            pick = st.selectbox("Inspect authorization chain", labels, key="eg_gov_chain_pick")
            idx = labels.index(pick)
            selected = decisions[idx]
            st.markdown(
                f"**Decision:** {_decision_badge(selected.get('execution_gate_decision', ''))}"
            )
            st.markdown(f"**Reason:** {selected.get('decision_reason', '')}")
            st.markdown(f"**Authority:** `{selected.get('authority_classification', '')}`")
            chain = selected.get("authorization_chain") or []
            if chain:
                st.markdown("**Authorization chain**")
                st.dataframe(pd.DataFrame(chain), use_container_width=True, hide_index=True)
            links = selected.get("evidence_links") or []
            if links:
                st.markdown("**Evidence links**")
                for link in links:
                    st.code(str(link))
    else:
        st.info("No execution gate decisions recorded yet.", icon="ℹ️")

    st.markdown("### Evidence Registry")
    ev_df = _read_csv(EVIDENCE_CSV)
    if not ev_df.empty:
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No evidence records yet.")

    st.markdown("### Authority Matrix")
    if matrix:
        st.json(matrix)
    else:
        st.caption("Authority matrix not generated yet.")

    st.markdown("### Operational Audit Link")
    audit_df = _read_csv(AUDIT_CSV)
    if not audit_df.empty:
        st.caption(f"Linked operational audit: `{AUDIT_CSV.as_posix()}` ({len(audit_df)} rows)")
        st.dataframe(audit_df.tail(25), use_container_width=True, hide_index=True)
    else:
        st.caption("No operational audit rows yet (`data/results/execution_gate_audit.csv`).")


def render() -> None:
    page_execution_gate_governance()
