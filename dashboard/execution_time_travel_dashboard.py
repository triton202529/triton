"""
TRITON THP-009 — Execution Time Travel & Regulatory Audit Dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from services.execution_decision_reconstructor import (
    POLICY_DRIFT_JSON,
    RECONSTRUCTION_JSON,
    load_reconstruction_reports,
    reconstruct_decision,
    run_reconstruction_batch,
)
from services.execution_regulatory_audit_pack import (
    AUDIT_PACKAGES_JSON,
    generate_audit_package,
    get_audit_package,
    load_audit_packages,
    run_audit_package_generation,
    save_audit_package,
)
from services.execution_time_machine import (
    SNAPSHOTS_JSON,
    load_snapshots,
    run_time_machine,
    verify_snapshot_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
SNAPSHOT_INDEX_CSV = GOVERNANCE_DIR / "execution_snapshot_index.csv"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _classification_badge(label: str) -> str:
    return {
        "SAME_DECISION": "🟢 SAME",
        "POLICY_CHANGED": "🟡 POLICY",
        "RISK_CHANGED": "🟠 RISK",
        "GOVERNANCE_CHANGED": "🔵 GOVERNANCE",
        "ENVIRONMENT_CHANGED": "🟣 ENVIRONMENT",
    }.get(label, label)


def page_execution_time_travel() -> None:
    st.title("⏳ Execution Time Travel & Regulatory Audit")
    st.info(
        "**Forensic-grade institutional memory.** Reconstruct historical execution decisions, "
        "compare against present-day rules, and export regulatory audit packages. Read-only."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Capture Snapshots", key="tt_snapshots"):
            with st.spinner("Running time machine..."):
                run_time_machine()
            st.success("Snapshots captured.")
    with col2:
        if st.button("Run Reconstruction", key="tt_reconstruct"):
            with st.spinner("Reconstructing decisions..."):
                run_reconstruction_batch()
            st.success("Reconstruction complete.")
    with col3:
        if st.button("Generate Audit Packages", key="tt_audit"):
            with st.spinner("Building audit packages..."):
                run_audit_package_generation()
            st.success("Audit packages generated.")

    snapshots = load_snapshots() or _read_json(SNAPSHOTS_JSON)
    recon = load_reconstruction_reports() or _read_json(RECONSTRUCTION_JSON)
    packages = load_audit_packages() or _read_json(AUDIT_PACKAGES_JSON)
    drift = _read_json(POLICY_DRIFT_JSON)

    st.markdown("### Snapshot Inventory")
    snap_summary = snapshots.get("summary") or {}
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Snapshots", snap_summary.get("total", len(snapshots.get("records") or [])))
    s2.metric("Reconstructions", (recon.get("summary") or {}).get("total", 0))
    s3.metric("Audit Packages", (packages.get("summary") or {}).get("total", 0))

    st.markdown("### Execution Search")
    index_df = pd.DataFrame()
    if SNAPSHOT_INDEX_CSV.is_file():
        try:
            index_df = pd.read_csv(SNAPSHOT_INDEX_CSV)
        except Exception:
            pass

    records = snapshots.get("records") or []
    if index_df.empty and records:
        index_df = pd.DataFrame(
            [
                {
                    "decision_id": (r.get("execution_identity") or {}).get("decision_id"),
                    "execution_path": (r.get("execution_identity") or {}).get("execution_path"),
                    "timestamp": r.get("decision_timestamp"),
                    "fingerprint": r.get("fingerprint"),
                }
                for r in records
            ]
        )

    search_symbol = st.text_input("Filter by symbol (optional)", key="tt_symbol")
    search_path = st.text_input("Filter by execution path (optional)", key="tt_path")

    if not index_df.empty:
        filtered = index_df.copy()
        if search_symbol and "symbol(s)" in filtered.columns:
            filtered = filtered[
                filtered["symbol(s)"].astype(str).str.contains(search_symbol, case=False, na=False)
            ]
        if search_path and "execution_path" in filtered.columns:
            filtered = filtered[
                filtered["execution_path"]
                .astype(str)
                .str.contains(search_path, case=False, na=False)
            ]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        decision_ids: List[str] = [
            str(x) for x in filtered.get("decision_id", pd.Series()).dropna().tolist()
        ]
    else:
        decision_ids = [
            str((r.get("frozen_decision") or {}).get("decision_id"))
            for r in records
            if (r.get("frozen_decision") or {}).get("decision_id")
        ]
        st.caption("No snapshots yet — click **Capture Snapshots**.")

    st.markdown("### Historical Replay")
    if decision_ids:
        selected_id = st.selectbox("Decision ID", decision_ids, key="tt_decision_select")
        if st.button("Reconstruct Selected", key="tt_reconstruct_one"):
            report = reconstruct_decision(selected_id)
            st.markdown(_classification_badge(str(report.get("comparison_classification"))))
            st.write(report.get("narrative_summary"))
            st.json(report)

        snap = next(
            (
                r
                for r in records
                if (r.get("execution_identity") or {}).get("decision_id") == selected_id
                or (r.get("frozen_decision") or {}).get("decision_id") == selected_id
            ),
            None,
        )
        if snap:
            integrity = verify_snapshot_integrity(snap)
            st.markdown("#### Snapshot Integrity")
            st.json(integrity)
            with st.expander("Frozen Decision & Evidence"):
                st.json(
                    {
                        "frozen_decision": snap.get("frozen_decision"),
                        "frozen_evidence": snap.get("frozen_evidence"),
                        "authorization_chain": (snap.get("governance_state") or {}).get(
                            "authorization_chain"
                        ),
                    }
                )

    st.markdown("### Present-Day Comparison & Policy Drift")
    drift_summary = drift.get("summary") or {}
    d1, d2 = st.columns(2)
    d1.metric("Policy Drift Events", drift_summary.get("drift_count", 0))
    d2.metric("Same Decision Count", drift_summary.get("same_decision_count", 0))
    if drift.get("records"):
        st.dataframe(pd.DataFrame(drift["records"]), use_container_width=True, hide_index=True)

    if recon.get("records"):
        st.markdown("#### Reconstruction Reports")
        st.dataframe(
            pd.DataFrame(recon["records"])[
                ["decision_id", "comparison_classification", "integrity_valid", "narrative_summary"]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Audit Export")
    if decision_ids:
        export_id = st.selectbox("Export decision ID", decision_ids, key="tt_export_select")
        if st.button("Generate Regulatory Audit Package", key="tt_export"):
            pkg = generate_audit_package(export_id)
            save_audit_package(pkg)
            st.success(f"Package {pkg.get('package_id')} created.")
            st.markdown("#### Executive Summary")
            st.json(pkg.get("executive_summary"))
            st.markdown("#### Decision Timeline")
            for step in pkg.get("decision_timeline") or []:
                st.write(f"**{step.get('step')}** — {step.get('status')}: {step.get('detail')}")
            st.markdown("#### Certification")
            st.json(pkg.get("certification"))

    if packages.get("records"):
        st.markdown("#### Saved Audit Packages")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "package_id": p.get("package_id"),
                        "decision_id": p.get("decision_id"),
                        "completeness": (p.get("certification") or {}).get("completeness_score"),
                        "fingerprint": (p.get("certification") or {}).get(
                            "integrity_fingerprint", ""
                        )[:16],
                    }
                    for p in packages["records"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render() -> None:
    page_execution_time_travel()
