"""
TRITON Phase 196 — Recovery Restart Diagnostics dashboard.
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
    st.markdown("#### 🔧 [RECOVERY_RESTART_DIAGNOSTICS_REPORT]")
    ex = report.get("executive_recommendation") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Restart Attempted", "YES" if ex.get("restart_attempted") else "NO")
    c2.metric("Failure Class", str(report.get("failure_classification", "—")))
    c3.metric("Autonomy Trustworthy", "YES" if ex.get("autonomous_recovery_trustworthy") else "NO")
    st.info(
        f"Failed stage: {ex.get('failed_stage') or '—'} · Cause: {ex.get('failure_cause') or '—'}",
        icon="💡",
    )
    gaps = ex.get("production_readiness_gaps") or []
    if gaps:
        with st.expander("Production readiness gaps"):
            for g in gaps:
                st.markdown(f"- {g}")


def _render_last_attempt(report: Dict[str, Any]) -> None:
    st.markdown("#### 🕐 Last Recovery Attempt")
    att = report.get("last_recovery_attempt") or {}
    st.json(att) if att else st.info("No recovery attempts recorded.", icon="ℹ️")


def _render_timeline(report: Dict[str, Any]) -> None:
    st.markdown("#### ⏱ Recovery Pipeline Timeline")
    chain = report.get("recovery_chain_trace") or []
    if not chain:
        st.info("No chain trace.", icon="ℹ️")
        return
    df = pd.DataFrame(chain)
    cols = [
        c for c in ["stage", "status", "error_message", "recommended_correction"] if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_command(report: Dict[str, Any]) -> None:
    st.markdown("#### 📝 Restart Command Analysis")
    forensics = report.get("forensic_evidence") or {}
    st.code(forensics.get("restart_command") or "—", language="powershell")
    tests = report.get("simulation_tests") or []
    cmd_test = next((t for t in tests if t.get("test") == "command_validation"), {})
    st.markdown(f"Validation: **{cmd_test.get('result', '—')}**")


def _render_env(report: Dict[str, Any]) -> None:
    st.markdown("#### 🌍 Environment Validation")
    env = report.get("environment_validation") or {}
    checks = [
        ("Venv available", env.get("venv_available")),
        ("Script exists", env.get("powershell_script_exists")),
        ("Execution policy OK", env.get("execution_policy_permits_launch")),
        ("Root exists", env.get("triton_root_exists")),
        ("Log dir exists", env.get("log_directory_exists")),
    ]
    for label, ok in checks:
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} {label}")


def _render_recommendations(results_dir: Path) -> None:
    st.markdown("#### 💡 Corrective Recommendations")
    df = _read_csv_safe(results_dir / "recovery_restart_recommendations.csv")
    if df.empty:
        st.info("No recommendations.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_history(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Historical Recovery Success Rate")
    stats = report.get("recovery_success_rate") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Attempts", str(stats.get("attempts", 0)))
    c2.metric("Successes", str(stats.get("successes", 0)))
    c3.metric("Success %", str(stats.get("success_pct", "—")))
    df = _read_csv_safe(results_dir / "recovery_restart_failure_history.csv")
    if not df.empty:
        st.dataframe(df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)


def render_recovery_restart_diagnostics(results_dir: Path) -> None:
    st.markdown("### 🔧 Recovery Restart Diagnostics")
    st.caption(
        "Root-cause analysis for failed autonomous loop restarts. "
        "Refresh with `python -m services.recovery_restart_diagnostics`."
    )
    report = _read_json_safe(results_dir / "recovery_restart_diagnostics.json")
    if not report:
        st.warning(
            "No diagnostics. Run `python -m services.recovery_restart_diagnostics`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_last_attempt(report)
    _render_timeline(report)
    _render_command(report)
    _render_env(report)
    _render_recommendations(results_dir)
    _render_history(results_dir, report)
