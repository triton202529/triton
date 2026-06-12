"""
TRITON Phase 194 — Continuous Loop Health Monitor dashboard.
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
    st.markdown("#### 🔄 [CONTINUOUS_LOOP_RESILIENCE_REPORT]")
    ex = report.get("executive_recommendation") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", str(ex.get("current_operational_status", "—")))
    c2.metric("Risk Level", str(ex.get("operational_risk_level", "—")))
    c3.metric("Recoveries OK", str(ex.get("total_recoveries_completed", 0)))
    c4.metric("Success %", str(ex.get("recovery_success_percentage", "—")))
    st.info(str(ex.get("recommended_operator_action") or ""), icon="💡")


def _render_status(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current Loop Status")
    cur = report.get("current_status") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health", str(cur.get("health_classification", "—")))
    c2.metric("Process Running", "YES" if cur.get("process_running") else "NO")
    c3.metric("Lock Present", "YES" if cur.get("lock_exists") else "NO")
    c4.metric("Total Cycles", str(cur.get("uptime_cycles", "—")))
    st.markdown(f"**Last cycle OK:** {cur.get('last_completed_cycle') or '—'}")
    st.markdown(f"**Next expected:** {cur.get('next_expected_cycle') or '—'}")


def _render_heartbeat(report: Dict[str, Any]) -> None:
    st.markdown("#### 💓 Last Heartbeat")
    hb = (report.get("health_metrics") or {}).get("heartbeat") or {}
    st.markdown(
        f"Watchdog: {hb.get('watchdog_heartbeat_ts')} ({hb.get('watchdog_heartbeat_age_minutes')} min ago)"
    )
    st.markdown(
        f"Loop refresh: {hb.get('continuous_loop_refresh_ts')} "
        f"({hb.get('continuous_loop_refresh_age_minutes')} min ago)"
    )
    if hb.get("stale_heartbeat_alerts_active"):
        st.warning(
            f"Stale heartbeat alerts active: {hb['stale_heartbeat_alerts_active']}", icon="⚠️"
        )


def _render_failure(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔍 Failure Classification")
    m = report.get("health_metrics") or {}
    st.markdown(f"**Classification:** {m.get('failure_classification') or 'None'}")
    st.markdown(f"**Root cause:** {m.get('root_cause_hypothesis') or '—'}")
    st.markdown(f"**Missed cycles:** {m.get('consecutive_missed_cycles', 0)}")
    dup = report.get("duplicate_loop_protection") or {}
    st.markdown(f"**Recovery verdict:** {dup.get('verdict')} — {dup.get('reason')}")


def _render_recovery(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔧 Recovery Timeline")
    rec = report.get("recovery") or {}
    if not rec:
        st.info("No recovery action this run.", icon="ℹ️")
        return
    st.markdown(f"**Result:** {rec.get('result')}")
    for step in rec.get("steps") or []:
        st.markdown(f"- {step.get('step')}: {step.get('action')} — {step.get('detail')}")


def _render_failures_csv(results_dir: Path) -> None:
    st.markdown("#### ⚠️ Recent Failure Events")
    df = _read_csv_safe(results_dir / "continuous_loop_failure_events.csv")
    if df.empty:
        st.info("No failure events recorded.", icon="ℹ️")
        return
    st.dataframe(df.tail(15).iloc[::-1], use_container_width=True, hide_index=True)


def _render_history(results_dir: Path) -> None:
    st.markdown("#### 📈 Recovery History")
    df = _read_csv_safe(results_dir / "continuous_loop_recovery_history.csv")
    if df.empty:
        st.info("No history yet.", icon="ℹ️")
        return
    if "readiness_score" in df.columns:
        st.line_chart(df.set_index("timestamp")["readiness_score"].tail(100))
    st.dataframe(df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    st.caption(str(report.get("disclaimer") or ""))
    st.code(
        "python -m services.continuous_loop_recovery_manager\n"
        "python -m services.continuous_loop_recovery_manager --recover\n"
        "python -m services.continuous_loop_recovery_manager --recover --dry-run",
        language="powershell",
    )


def render_continuous_loop_health_monitor(results_dir: Path) -> None:
    st.markdown("### 🔄 Continuous Loop Health Monitor")
    st.caption(
        "45-minute loop health, failure classification, and optional self-healing. "
        "Refresh with `python -m services.continuous_loop_recovery_manager`."
    )
    report = _read_json_safe(results_dir / "continuous_loop_health.json")
    if not report:
        st.warning(
            "No health data. Run `python -m services.continuous_loop_recovery_manager`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_status(report)
    _render_heartbeat(report)
    _render_failure(report)
    _render_recovery(report)
    _render_failures_csv(results_dir)
    _render_history(results_dir)
    _render_certification(report)
