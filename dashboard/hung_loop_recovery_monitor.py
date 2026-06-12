"""
TRITON Phase 195 — Hung Loop Recovery Intelligence dashboard.
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
    st.markdown("#### 🧠 [HUNG_LOOP_RECOVERY_REPORT]")
    ex = report.get("executive_recommendation") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Truly Alive", "YES" if ex.get("loop_truly_alive") else "NO")
    c2.metric("Hung", "YES" if ex.get("loop_hung") else "NO")
    c3.metric("Recovery Safe", "YES" if ex.get("recovery_safe") else "NO")
    c4.metric("Classification", str(report.get("process_health_classification", "—")))
    st.info(str(ex.get("recommended_operator_action") or ""), icon="💡")
    if ex.get("why_self_healing_refused"):
        st.warning(f"Self-healing refused: {ex['why_self_healing_refused']}", icon="⚠️")


def _render_process(report: Dict[str, Any]) -> None:
    st.markdown("#### 🖥 Current Process Health")
    intel = report.get("intelligence") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Process Running", "YES" if intel.get("process_existence") else "NO")
    c2.metric("Missed Cycles", str(intel.get("consecutive_missed_cycles", "—")))
    c3.metric("Cycle Age (min)", str(intel.get("last_cycle_age_minutes", "—")))
    pids = intel.get("process_pids") or []
    st.markdown(f"**Active PIDs:** {pids or 'None'}")
    details = intel.get("process_details") or []
    if details:
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)


def _render_timelines(report: Dict[str, Any]) -> None:
    st.markdown("#### ⏱ Heartbeat & Cycle Timeline")
    intel = report.get("intelligence") or {}
    hb = intel.get("heartbeat") or {}
    st.markdown(
        f"Loop refresh: {hb.get('continuous_loop_refresh_ts')} ({hb.get('continuous_loop_refresh_age_minutes')} min)"
    )
    st.markdown(
        f"Watchdog HB: {hb.get('watchdog_heartbeat_ts')} ({hb.get('watchdog_heartbeat_age_minutes')} min)"
    )
    st.markdown(f"Last cycle OK: {intel.get('last_completed_cycle')}")
    log_act = intel.get("log_activity") or {}
    st.markdown(
        f"Log mtime age: {log_act.get('log_mtime_age_minutes')} min · stale={log_act.get('log_stale')}"
    )


def _render_forensics(results_dir: Path) -> None:
    st.markdown("#### 🔬 Forensic Evidence")
    f = _read_json_safe(results_dir / "hung_loop_forensics.json")
    if not f:
        st.info("No forensics.", icon="ℹ️")
        return
    st.json(f)


def _render_recovery(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔧 Recovery Recommendation")
    dec = report.get("recovery_decision") or {}
    st.markdown(f"**Verdict:** {dec.get('verdict')}")
    st.markdown(f"**Reason:** {dec.get('reason')}")
    rec = report.get("recovery_execution") or {}
    if rec.get("result") and rec.get("result") != "NOT_ATTEMPTED":
        st.markdown(f"**Recovery result:** {rec.get('result')}")


def _render_history(results_dir: Path) -> None:
    st.markdown("#### 📈 Recovery History")
    df = _read_csv_safe(results_dir / "hung_loop_history.csv")
    if not df.empty:
        st.line_chart(df.tail(50).reset_index(drop=True).assign(idx=range(len(df.tail(50)))))
    actions = _read_csv_safe(results_dir / "hung_loop_recovery_actions.csv")
    if not actions.empty:
        st.dataframe(actions.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
    elif df.empty:
        st.info("No history yet.", icon="ℹ️")


def _render_risk(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Operational Risk")
    intel = report.get("intelligence") or {}
    risk = (
        "CRITICAL"
        if report.get("process_health_classification") == "HUNG_PROCESS"
        else ("ELEVATED" if report.get("process_health_classification") == "DEGRADED" else "NORMAL")
    )
    st.metric("Risk Level", risk)
    st.caption(str(report.get("disclaimer") or ""))


def render_hung_loop_recovery_monitor(results_dir: Path) -> None:
    st.markdown("### 🧠 Hung Loop Recovery Intelligence")
    st.caption(
        "Detects frozen continuous loop PIDs and recommends safe recovery. "
        "Refresh with `python -m services.hung_loop_detector`."
    )
    report = _read_json_safe(results_dir / "hung_loop_detection.json")
    if not report:
        st.warning("No detection data. Run `python -m services.hung_loop_detector`.", icon="⚠️")
        return
    _render_executive(report)
    _render_process(report)
    _render_timelines(report)
    _render_forensics(results_dir)
    _render_recovery(report)
    _render_history(results_dir)
    _render_risk(report)
