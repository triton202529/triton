"""
TRITON Phase 197 — Recovery Stabilization Intelligence dashboard.
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


def _state_color(state: str) -> str:
    colors = {
        "HEALTHY": "🟢",
        "WARMING_UP": "🟡",
        "STARTING": "🔵",
        "DEGRADED": "🟠",
        "FAILED": "🔴",
    }
    return colors.get(state, "⚪")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 🟢 [RECOVERY_STABILIZATION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Truly Recovered", "YES" if ex.get("recovery_truly_succeeded") else "NO")
    c2.metric("Still Warming Up", "YES" if ex.get("still_warming_up") else "NO")
    c3.metric("Loop Stable", "YES" if ex.get("loop_stable") else "NO")
    c4.metric(
        "Autonomy Certifiable",
        "YES" if ex.get("executive_questions", {}).get("autonomous_recovery_certifiable") else "NO",
    )
    eq = ex.get("executive_questions") or {}
    st.info(
        f"Running vs operational: **{eq.get('running_vs_operational', '—')}** · "
        f"Healthy startup window: **{eq.get('healthy_startup_duration_seconds', '—')}s**",
        icon="💡",
    )


def _render_current_state(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current Recovery State")
    state = report.get("current_recovery_state", "—")
    st.markdown(f"{_state_color(str(state))} **{state}**")
    scores = report.get("confidence_scores") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Process Health", str(scores.get("process_health_score", "—")))
    c2.metric("Startup Confidence", str(scores.get("startup_confidence", "—")))
    c3.metric("Recovery Confidence", str(scores.get("recovery_confidence", "—")))
    c4.metric("Readiness", str(scores.get("readiness_classification", "—")))


def _render_timeline(report: Dict[str, Any]) -> None:
    st.markdown("#### ⏱ Startup Timeline")
    timeline = report.get("startup_timeline") or {}
    if not timeline:
        st.info("No live observation timeline. Run with `--observe` for full trace.", icon="ℹ️")
        return
    for key, label in [
        ("restart", "Restart"),
        ("first_lock", "Lock created"),
        ("first_heartbeat", "First heartbeat"),
        ("first_cycle_ok", "First cycle OK"),
        ("stabilization_complete", "Stabilization complete"),
    ]:
        st.markdown(f"- **{label}:** {timeline.get(key) or '—'}")


def _render_warmup(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔥 Warm-up Progress")
    wu = report.get("warm_up_progress") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("State", str(wu.get("final_state", "—")))
    c2.metric("Heartbeat progressions", str(wu.get("heartbeat_progressions", "—")))
    c3.metric("Observation (s)", str(wu.get("observation_seconds", "—")))
    st.markdown(f"Warm-up complete: **{'YES' if wu.get('warm_up_complete') else 'NO'}**")


def _render_pid_heartbeat(report: Dict[str, Any]) -> None:
    st.markdown("#### 💓 PID & Heartbeat")
    snap = (report.get("current_assessment") or {}).get("snapshot") or {}
    st.markdown(f"**PIDs:** {snap.get('pids') or 'None'}")
    st.markdown(f"**Heartbeat refresh:** {snap.get('heartbeat_refresh_ts') or '—'}")
    st.markdown(f"**Lock:** {'exists' if snap.get('lock_exists') else 'missing'}")


def _render_cycle(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Cycle Verification")
    snap = (report.get("current_assessment") or {}).get("snapshot") or {}
    st.markdown(f"Cycle count: **{snap.get('cycle_count', '—')}**")
    st.markdown(f"Consecutive OK: **{snap.get('consecutive_ok', '—')}**")
    st.markdown(f"Last cycle OK: **{snap.get('last_cycle_ok') or '—'}**")
    age = (report.get("current_assessment") or {}).get("cycle_age_minutes")
    if age is not None:
        st.markdown(f"Cycle age (min): **{round(age, 1)}**")


def _render_forensics(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔬 Startup Forensics")
    f = report.get("startup_forensics") or {}
    st.json(f) if f else st.info("No forensics.", icon="ℹ️")


def _render_history(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Historical Recovery Success Rate")
    stats = report.get("recovery_success_rate") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Attempts", str(stats.get("attempts", 0)))
    c2.metric("Successes", str(stats.get("successes", 0)))
    c3.metric("Success %", str(stats.get("success_pct", "—")))
    hist = _read_csv_safe(results_dir / "recovery_stabilization_history.csv")
    if not hist.empty:
        st.dataframe(hist.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
    events = _read_csv_safe(results_dir / "recovery_verification_events.csv")
    if not events.empty:
        with st.expander("Verification events (recent)"):
            st.dataframe(events.tail(20).iloc[::-1], use_container_width=True, hide_index=True)


def render_recovery_stabilization_monitor(results_dir: Path) -> None:
    st.markdown("### 🟢 Recovery Stabilization Intelligence")
    st.caption(
        "Adaptive operational readiness assessment for loop recovery. "
        "Refresh with `python -m services.recovery_stabilization_monitor` "
        "(add `--observe` for 30–60s live observation)."
    )
    report = _read_json_safe(results_dir / "recovery_stabilization_status.json")
    if not report:
        st.warning(
            "No stabilization report. Run `python -m services.recovery_stabilization_monitor`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_current_state(report)
    _render_timeline(report)
    _render_warmup(report)
    _render_pid_heartbeat(report)
    _render_cycle(report)
    _render_forensics(report)
    _render_history(results_dir, report)
