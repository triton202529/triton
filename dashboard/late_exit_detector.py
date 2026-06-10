"""
TRITON Phase 172 — 🚨 Late Exit Detector (OBSERVE_ONLY dashboard).

Displays open-position late-exit scores, candidate registry, tracking history,
opportunity cost, historical accuracy, and governance enforcement banners.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _money, _num, _plotly_bar


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


_CLASS_BADGE = {
    "NORMAL": "🟢 NORMAL",
    "WATCH": "🟡 WATCH",
    "REVIEW": "🟠 REVIEW",
    "ESCALATE": "🔴 ESCALATE",
    "EXIT_CANDIDATE": "🟣 EXIT_CANDIDATE",
}
_ACC_BADGE = {
    "TRUE_POSITIVE": "🟢 TRUE_POSITIVE",
    "FALSE_POSITIVE": "🔴 FALSE_POSITIVE",
    "TRUE_NEGATIVE": "🟢 TRUE_NEGATIVE",
    "FALSE_NEGATIVE": "🟠 FALSE_NEGATIVE",
}
_CERT_BADGE = {
    "CERTIFIED": "🟢",
    "PARTIALLY_CERTIFIED": "🟡",
    "NOT_CERTIFIED": "🔴",
}


def _render_governance_banner(report: Dict[str, Any]) -> None:
    gov = report.get("governance") or {}
    st.error(
        "**OBSERVE_ONLY MODE** — "
        "NO ACTIONS GENERATED · NO ORDERS GENERATED · "
        "NO LIFECYCLE MODIFICATIONS · NO EXECUTION MODIFICATIONS",
        icon="🔒",
    )
    st.caption(str(gov.get("notice") or ""))


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    ex = report.get("executive_report") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open Positions Reviewed", str(ex.get("open_positions_reviewed", "—")))
    c2.metric("Candidates (WATCH+)", str(ex.get("candidate_count", "—")))
    c3.metric("Average Score", _num(ex.get("average_score")))
    c4.metric("Governance Mode", str(ex.get("governance_mode", "OBSERVE_ONLY")))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Highest Risk", str(ex.get("highest_risk_position", "—")))
    c6.metric("Avg Opportunity Cost", _money(ex.get("average_opportunity_cost")))
    c7.metric("Detector Accuracy", _num(ex.get("detector_accuracy")))
    c8.metric("Precision / Recall", f"{_num(ex.get('precision'))} / {_num(ex.get('recall'))}")


def _render_candidates(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Current Candidates")
    df = _read_csv_safe(results_dir / "late_exit_candidates.csv")
    if df.empty:
        st.info("No open positions to score.", icon="ℹ️")
        return
    view = df.copy()
    view["classification"] = view["classification"].map(lambda c: _CLASS_BADGE.get(str(c), str(c)))
    st.dataframe(view, use_container_width=True, hide_index=True)
    bands = report.get("score_bands") or {}
    if bands:
        st.caption("Score bands: " + " · ".join(f"{k}={v}" for k, v in bands.items()))


def _render_distribution(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Score Distribution")
    rows = report.get("current_candidates") or []
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No scores to chart.", icon="ℹ️")
        return
    dist = df["classification"].value_counts().reset_index()
    dist.columns = ["classification", "count"]
    if not _plotly_bar(dist, "classification", "count", "Classification Distribution"):
        st.bar_chart(dist.set_index("classification")["count"])
    if "score" in df.columns:
        st.dataframe(
            df[["ticker", "score", "classification"]].sort_values("score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def _render_tracking(results_dir: Path) -> None:
    st.markdown("#### 📈 Tracking History")
    hist = _read_csv_safe(results_dir / "late_exit_tracking_history.csv")
    if hist.empty:
        st.info("Tracking history will build as the detector runs over time.", icon="ℹ️")
        return
    tickers = sorted(hist["ticker"].unique().tolist())
    pick = st.selectbox("Ticker", tickers, key="led_ticker")
    sub = hist[hist["ticker"] == pick].copy()
    if "generated_at" in sub.columns:
        sub["generated_at"] = pd.to_datetime(sub["generated_at"], errors="coerce")
        sub = sub.sort_values("generated_at")
    if len(sub) >= 2 and "score" in sub.columns:
        delta = float(sub["score"].iloc[-1]) - float(sub["score"].iloc[0])
        trend = "improving" if delta < -5 else "deteriorating" if delta > 5 else "stable"
        st.metric("Score trend", f"{delta:+.1f} ({trend})")
    cols = [
        c
        for c in (
            "generated_at",
            "score",
            "classification",
            "estimated_opportunity_cost",
            "confidence_change",
        )
        if c in sub.columns
    ]
    st.dataframe(sub[cols], use_container_width=True, hide_index=True)


def _render_opportunity(report: Dict[str, Any]) -> None:
    st.markdown("#### 💸 Opportunity Cost Analysis")
    df = pd.DataFrame(report.get("current_candidates") or [])
    if df.empty:
        st.info("No opportunity cost data.", icon="ℹ️")
        return
    opp = pd.to_numeric(df["estimated_opportunity_cost"], errors="coerce").fillna(0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Estimated Cost", _money(opp.sum()))
    c2.metric("Average", _money(opp.mean()))
    c3.metric("Largest", _money(opp.max()))
    chart = df[["ticker", "estimated_opportunity_cost"]].sort_values(
        "estimated_opportunity_cost", ascending=False
    )
    if not _plotly_bar(
        chart, "ticker", "estimated_opportunity_cost", "Estimated Opportunity Cost by Position"
    ):
        st.bar_chart(chart.set_index("ticker")["estimated_opportunity_cost"])


def _render_accuracy(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Historical Accuracy")
    acc = report.get("accuracy") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", _num(acc.get("detector_accuracy")))
    c2.metric("Precision", _num(acc.get("precision")))
    c3.metric("Recall", _num(acc.get("recall")))
    c4.metric("Hit Rate", _num(acc.get("historical_hit_rate")))
    conf = acc.get("confusion") or {}
    if conf:
        st.caption(
            f"TP={conf.get('TP', 0)} · FP={conf.get('FP', 0)} · "
            f"TN={conf.get('TN', 0)} · FN={conf.get('FN', 0)} · "
            f"samples={acc.get('samples', 0)}"
        )
    rows = pd.DataFrame(acc.get("rows") or [])
    if not rows.empty:
        view = rows.copy()
        view["accuracy_label"] = view["accuracy_label"].map(
            lambda x: _ACC_BADGE.get(str(x), str(x))
        )
        st.dataframe(view, use_container_width=True, hide_index=True)


def _render_governance_status(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Governance Status")
    gov = report.get("governance") or {}
    ref = report.get("governance_reference") or {}
    st.json(
        {
            "mode": gov.get("mode"),
            "actions_allowed": gov.get("actions_allowed"),
            "execution_routing_allowed": gov.get("execution_routing_allowed"),
            "phase_171_approved_proposals": ref.get("phase_171_approval"),
            "framework_certification": ref.get("framework_certification"),
        }
    )
    st.warning(
        "Rollout sequence from Phase 171: OBSERVE_ONLY → ADVISORY_ONLY → PAPER_TRIAL. "
        "This phase implements step 1 only.",
        icon="🧭",
    )


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification Status")
    cert = report.get("detector_certification") or {}
    if not cert:
        st.info("No certification data.", icon="ℹ️")
        return
    status = str(cert.get("detector_certification_status", "—"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", f"{_CERT_BADGE.get(status, '⚪')} {status}")
    c2.metric("Data Quality", _num(cert.get("data_quality_score")))
    c3.metric("Tracking Snapshots", str(cert.get("tracking_snapshots", "—")))
    c4.metric("Accuracy Samples", str(cert.get("accuracy_sample_size", "—")))
    for r in cert.get("reasons") or []:
        st.warning(r, icon="⚠️")
    st.success(
        "Evidence-only detector — no lifecycle, execution, or broker behavior modified.",
        icon="🔒",
    )


def render_late_exit_detector(results_dir: Path) -> None:
    st.markdown("### 🚨 Late Exit Detector")
    st.caption(
        "OBSERVE_ONLY mode — detects, scores, logs, and tracks potential late exits. "
        "Refresh with `python -m services.late_exit_detector`."
    )

    report = _read_json_safe(results_dir / "late_exit_detector.json")
    if not report:
        st.info(
            "No detector report yet. Run `python -m services.late_exit_detector` to generate it.",
            icon="ℹ️",
        )
        return

    _render_governance_banner(report)
    _render_executive(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "🎯 Current Candidates",
            "📊 Score Distribution",
            "📈 Tracking History",
            "💸 Opportunity Cost",
            "🎯 Historical Accuracy",
            "⚖️ Governance Status",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="led_section",
    )
    if section == "🎯 Current Candidates":
        _render_candidates(results_dir, report)
    elif section == "📊 Score Distribution":
        _render_distribution(report)
    elif section == "📈 Tracking History":
        _render_tracking(results_dir)
    elif section == "💸 Opportunity Cost":
        _render_opportunity(report)
    elif section == "🎯 Historical Accuracy":
        _render_accuracy(report)
    elif section == "⚖️ Governance Status":
        _render_governance_status(report)
    else:
        _render_certification(report)
