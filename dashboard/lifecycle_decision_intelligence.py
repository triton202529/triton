"""
TRITON Phase 167 — 🧠 Lifecycle Decision Intelligence (read-only dashboard).

Renders the lifecycle decision intelligence audit: which decisions create
alpha, which destroy value, confidence calibration, risk-flag predictive
value, transition rankings, exit-timing cost, and symbol trust rankings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _money, _num, _pct, _plotly_bar


def _read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _table(records: Any) -> pd.DataFrame:
    try:
        df = pd.DataFrame(records or [])
        return df
    except Exception:
        return pd.DataFrame()


def _badge(classification: str) -> str:
    return {
        "Elite": "🟣 Elite",
        "Strong": "🟢 Strong",
        "Average": "🟡 Average",
        "Weak": "🟠 Weak",
        "Harmful": "🔴 Harmful",
    }.get(str(classification), str(classification or "—"))


def _render_summary(summary: Dict[str, Any], cert: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    grade = str(summary.get("overall_lifecycle_grade") or "—")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Lifecycle Grade", grade)
    c2.metric("Overall Alpha Score", _num(summary.get("overall_alpha_score")))
    c3.metric("Late Exit Cost", _money(summary.get("late_exit_cost")))
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        str(cert.get("status")), "⚪"
    )
    c4.metric("Certification", f"{badge} {cert.get('status', '—')}")

    pairs = [
        ("Best Decision Type", "best_decision_type", "Worst Decision Type", "worst_decision_type"),
        (
            "Best Confidence Range",
            "best_confidence_range",
            "Worst Confidence Range",
            "worst_confidence_range",
        ),
        (
            "Most Valuable Risk Flag",
            "most_valuable_risk_flag",
            "Least Valuable Risk Flag",
            "least_valuable_risk_flag",
        ),
        (
            "Best Transition",
            "best_lifecycle_transition",
            "Worst Transition",
            "worst_lifecycle_transition",
        ),
        (
            "Top Alpha Generator",
            "top_alpha_generator",
            "Top Alpha Destroyer",
            "top_alpha_destroyer",
        ),
    ]
    for left_label, left_key, right_label, right_key in pairs:
        l, r = st.columns(2)
        l.metric(left_label, str(summary.get(left_key) or "—"))
        r.metric(right_label, str(summary.get(right_key) or "—"))


def _render_rankings(df: pd.DataFrame) -> None:
    st.markdown("#### 🏆 Decision Rankings")
    if df.empty:
        st.info("No decision rankings available.", icon="ℹ️")
        return
    view = df.copy()
    view["classification"] = view["classification"].map(_badge)
    st.dataframe(view, use_container_width=True, hide_index=True)
    chart = df.dropna(subset=["alpha_score"])
    if not chart.empty and not _plotly_bar(
        chart, "action", "alpha_score", "Alpha Score by Decision Type"
    ):
        st.bar_chart(chart.set_index("action")["alpha_score"])
    st.caption(
        "Win rate / returns use alpha-aligned 5-day forward moves: entries win when "
        "price rises, trims/exits win when price falls afterwards."
    )


def _render_confidence(df: pd.DataFrame) -> None:
    st.markdown("#### 🎯 Confidence Analysis")
    if df.empty:
        st.info("No confidence data available.", icon="ℹ️")
        return
    view = df.copy()
    view["classification"] = view["classification"].map(_badge)
    st.dataframe(view, use_container_width=True, hide_index=True)
    chart = df.dropna(subset=["alpha_score"])
    if not chart.empty and not _plotly_bar(
        chart, "bucket", "alpha_score", "Alpha Score by Confidence Bucket"
    ):
        st.bar_chart(chart.set_index("bucket")["alpha_score"])

    # Does higher confidence actually lead to better outcomes?
    scored = df.dropna(subset=["alpha_score"])
    scored = scored[pd.to_numeric(scored["decision_count"], errors="coerce") >= 5]
    if len(scored) >= 2:
        order = list(scored["alpha_score"])
        monotone = all(order[i] <= order[i + 1] for i in range(len(order) - 1))
        top_is_best = scored["alpha_score"].idxmax() == scored.index[-1]
        if monotone:
            verdict = "YES — alpha rises consistently with confidence."
        elif top_is_best:
            verdict = "PARTIALLY — the highest-confidence bucket performs best, but the middle ranges are not monotonic."
        else:
            verdict = "NO — higher confidence does not currently translate into better outcomes."
        st.markdown(f"**Does higher confidence lead to better outcomes?** {verdict}")


def _render_flags(df: pd.DataFrame) -> None:
    st.markdown("#### 🚩 Risk Flag Intelligence")
    if df.empty:
        st.info("No risk-flag observations available.", icon="ℹ️")
        return
    observed = df[df.get("observed", pd.Series(dtype=bool)) == True]  # noqa: E712
    missing = df[df.get("observed", pd.Series(dtype=bool)) != True]  # noqa: E712
    if not observed.empty:
        view = observed.drop(columns=["observed"], errors="ignore").copy()
        view["classification"] = view["classification"].map(_badge)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption(
            "Flag win rate = share of flagged events followed by a price decline "
            "(the flag correctly predicted trouble)."
        )
    if not missing.empty:
        st.caption("Not yet observed in history: " + ", ".join(missing["flag"].astype(str)))


def _render_transitions(df: pd.DataFrame) -> None:
    st.markdown("#### 🔁 Lifecycle Transition Analysis")
    if df.empty:
        st.info("No lifecycle transitions recorded.", icon="ℹ️")
        return
    view = df.copy()
    view["classification"] = view["classification"].map(_badge)
    st.dataframe(view, use_container_width=True, hide_index=True)
    chart = df.dropna(subset=["alpha_score"]).head(12)
    if not chart.empty and not _plotly_bar(
        chart, "transition", "alpha_score", "Alpha Score by Transition"
    ):
        st.bar_chart(chart.set_index("transition")["alpha_score"])


def _render_exits(df: pd.DataFrame) -> None:
    st.markdown("#### 🚪 Exit Timing Audit")
    if df.empty:
        st.info("No classified exits available.", icon="ℹ️")
        return
    view = df.copy()
    if "classification" in view.columns:
        view["classification"] = view["classification"].map(_badge)
    st.dataframe(view, use_container_width=True, hide_index=True)

    late = df[df["exit_outcome"] == "LATE_EXIT"]
    early = df[df["exit_outcome"] == "EARLY_EXIT"]
    c1, c2, c3 = st.columns(3)
    saved = pd.to_numeric(df.get("capital_saved"), errors="coerce").sum()
    c1.metric("Capital Saved by Exits", _money(saved))
    late_cost = (
        pd.to_numeric(late.get("capital_lost"), errors="coerce").sum() if not late.empty else 0.0
    )
    c2.metric("Cost of Late Exits", _money(late_cost))
    early_cost = (
        pd.to_numeric(early.get("capital_lost"), errors="coerce").sum() if not early.empty else 0.0
    )
    c3.metric("Missed Gains (Early Exits)", _money(early_cost))
    st.caption(
        "Capital saved = exit value × subsequent 10-day decline avoided. "
        "Capital lost = exit value × subsequent 10-day rally missed."
    )


def _render_symbols(df: pd.DataFrame) -> None:
    st.markdown("#### 🧪 Symbol Intelligence")
    if df.empty:
        st.info("No symbol intelligence available.", icon="ℹ️")
        return
    scored = df.copy()
    scored["alpha_score"] = pd.to_numeric(scored["alpha_score"], errors="coerce")
    scored = scored[
        scored["alpha_score"].notna()
        & (pd.to_numeric(scored["decision_count"], errors="coerce").fillna(0) > 0)
    ]
    if scored.empty:
        st.info("No scored symbols yet.", icon="ℹ️")
        return
    gen = scored.sort_values("alpha_score", ascending=False).head(20).copy()
    dest = scored.sort_values("alpha_score", ascending=True).head(20).copy()
    gen["classification"] = gen["classification"].map(_badge)
    dest["classification"] = dest["classification"].map(_badge)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Top 20 Alpha Generators**")
        st.dataframe(gen, use_container_width=True, hide_index=True)
    with col_r:
        st.markdown("**Top 20 Alpha Destroyers**")
        st.dataframe(dest, use_container_width=True, hide_index=True)
    chart = pd.concat([gen.head(10), dest.head(10)]).drop_duplicates(subset=["symbol"])
    if not chart.empty and not _plotly_bar(
        chart, "symbol", "alpha_score", "Symbol Alpha Scores (extremes)"
    ):
        st.bar_chart(chart.set_index("symbol")["alpha_score"])


def _render_alpha_scores(results_dir: Path) -> None:
    st.markdown("#### 🧮 Decision Alpha Scores")
    p = results_dir / "lifecycle_alpha_scores.csv"
    if not p.is_file():
        st.info("lifecycle_alpha_scores.csv not generated yet.", icon="ℹ️")
        return
    try:
        df = pd.read_csv(p)
    except Exception:
        st.info("Could not read lifecycle_alpha_scores.csv.", icon="ℹ️")
        return
    if df.empty:
        st.info("Alpha score file is empty.", icon="ℹ️")
        return
    df["classification"] = df["classification"].map(_badge)
    pick = st.multiselect(
        "Entity types",
        options=sorted(df["entity_type"].astype(str).unique()),
        default=sorted(df["entity_type"].astype(str).unique()),
        key="ldi_entity_types",
    )
    view = df[df["entity_type"].isin(pick)] if pick else df
    st.dataframe(
        view.sort_values("alpha_score", ascending=False, na_position="last"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Score bands: 90–100 Elite · 75–89 Strong · 60–74 Average · 40–59 Weak · 0–39 Harmful"
    )


def _render_certification(cert: Dict[str, Any], report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification Status")
    if not cert:
        st.info("No certification data available.", icon="ℹ️")
        return
    c1, c2, c3, c4 = st.columns(4)
    badge = {"CERTIFIED": "🟢", "PARTIALLY_CERTIFIED": "🟡", "NOT_CERTIFIED": "🔴"}.get(
        str(cert.get("status")), "⚪"
    )
    c1.metric("Status", f"{badge} {cert.get('status', '—')}")
    c2.metric("Graded Decisions", f"{cert.get('graded_decisions', 0):,}")
    c3.metric("Classified Exits", f"{cert.get('classified_exits', 0):,}")
    c4.metric("Ledger Integrity", str(cert.get("ledger_integrity_status", "—")))
    reasons = cert.get("reasons") or []
    if reasons:
        for r in reasons:
            st.warning(r, icon="⚠️")
    else:
        st.success("Intelligence report meets data-sufficiency requirements.", icon="✅")
    st.caption(
        f"Events analyzed: {report.get('events_analyzed', 0):,} · generated {report.get('generated_at', '—')}"
    )


def render_lifecycle_decision_intelligence(results_dir: Path) -> None:
    st.markdown("### 🧠 Lifecycle Decision Intelligence")
    st.caption(
        "Which lifecycle decisions create alpha and which destroy value — audit and "
        "analytics only. Refresh with `python -m services.lifecycle_decision_intelligence`."
    )

    report = _read_json_safe(results_dir / "lifecycle_decision_intelligence.json")
    if not report:
        st.info(
            "No intelligence report yet. Run "
            "`python -m services.lifecycle_decision_intelligence` to generate it.",
            icon="ℹ️",
        )
        return

    summary = report.get("summary") or {}
    cert = report.get("certification") or {}

    _render_summary(summary, cert)
    st.divider()

    section = st.radio(
        "Section",
        [
            "🏆 Decision Rankings",
            "🎯 Confidence Analysis",
            "🚩 Risk Flags",
            "🔁 Transitions",
            "🚪 Exit Timing",
            "🧪 Symbol Intelligence",
            "🧮 Alpha Scores",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="ldi_section",
    )
    if section == "🏆 Decision Rankings":
        _render_rankings(_table(report.get("decision_rankings")))
    elif section == "🎯 Confidence Analysis":
        _render_confidence(_table(report.get("confidence_performance_table")))
    elif section == "🚩 Risk Flags":
        _render_flags(_table(report.get("risk_flag_intelligence")))
    elif section == "🔁 Transitions":
        _render_transitions(_table(report.get("transition_rankings")))
    elif section == "🚪 Exit Timing":
        _render_exits(_table(report.get("exit_timing_audit")))
    elif section == "🧪 Symbol Intelligence":
        _render_symbols(_table(report.get("symbol_intelligence")))
    elif section == "🧮 Alpha Scores":
        _render_alpha_scores(results_dir)
    else:
        _render_certification(cert, report)
