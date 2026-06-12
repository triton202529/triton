"""
TRITON Phase 185 — Recovery Candidate Deterioration Analysis dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _plotly_bar


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


_REV = {
    "LOW": "🔴 LOW",
    "MODERATE": "🟡 MODERATE",
    "HIGH": "🟢 HIGH",
    "VERY_HIGH": "🟢 VERY_HIGH",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [RECOVERY_CANDIDATE_DETERIORATION_REPORT]")
    ex = report.get("executive_report") or {}
    qs = ex.get("executive_questions") or {}

    for row in ex.get("candidates") or []:
        st.markdown(
            f"**{row.get('candidate')}** — current `{row.get('current_status')}` "
            f"(was `{row.get('historical_status')}`) · cause: **{row.get('deterioration_cause')}** · "
            f"recovery: {_REV.get(str(row.get('recovery_probability')), row.get('recovery_probability'))} · "
            f"replacements: {', '.join(row.get('replacement_candidates') or []) or '—'}"
        )
        st.caption(str(row.get("deterioration_detail") or ""))

    st.markdown(
        f"**Recovery basket viability:** {ex.get('recovery_basket_viability') or '—'} · "
        f"**Proposed rebuild:** {', '.join(ex.get('proposed_rebuilt_basket') or []) or '—'}"
    )

    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_summary(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Deterioration Summary")
    s = report.get("summary") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols Analyzed", str(s.get("symbols_analyzed", "—")))
    c2.metric("Deteriorated", str(s.get("deteriorated_count", "—")))
    c3.metric("OPEN_NEW Now", str(s.get("open_new_candidates_now", "—")))
    c4.metric("Basket Viability", str(s.get("recovery_basket_viability", "—")))
    st.caption(f"Pattern: {s.get('primary_deterioration_pattern', '—')}")


def _render_focus_comparison(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔍 HD / XLK / QQQ — Current vs Historical")
    rows = []
    for sym in ("HD", "XLK", "QQQ"):
        cand = next(
            (c for c in (report.get("candidate_analysis") or []) if c.get("symbol") == sym),
            {},
        )
        comp = cand.get("comparison") or {}
        rows.append(
            {
                "symbol": sym,
                "historical_status": cand.get("historical_status"),
                "current_status": cand.get("effective_stance"),
                "conf_hist": (comp.get("confidence") or {}).get("historical"),
                "conf_cur": (comp.get("confidence") or {}).get("current"),
                "edge_hist": (comp.get("edge_score") or {}).get("historical"),
                "edge_cur": (comp.get("edge_score") or {}).get("current"),
                "delta_cur": (comp.get("delta_pct") or {}).get("current"),
                "signal": cand.get("signal"),
                "root_cause": cand.get("root_cause_classification"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No focus data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_deterioration_flags(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Deterioration Dimensions")
    focus = [
        c for c in (report.get("candidate_analysis") or []) if c.get("symbol") in ("HD", "XLK")
    ]
    for c in focus:
        flags = c.get("deterioration_flags") or {}
        active = [k.replace("_", " ").title() for k, v in flags.items() if v is True]
        st.markdown(
            f"**{c['symbol']}** ({c.get('root_cause_classification')}): {', '.join(active) or '—'}"
        )


def _render_reversal(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔮 Reversal Analysis")
    rows = []
    for c in report.get("candidate_analysis") or []:
        if c.get("symbol") not in ("HD", "XLK", "BITO"):
            continue
        rev = c.get("reversal_analysis") or {}
        rows.append(
            {
                "symbol": c["symbol"],
                "probability_of_recovery": rev.get("probability_of_recovery"),
                "return_to_open_new": rev.get("return_to_open_new_probability"),
                "days_to_recovery": rev.get("days_to_recovery"),
                "promotion_30d_pct": rev.get("promotion_probability_30d_pct"),
                "reversal_nature": rev.get("reversal_nature"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No reversal data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_rebuild(report: Dict[str, Any]) -> None:
    st.markdown("#### 🧺 Recovery Basket Rebuild")
    rb = report.get("recovery_basket_rebuild") or {}
    c1, c2 = st.columns(2)
    c1.metric("Viability", str(rb.get("recovery_basket_viability", "—")))
    c2.metric("Lifecycle Actionable", str(rb.get("lifecycle_actionable_count", "—")))
    st.markdown(f"**Remove:** {', '.join(rb.get('should_remove_from_basket') or [])}")
    st.markdown(f"**HD replacements:** {', '.join(rb.get('hd_replacement_candidates') or [])}")
    st.markdown(f"**XLK replacements:** {', '.join(rb.get('xlk_replacement_candidates') or [])}")
    st.markdown(f"**Proposed basket:** {', '.join(rb.get('proposed_rebuilt_basket') or [])}")
    st.caption(str(rb.get("viability_note") or ""))
    caveat = (rb.get("simulation_reference") or {}).get("caveat")
    if caveat:
        st.warning(caveat, icon="⚠️")


def _render_viability(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### ✅ Candidate Viability")
    df = _read_csv_safe(results_dir / "recovery_candidate_viability.csv")
    if df.empty:
        repl = report.get("replacement_candidates") or []
        df = pd.DataFrame(repl)
    if df.empty:
        st.info("No viability data.", icon="ℹ️")
        return
    focus = df[df["symbol"].isin(["HD", "XLK", "QQQ"])] if "symbol" in df.columns else df
    repl = (
        df[df.get("role", pd.Series()) == "REPLACEMENT_CANDIDATE"]
        if "role" in df.columns
        else pd.DataFrame()
    )
    st.markdown("**Focus symbols**")
    st.dataframe(
        focus if not focus.empty else df.head(10), use_container_width=True, hide_index=True
    )
    if not repl.empty:
        st.markdown("**Top replacement candidates**")
        st.dataframe(repl.head(10), use_container_width=True, hide_index=True)
        if "viability_score" in repl.columns:
            chart = repl.head(8)[["symbol", "viability_score"]].copy()
            if not _plotly_bar(chart, "symbol", "viability_score", "Replacement Viability Score"):
                st.bar_chart(chart.set_index("symbol")["viability_score"])


def _render_history(results_dir: Path) -> None:
    st.markdown("#### 🕐 Candidate History")
    df = _read_csv_safe(results_dir / "recovery_candidate_history.csv")
    if df.empty:
        st.info("No history.", icon="ℹ️")
        return
    focus = df[df["symbol"].isin(["HD", "XLK", "QQQ"])] if "symbol" in df.columns else df
    st.dataframe(focus if not focus.empty else df, use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_recovery_candidate_deterioration_analysis(results_dir: Path) -> None:
    st.markdown("### 📉 Recovery Candidate Deterioration Analysis")
    st.caption(
        "Analyzes HD/XLK deterioration from OPEN_NEW to WAIT, reversal probability, "
        "and recovery basket rebuild options. "
        "Refresh with `python -m services.recovery_candidate_deterioration_analysis`."
    )
    report = _read_json_safe(results_dir / "recovery_candidate_deterioration_analysis.json")
    if not report:
        st.warning(
            "No analysis found. Run `python -m services.recovery_candidate_deterioration_analysis`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_summary(report)
    _render_focus_comparison(report)
    _render_deterioration_flags(report)
    _render_reversal(report)
    _render_rebuild(report)
    _render_viability(results_dir, report)
    _render_history(results_dir)
    _render_certification(report)
