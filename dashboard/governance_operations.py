"""
Governance Operations Platform — Steps 139–146 (read-only).

CSV registries under data/governance/. No broker, execution, or trading logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DATA_DIR = ROOT / "data" / "governance"
RESULTS_DIR = ROOT / "data" / "results"

GOV_EVIDENCE_CSV = GOVERNANCE_DATA_DIR / "governance_evidence.csv"
GOV_AUDIT_CSV = GOVERNANCE_DATA_DIR / "governance_audit_log.csv"
GOV_DECISIONS_CSV = GOVERNANCE_DATA_DIR / "governance_decisions.csv"
GOV_ESCALATIONS_CSV = GOVERNANCE_DATA_DIR / "governance_escalations.csv"
GOV_INVESTIGATIONS_CSV = GOVERNANCE_DATA_DIR / "governance_investigations.csv"

SCHEMA_EVIDENCE = (
    "evidence_id",
    "category",
    "severity",
    "source",
    "title",
    "description",
    "created_at",
    "owner",
    "linked_decision_id",
    "linked_escalation_id",
    "linked_investigation_id",
    "linked_audit_id",
    "framework",
    "status",
    "tags",
)
SCHEMA_AUDIT = (
    "audit_id",
    "audit_type",
    "title",
    "description",
    "severity",
    "status",
    "owner",
    "opened_at",
    "closed_at",
    "linked_evidence_id",
    "linked_finding_id",
    "linked_remediation_id",
    "framework",
    "notes",
)
SCHEMA_DECISIONS = (
    "decision_id",
    "title",
    "category",
    "committee",
    "decision_type",
    "rationale",
    "vote_result",
    "approved_by",
    "created_at",
    "effective_date",
    "status",
    "linked_evidence_id",
    "linked_escalation_id",
    "linked_investigation_id",
    "framework",
    "notes",
)
SCHEMA_ESCALATIONS = (
    "escalation_id",
    "level",
    "source",
    "severity",
    "title",
    "description",
    "owner",
    "status",
    "opened_at",
    "closed_at",
    "linked_evidence_id",
    "linked_decision_id",
    "linked_investigation_id",
    "framework",
    "sla_due_at",
    "notes",
)
SCHEMA_INVESTIGATIONS = (
    "investigation_id",
    "title",
    "trigger",
    "owner",
    "severity",
    "findings",
    "recommendations",
    "status",
    "opened_at",
    "closed_at",
    "linked_evidence_id",
    "linked_escalation_id",
    "linked_decision_id",
    "linked_audit_id",
    "framework",
    "notes",
)

OPEN_ESCALATION = frozenset({"OPEN", "IN_REVIEW", "AWAITING_EVIDENCE", "ESCALATED"})
OPEN_INVESTIGATION = frozenset(
    {"OPEN", "IN_REVIEW", "FINDINGS_DRAFTED", "RECOMMENDATIONS_PENDING", "REMEDIATION_REQUIRED"}
)
OPEN_AUDIT = frozenset({"OPEN", "IN_REVIEW", "REMEDIATION_REQUIRED"})
PENDING_DECISION = frozenset({"PENDING", "ESCALATED"})
UNVERIFIED_EVIDENCE = frozenset({"PENDING", "UNKNOWN", ""})


def load_governance_csv(path: Path, expected_columns: Tuple[str, ...]) -> pd.DataFrame:
    """Load governance CSV; missing/empty/malformed → empty DataFrame with schema."""
    GOVERNANCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(expected_columns)
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        if path.stat().st_size == 0:
            return pd.DataFrame(columns=cols)
    except OSError:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame(columns=cols)
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    extra = [c for c in df.columns if c not in cols]
    if extra:
        df = df.drop(columns=extra, errors="ignore")
    return df[cols].fillna("")


def normalize_governance_status(value: Any, allowed: Optional[frozenset] = None) -> str:
    s = str(value or "").strip().upper()
    if not s:
        return "UNKNOWN"
    if allowed and s not in allowed:
        return s if s else "UNKNOWN"
    return s


def safe_parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    try:
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        if hasattr(ts, "to_pydatetime"):
            dt = ts.to_pydatetime()
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        pass
    return None


def compute_record_age_days(opened_at: Any, closed_at: Any = None) -> Optional[float]:
    start = safe_parse_datetime(opened_at)
    if start is None:
        return None
    end = safe_parse_datetime(closed_at) if closed_at else datetime.now(timezone.utc)
    if end is None:
        end = datetime.now(timezone.utc)
    return max(0.0, (end - start).total_seconds() / 86400.0)


def filter_dataframe_by_search(df: pd.DataFrame, query: str, columns: List[str]) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q or df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
    return df.loc[mask]


def render_governance_kpi_card(label: str, value: Any, helper: Optional[str] = None) -> None:
    st.metric(label, value, help=helper)


def render_selected_record(record: Dict[str, Any], title: str = "Record detail") -> None:
    with st.expander(title, expanded=True):
        if not record:
            st.caption("No record selected.")
            return
        for k, v in record.items():
            if v is not None and str(v).strip():
                st.markdown(f"**{k}:** `{v}`")


@st.cache_data(show_spinner=False)
def _load_all_governance_registries() -> Dict[str, pd.DataFrame]:
    return {
        "evidence": load_governance_csv(GOV_EVIDENCE_CSV, SCHEMA_EVIDENCE),
        "audit": load_governance_csv(GOV_AUDIT_CSV, SCHEMA_AUDIT),
        "decisions": load_governance_csv(GOV_DECISIONS_CSV, SCHEMA_DECISIONS),
        "escalations": load_governance_csv(GOV_ESCALATIONS_CSV, SCHEMA_ESCALATIONS),
        "investigations": load_governance_csv(GOV_INVESTIGATIONS_CSV, SCHEMA_INVESTIGATIONS),
    }


def _gov_empty_state(name: str) -> None:
    st.info(f"No {name} records found yet. Add rows to `data/governance/` CSV files when ready.")


def _gov_readonly_banner() -> None:
    st.info(
        "**Read-only governance operations.** This UI observes CSV registries only — "
        "no trading, broker, execution, or lifecycle changes."
    )


def _gov_plotly_bar(labels: List[str], values: List[int], title: str) -> None:
    if not values or sum(values) == 0:
        st.caption(f"No data for {title}.")
        return
    try:
        import plotly.express as px  # type: ignore

        fig = px.bar(x=labels, y=values, title=title, labels={"x": "", "y": "Count"})
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.bar_chart(pd.DataFrame({"count": values}, index=labels))


def _gov_last_updated(reg: Dict[str, pd.DataFrame]) -> str:
    times: List[datetime] = []
    for path in (
        GOV_EVIDENCE_CSV,
        GOV_AUDIT_CSV,
        GOV_DECISIONS_CSV,
        GOV_ESCALATIONS_CSV,
        GOV_INVESTIGATIONS_CSV,
    ):
        try:
            if path.exists():
                times.append(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
        except OSError:
            pass
    if not times:
        return "—"
    return max(times).strftime("%Y-%m-%d %H:%M UTC")


def _gov_compute_health(reg: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    esc = reg["escalations"]
    inv = reg["investigations"]
    aud = reg["audit"]
    dec = reg["decisions"]
    ev = reg["evidence"]

    open_esc = (
        len(esc[esc["status"].astype(str).str.upper().isin(OPEN_ESCALATION)])
        if not esc.empty
        else 0
    )
    crit_esc = (
        len(
            esc[
                esc["severity"].astype(str).str.upper().isin({"CRITICAL", "HIGH"})
                & esc["status"].astype(str).str.upper().isin(OPEN_ESCALATION)
            ]
        )
        if not esc.empty
        else 0
    )
    open_inv = (
        len(inv[inv["status"].astype(str).str.upper().isin(OPEN_INVESTIGATION)])
        if not inv.empty
        else 0
    )
    open_aud = (
        len(aud[aud["status"].astype(str).str.upper().isin(OPEN_AUDIT)]) if not aud.empty else 0
    )
    pend_dec = (
        len(dec[dec["status"].astype(str).str.upper().isin(PENDING_DECISION)])
        if not dec.empty
        else 0
    )
    unver_ev = (
        len(ev[ev["status"].astype(str).str.upper().isin(UNVERIFIED_EVIDENCE)])
        if not ev.empty
        else 0
    )

    score = 100.0
    drivers: List[str] = []
    if open_esc:
        score -= min(30, open_esc * 5)
        drivers.append(f"{open_esc} open escalation(s)")
    if crit_esc:
        score -= min(25, crit_esc * 8)
        drivers.append(f"{crit_esc} critical/high escalation(s)")
    if open_inv:
        score -= min(20, open_inv * 4)
        drivers.append(f"{open_inv} open investigation(s)")
    if open_aud:
        score -= min(20, open_aud * 5)
        drivers.append(f"{open_aud} open audit(s)")
    if pend_dec:
        score -= min(15, pend_dec * 3)
        drivers.append(f"{pend_dec} pending decision(s)")
    if unver_ev:
        score -= min(10, unver_ev * 2)
        drivers.append(f"{unver_ev} unverified evidence item(s)")

    score = max(0.0, min(100.0, score))
    if score >= 85:
        state = "HEALTHY"
    elif score >= 70:
        state = "WATCH"
    elif score >= 50:
        state = "DEGRADED"
    else:
        state = "CRITICAL"
    if not any([open_esc, crit_esc, open_inv, open_aud, pend_dec, unver_ev]) and all(
        df.empty for df in reg.values()
    ):
        state = "UNKNOWN"

    return {
        "score": round(score, 1),
        "state": state,
        "drivers": drivers[:6] or ["No registry activity recorded."],
        "open_escalations": open_esc,
        "critical_escalations": crit_esc,
        "open_investigations": open_inv,
        "open_audits": open_aud,
        "pending_decisions": pend_dec,
        "unverified_evidence": unver_ev,
    }


def build_governance_timeline(
    reg: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    reg = reg or _load_all_governance_registries()
    rows: List[Dict[str, Any]] = []

    def _add(
        df: pd.DataFrame, etype: str, id_col: str, time_cols: List[str], title_col: str
    ) -> None:
        if df.empty:
            return
        for _, r in df.iterrows():
            for tc in time_cols:
                if tc not in df.columns:
                    continue
                ts = safe_parse_datetime(r.get(tc))
                if ts is None:
                    continue
                rows.append(
                    {
                        "event_time": ts,
                        "event_type": etype,
                        "event_id": str(r.get(id_col, "")),
                        "title": str(r.get(title_col, "")),
                        "severity": str(r.get("severity", "")),
                        "status": str(r.get("status", "")),
                        "owner": str(r.get("owner", "")),
                        "source_file": etype,
                        "summary": str(r.get("description", r.get("notes", "")))[:200],
                    }
                )

    _add(reg["evidence"], "EVIDENCE", "evidence_id", ["created_at"], "title")
    _add(reg["audit"], "AUDIT", "audit_id", ["opened_at", "closed_at"], "title")
    _add(reg["decisions"], "DECISION", "decision_id", ["created_at", "effective_date"], "title")
    _add(reg["escalations"], "ESCALATION", "escalation_id", ["opened_at", "closed_at"], "title")
    _add(
        reg["investigations"],
        "INVESTIGATION",
        "investigation_id",
        ["opened_at", "closed_at"],
        "title",
    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_time",
                "event_type",
                "event_id",
                "title",
                "severity",
                "status",
                "owner",
                "source_file",
                "summary",
            ]
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("event_time", ascending=False)
    out["event_time"] = out["event_time"].apply(
        lambda t: t.strftime("%Y-%m-%d %H:%M UTC") if isinstance(t, datetime) else str(t)
    )
    return out


def _gov_collect_links(reg: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], List[Dict]]:
    """Return (edges, orphans) for traceability."""
    edges: List[Dict[str, str]] = []
    known_ids: Dict[str, set] = {
        "evidence": set(),
        "audit": set(),
        "decision": set(),
        "escalation": set(),
        "investigation": set(),
    }

    def _kid(etype: str, row: pd.Series, col: str) -> Optional[str]:
        v = str(row.get(col, "")).strip()
        return v if v else None

    link_specs = [
        (
            "evidence",
            reg["evidence"],
            "evidence_id",
            [
                ("linked_decision_id", "decision"),
                ("linked_escalation_id", "escalation"),
                ("linked_investigation_id", "investigation"),
                ("linked_audit_id", "audit"),
            ],
        ),
        (
            "audit",
            reg["audit"],
            "audit_id",
            [
                ("linked_evidence_id", "evidence"),
                ("linked_finding_id", "finding"),
                ("linked_remediation_id", "remediation"),
            ],
        ),
        (
            "decisions",
            reg["decisions"],
            "decision_id",
            [
                ("linked_evidence_id", "evidence"),
                ("linked_escalation_id", "escalation"),
                ("linked_investigation_id", "investigation"),
            ],
        ),
        (
            "escalations",
            reg["escalations"],
            "escalation_id",
            [
                ("linked_evidence_id", "evidence"),
                ("linked_decision_id", "decision"),
                ("linked_investigation_id", "investigation"),
            ],
        ),
        (
            "investigations",
            reg["investigations"],
            "investigation_id",
            [
                ("linked_evidence_id", "evidence"),
                ("linked_escalation_id", "escalation"),
                ("linked_decision_id", "decision"),
                ("linked_audit_id", "audit"),
            ],
        ),
    ]

    for src_type, df, id_col, links in link_specs:
        if df.empty:
            continue
        for _, row in df.iterrows():
            sid = _kid(src_type, row, id_col)
            if sid:
                known_ids[src_type if src_type != "decisions" else "decision"].add(sid)
            for lcol, tgt in links:
                tid = _kid(src_type, row, lcol)
                if sid and tid:
                    edges.append(
                        {"from_type": src_type, "from_id": sid, "to_type": tgt, "to_id": tid}
                    )

    orphans: List[Dict[str, str]] = []
    for etype, df, idc in [
        ("evidence", reg["evidence"], "evidence_id"),
        ("audit", reg["audit"], "audit_id"),
        ("decision", reg["decisions"], "decision_id"),
        ("escalation", reg["escalations"], "escalation_id"),
        ("investigation", reg["investigations"], "investigation_id"),
    ]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            eid = str(row.get(idc, "")).strip()
            if not eid:
                continue
            linked = False
            for e in edges:
                if e["from_id"] == eid or e["to_id"] == eid:
                    linked = True
                    break
            if not linked and len(df) > 1:
                orphans.append({"entity_type": etype, "entity_id": eid})

    return edges, orphans


def _gov_traceability_score(reg: Dict[str, pd.DataFrame]) -> float:
    total = sum(len(df) for df in reg.values() if not df.empty)
    if total == 0:
        return 0.0
    edges, orphans = _gov_collect_links(reg)
    if not edges and total > 0:
        return 25.0
    linked_entities = len({e["from_id"] for e in edges} | {e["to_id"] for e in edges})
    score = min(100.0, (linked_entities / max(total, 1)) * 100.0)
    score = max(0.0, score - len(orphans) * 2)
    return round(score, 1)


def _gov_filters_row(
    df: pd.DataFrame,
    *,
    search_cols: List[str],
    filters: List[Tuple[str, str, List[str]]],
) -> pd.DataFrame:
    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("Search", key=f"gov_search_{id(df)}")
    out = filter_dataframe_by_search(df, q, search_cols)
    cols = st.columns(min(len(filters), 4) or 1)
    for i, (label, col, options) in enumerate(filters):
        if col not in out.columns:
            continue
        with cols[i % len(cols)]:
            pick = st.selectbox(label, ["All"] + options, key=f"gov_f_{col}_{id(df)}")
            if pick != "All":
                out = out[out[col].astype(str).str.upper() == pick.upper()]
    return out


def _gov_row_dict(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    if df.empty or idx < 0 or idx >= len(df):
        return {}
    return {k: ("" if pd.isna(v) else v) for k, v in df.iloc[idx].items()}


# ── Pages ─────────────────────────────────────────────────────────────


def page_governance_evidence_registry() -> None:
    st.title("🗂 Governance Evidence Registry")
    _gov_readonly_banner()
    reg = _load_all_governance_registries()
    df = reg["evidence"]

    verified = len(df[df["status"].astype(str).str.upper() == "VERIFIED"]) if not df.empty else 0
    pending = (
        len(df[df["status"].astype(str).str.upper().isin(UNVERIFIED_EVIDENCE)])
        if not df.empty
        else 0
    )
    critical = len(df[df["severity"].astype(str).str.upper() == "CRITICAL"]) if not df.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Evidence Records", len(df))
    c2.metric("Verified Records", verified)
    c3.metric("Pending Verification", pending)
    c4.metric("Critical Evidence", critical)
    c5.metric("Last Updated", _gov_last_updated(reg))

    if df.empty:
        _gov_empty_state("evidence")
        return

    cats = sorted({str(x) for x in df["category"].unique() if str(x).strip()}) or ["OTHER"]
    sevs = sorted({str(x) for x in df["severity"].unique() if str(x).strip()})
    stats = sorted({str(x) for x in df["status"].unique() if str(x).strip()})
    owners = sorted({str(x) for x in df["owner"].unique() if str(x).strip()})

    filtered = _gov_filters_row(
        df,
        search_cols=list(SCHEMA_EVIDENCE),
        filters=[
            ("Category", "category", cats),
            ("Severity", "severity", sevs),
            ("Status", "status", stats),
            ("Owner", "owner", owners),
        ],
    )
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if not filtered.empty:
        labels = [
            f"{r.get('evidence_id','')} — {r.get('title','')[:40]}" for _, r in filtered.iterrows()
        ]
        pick = st.selectbox("Select evidence", labels, key="gov_ev_pick")
        idx = labels.index(pick) if pick in labels else 0
        render_selected_record(_gov_row_dict(filtered.reset_index(drop=True), idx))
        st.caption(
            f"Quality: {verified} verified / {len(df)} total · "
            f"{len(df) - verified - pending} other statuses"
        )


def page_governance_audit_center() -> None:
    st.title("🧾 Governance Audit Center")
    _gov_readonly_banner()
    reg = _load_all_governance_registries()
    df = reg["audit"]

    open_a = len(df[df["status"].astype(str).str.upper().isin(OPEN_AUDIT)]) if not df.empty else 0
    crit = (
        len(df[df["severity"].astype(str).str.upper().isin({"CRITICAL", "HIGH"})])
        if not df.empty
        else 0
    )
    unresolved = open_a
    res_times: List[float] = []
    if not df.empty:
        closed = df[df["status"].astype(str).str.upper().isin({"RESOLVED", "CLOSED"})]
        for _, r in closed.iterrows():
            age = compute_record_age_days(r.get("opened_at"), r.get("closed_at"))
            if age is not None:
                res_times.append(age)
    avg_res = f"{sum(res_times) / len(res_times):.1f}d" if res_times else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Audits", len(df))
    c2.metric("Open Audits", open_a)
    c3.metric("Critical Audits", crit)
    c4.metric("Unresolved Findings", unresolved)
    c5.metric("Avg Resolution Time", avg_res)

    if df.empty:
        _gov_empty_state("audit")
        return

    filtered = _gov_filters_row(
        df,
        search_cols=list(SCHEMA_AUDIT),
        filters=[
            ("Status", "status", sorted(df["status"].unique().astype(str))),
            ("Severity", "severity", sorted(df["severity"].unique().astype(str))),
        ],
    )
    st.markdown("### Audit log")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    recent = df.copy()
    if "opened_at" in recent.columns:
        recent["_ts"] = recent["opened_at"].apply(safe_parse_datetime)
        recent = recent.sort_values("_ts", ascending=False, na_position="last").head(10)
    st.markdown("### Recent audit trail")
    st.dataframe(
        recent.drop(columns=["_ts"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

    if not filtered.empty:
        labels = [
            f"{r.get('audit_id','')} — {r.get('title','')[:40]}" for _, r in filtered.iterrows()
        ]
        pick = st.selectbox("Select audit", labels, key="gov_aud_pick")
        idx = labels.index(pick) if pick in labels else 0
        render_selected_record(_gov_row_dict(filtered.reset_index(drop=True), idx))


def page_governance_decision_registry() -> None:
    st.title("⚖️ Governance Decision Registry")
    _gov_readonly_banner()
    reg = _load_all_governance_registries()
    df = reg["decisions"]

    approved = len(df[df["status"].astype(str).str.upper() == "APPROVED"]) if not df.empty else 0
    rejected = len(df[df["status"].astype(str).str.upper() == "REJECTED"]) if not df.empty else 0
    pending = (
        len(df[df["status"].astype(str).str.upper().isin(PENDING_DECISION)]) if not df.empty else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Decisions", len(df))
    c2.metric("Approved", approved)
    c3.metric("Rejected", rejected)
    c4.metric("Pending", pending)
    c5.metric("Recent (90d)", len(df))

    if df.empty:
        _gov_empty_state("decision")
        return

    filtered = _gov_filters_row(
        df,
        search_cols=list(SCHEMA_DECISIONS),
        filters=[
            ("Status", "status", sorted(df["status"].unique().astype(str))),
            ("Category", "category", sorted(df["category"].unique().astype(str))),
        ],
    )
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if not df.empty and "status" in df.columns:
        vc = df["status"].astype(str).str.upper().value_counts()
        _gov_plotly_bar(list(vc.index), list(vc.values), "Decision status distribution")

    if not filtered.empty:
        labels = [
            f"{r.get('decision_id','')} — {r.get('title','')[:40]}" for _, r in filtered.iterrows()
        ]
        pick = st.selectbox("Select decision", labels, key="gov_dec_pick")
        idx = labels.index(pick) if pick in labels else 0
        render_selected_record(_gov_row_dict(filtered.reset_index(drop=True), idx))


def page_governance_escalation_registry() -> None:
    st.title("🚨 Governance Escalation Registry")
    _gov_readonly_banner()
    reg = _load_all_governance_registries()
    df = reg["escalations"]

    open_e = df[df["status"].astype(str).str.upper().isin(OPEN_ESCALATION)] if not df.empty else df
    crit = (
        open_e[open_e["severity"].astype(str).str.upper().isin({"CRITICAL", "HIGH"})]
        if not open_e.empty
        else open_e
    )
    overdue = 0
    ages: List[float] = []
    if not open_e.empty:
        now = datetime.now(timezone.utc)
        for _, r in open_e.iterrows():
            age = compute_record_age_days(r.get("opened_at"))
            if age is not None:
                ages.append(age)
            sla = safe_parse_datetime(r.get("sla_due_at"))
            if sla and sla < now:
                overdue += 1
    closed = len(df) - len(open_e) if not df.empty else 0
    avg_age = f"{sum(ages) / len(ages):.1f}d" if ages else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Escalations", len(open_e))
    c2.metric("Critical Escalations", len(crit))
    c3.metric("Overdue Escalations", overdue)
    c4.metric("Average Age", avg_age)
    c5.metric("Closed Escalations", closed)

    if df.empty:
        _gov_empty_state("escalation")
        return

    st.markdown("### Active escalations")
    st.dataframe(open_e, use_container_width=True, hide_index=True)

    if not df.empty and "severity" in df.columns:
        vc = df["severity"].astype(str).value_counts()
        _gov_plotly_bar(list(vc.index), list(vc.values), "Severity distribution")

    filtered = _gov_filters_row(
        df,
        search_cols=list(SCHEMA_ESCALATIONS),
        filters=[
            ("Level", "level", sorted(df["level"].unique().astype(str))),
            ("Status", "status", sorted(df["status"].unique().astype(str))),
        ],
    )
    if not filtered.empty:
        labels = [
            f"{r.get('escalation_id','')} — {r.get('title','')[:40]}"
            for _, r in filtered.iterrows()
        ]
        pick = st.selectbox("Select escalation", labels, key="gov_esc_pick")
        idx = labels.index(pick) if pick in labels else 0
        render_selected_record(_gov_row_dict(filtered.reset_index(drop=True), idx))


def page_governance_traceability_explorer() -> None:
    st.title("🔗 Governance Traceability Explorer")
    _gov_readonly_banner()
    reg = _load_all_governance_registries()
    score = _gov_traceability_score(reg)
    edges, orphans = _gov_collect_links(reg)

    st.metric("Traceability completeness score", f"{score}%")
    id_q = st.text_input("Search by any ID", placeholder="GOV-… / evidence_id / escalation_id …")

    entity_types = ["evidence", "audit", "decision", "escalation", "investigation"]
    etype = st.selectbox("Entity type", entity_types, key="gov_tr_type")

    df_map = {
        "evidence": (reg["evidence"], "evidence_id"),
        "audit": (reg["audit"], "audit_id"),
        "decision": (reg["decisions"], "decision_id"),
        "escalation": (reg["escalations"], "escalation_id"),
        "investigation": (reg["investigations"], "investigation_id"),
    }
    df, id_col = df_map[etype]

    if df.empty:
        _gov_empty_state("registry")
        return

    if id_q:
        df = filter_dataframe_by_search(df, id_q, [id_col] + list(df.columns))

    labels = [str(r.get(id_col, "")) for _, r in df.iterrows() if str(r.get(id_col, "")).strip()]
    if not labels:
        st.info("No matching entities.")
        return

    sel = st.selectbox("Select entity", labels, key="gov_tr_sel")
    match = df[df[id_col].astype(str) == sel] if sel else pd.DataFrame()
    if not match.empty:
        render_selected_record(_gov_row_dict(match.reset_index(drop=True), 0))

    up = [e for e in edges if e["to_id"] == sel]
    down = [e for e in edges if e["from_id"] == sel]
    st.markdown("### Upstream links")
    st.dataframe(pd.DataFrame(up) if up else pd.DataFrame(), hide_index=True)
    st.markdown("### Downstream links")
    st.dataframe(pd.DataFrame(down) if down else pd.DataFrame(), hide_index=True)
    st.markdown("### Relationship table")
    st.dataframe(
        pd.DataFrame(edges) if edges else pd.DataFrame(), use_container_width=True, hide_index=True
    )

    broken = [e for e in edges if not e.get("to_id") or not e.get("from_id")]
    st.markdown("### Broken / orphan detection")
    if broken:
        st.warning(f"{len(broken)} broken link(s) detected.")
        st.dataframe(pd.DataFrame(broken), hide_index=True)
    if orphans:
        st.warning(f"{len(orphans)} potentially orphaned record(s).")
        st.dataframe(pd.DataFrame(orphans), hide_index=True)
    if not broken and not orphans:
        st.success("No broken links or orphans detected in current registries.")


def page_governance_investigation_center() -> None:
    st.title("🕵️ Governance Investigation Center")
    _gov_readonly_banner()
    reg = _load_all_governance_registries()
    df = reg["investigations"]

    open_i = (
        df[df["status"].astype(str).str.upper().isin(OPEN_INVESTIGATION)] if not df.empty else df
    )
    crit = (
        open_i[open_i["severity"].astype(str).str.upper().isin({"CRITICAL", "HIGH"})]
        if not open_i.empty
        else open_i
    )
    closed = (
        df[~df["status"].astype(str).str.upper().isin(OPEN_INVESTIGATION)] if not df.empty else df
    )
    ages = [
        compute_record_age_days(r.get("opened_at"))
        for _, r in open_i.iterrows()
        if compute_record_age_days(r.get("opened_at")) is not None
    ]
    rec_pending = (
        len(open_i[open_i["status"].astype(str).str.upper() == "RECOMMENDATIONS_PENDING"])
        if not open_i.empty
        else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Investigations", len(open_i))
    c2.metric("Critical Investigations", len(crit))
    c3.metric("Closed Investigations", len(closed))
    c4.metric("Avg Open Age (days)", f"{sum(ages)/len(ages):.1f}" if ages else "—")
    c5.metric("Recommendations Pending", rec_pending)

    if df.empty:
        _gov_empty_state("investigation")
        return

    st.markdown("### Open investigations")
    st.dataframe(open_i, use_container_width=True, hide_index=True)
    st.markdown("### Closed investigations")
    st.dataframe(closed, use_container_width=True, hide_index=True)

    if "status" in df.columns:
        vc = df["status"].astype(str).value_counts()
        _gov_plotly_bar(list(vc.index), list(vc.values), "Investigation status distribution")

    if not open_i.empty:
        labels = [
            f"{r.get('investigation_id','')} — {r.get('title','')[:40]}"
            for _, r in open_i.iterrows()
        ]
        pick = st.selectbox("Select investigation", labels, key="gov_inv_pick")
        idx = labels.index(pick) if pick in labels else 0
        rec = _gov_row_dict(open_i.reset_index(drop=True), idx)
        render_selected_record(rec)
        st.markdown("**Findings**")
        st.write(rec.get("findings") or "—")
        st.markdown("**Recommendations**")
        st.write(rec.get("recommendations") or "—")


def page_governance_intelligence_lab() -> None:
    st.title("🧠 Governance Intelligence Lab")
    _gov_readonly_banner()
    reg = _load_all_governance_registries()
    health = _gov_compute_health(reg)

    st.markdown("### Governance Health")
    c1, c2, c3 = st.columns(3)
    c1.metric("Governance Health Score", health["score"])
    c2.metric("Governance Health State", health["state"])
    c3.metric("Open Escalations", health["open_escalations"])
    st.markdown("**Top risk drivers**")
    for d in health["drivers"]:
        st.markdown(f"- {d}")

    esc, inv, dec, aud, ev = (
        reg["escalations"],
        reg["investigations"],
        reg["decisions"],
        reg["audit"],
        reg["evidence"],
    )

    st.markdown("### Escalation analytics")
    if esc.empty:
        st.caption("No escalation data.")
    else:
        _gov_plotly_bar(
            list(esc["severity"].astype(str).value_counts().index),
            list(esc["severity"].astype(str).value_counts().values),
            "Escalation severity",
        )
        if "owner" in esc.columns:
            ob = esc[esc["status"].astype(str).str.upper().isin(OPEN_ESCALATION)][
                "owner"
            ].value_counts()
            st.caption(f"Owner backlog (open): {ob.to_dict() if not ob.empty else 'none'}")

    st.markdown("### Investigation analytics")
    if inv.empty:
        st.caption("No investigation data.")
    else:
        open_n = len(inv[inv["status"].astype(str).str.upper().isin(OPEN_INVESTIGATION)])
        st.write(f"Open: {open_n} · Closed: {len(inv) - open_n}")

    st.markdown("### Decision analytics")
    if dec.empty:
        st.caption("No decision data.")
    else:
        total = max(len(dec), 1)
        st.write(
            f"Approval rate: {len(dec[dec['status'].astype(str).str.upper()=='APPROVED'])/total*100:.0f}% · "
            f"Rejection: {len(dec[dec['status'].astype(str).str.upper()=='REJECTED'])/total*100:.0f}% · "
            f"Pending: {len(dec[dec['status'].astype(str).str.upper().isin(PENDING_DECISION)])}"
        )

    st.markdown("### Audit analytics")
    if aud.empty:
        st.caption("No audit data.")
    else:
        st.write(
            f"Open audits: {len(aud[aud['status'].astype(str).str.upper().isin(OPEN_AUDIT)])} · "
            f"Unresolved: {len(aud[aud['status'].astype(str).str.upper().isin(OPEN_AUDIT)])}"
        )

    st.markdown("### Evidence analytics")
    if ev.empty:
        st.caption("No evidence data.")
    else:
        ver = len(ev[ev["status"].astype(str).str.upper() == "VERIFIED"])
        st.write(f"Verified: {ver} · Pending/unknown: {len(ev) - ver}")
        edges, orphans = _gov_collect_links(reg)
        orphan_ev = [o for o in orphans if o.get("entity_type") == "evidence"]
        st.write(f"Orphan evidence (heuristic): {len(orphan_ev)}")

    st.markdown("### Committee performance")
    pend = health["pending_decisions"]
    open_esc = health["open_escalations"]
    st.write(
        f"Pending decisions: {pend} · Open escalations: {open_esc} · "
        f"Escalation closure rate: "
        f"{(len(reg['escalations']) - open_esc) / max(len(reg['escalations']), 1) * 100:.0f}%"
        if not reg["escalations"].empty
        else "n/a"
    )


def page_governance_timeline_center() -> None:
    st.title("🕰 Governance Timeline Center")
    _gov_readonly_banner()
    reg = _load_all_governance_registries()
    tl = build_governance_timeline(reg)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total events", len(tl))
    c2.metric("Sources", tl["event_type"].nunique() if not tl.empty else 0)
    c3.metric("Last Updated", _gov_last_updated(reg))

    if tl.empty:
        _gov_empty_state("timeline")
        return

    window = st.radio("Time window", ["Recent 30", "Recent 90", "All"], horizontal=True)
    filtered = tl.copy()
    if window != "All" and "event_time" in filtered.columns:
        days = 30 if window == "Recent 30" else 90
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        ts = pd.to_datetime(filtered["event_time"], utc=True, errors="coerce")
        filtered = filtered[ts >= cutoff]

    q = st.text_input("Search timeline", key="gov_tl_search")
    filtered = filter_dataframe_by_search(
        filtered, q, ["event_id", "title", "event_type", "owner", "summary"]
    )
    et = st.multiselect(
        "Event types",
        sorted(tl["event_type"].unique().tolist()),
        default=sorted(tl["event_type"].unique().tolist()),
        key="gov_tl_types",
    )
    if et:
        filtered = filtered[filtered["event_type"].isin(et)]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if not filtered.empty:
        labels = [
            f"{r.get('event_time','')} | {r.get('event_type','')} | {r.get('event_id','')}"
            for _, r in filtered.iterrows()
        ]
        pick = st.selectbox("Selected event", labels, key="gov_tl_pick")
        idx = labels.index(pick) if pick in labels else 0
        render_selected_record(_gov_row_dict(filtered.reset_index(drop=True), idx))

    st.download_button(
        "Export timeline (CSV)",
        data=filtered.to_csv(index=False),
        file_name="governance_timeline_export.csv",
        mime="text/csv",
    )


def render_gcc_operations_overview(
    *,
    dossier_summary: Dict[str, Any],
    guard_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """Governance Operations Overview — top section for GCC (Card 9)."""
    reg = _load_all_governance_registries()
    health = _gov_compute_health(reg)
    guard = guard_snapshot or {}

    st.markdown("### Governance Operations Overview")
    st.caption("Institutional operations dashboard — CSV registries + runtime GCC posture.")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Governance Health Score", health["score"])
    k2.metric("Open Escalations", health["open_escalations"])
    k3.metric("Active Investigations", health["open_investigations"])
    k4.metric("Open Audits", health["open_audits"])
    k5.metric("Pending Decisions", health["pending_decisions"])
    k6.metric("Unverified Evidence", health["unverified_evidence"])

    st.markdown("#### Constitutional status")
    mode = str(guard.get("mode") or "UNKNOWN").upper()
    hard_halt = "YES" if mode in ("HARD_HALT", "HALT", "LOCKED") else "NO"
    override_st = str(guard.get("reason") or "—")[:80]
    mut = bool(dossier_summary.get("runtime_mutation_allowed", False))
    esc = reg["escalations"]
    viol = (
        len(
            esc[
                (esc["level"].astype(str).str.upper() == "CONSTITUTIONAL")
                & (esc["status"].astype(str).str.upper().isin(OPEN_ESCALATION))
            ]
        )
        if not esc.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hard Halt Status", hard_halt)
    c2.metric("Override / Guard Mode", mode)
    c3.metric("Governance Violations (open L4/const.)", viol)
    c4.metric("Runtime Mutation Allowed", "Yes" if mut else "No")
    st.caption(f"Guard reason: {override_st}")

    st.markdown("#### Action queue (review required)")
    queue: List[Dict[str, str]] = []

    def _queue_from(
        df: pd.DataFrame, idc: str, titlec: str, kind: str, open_set: frozenset
    ) -> None:
        if df.empty:
            return
        sub = df[df["status"].astype(str).str.upper().isin(open_set)]
        for _, r in sub.head(5).iterrows():
            queue.append(
                {
                    "type": kind,
                    "id": str(r.get(idc, "")),
                    "title": str(r.get(titlec, ""))[:60],
                    "severity": str(r.get("severity", "")),
                }
            )

    _queue_from(reg["escalations"], "escalation_id", "title", "Escalation", OPEN_ESCALATION)
    _queue_from(
        reg["investigations"], "investigation_id", "title", "Investigation", OPEN_INVESTIGATION
    )
    _queue_from(reg["decisions"], "decision_id", "title", "Decision", PENDING_DECISION)
    _queue_from(reg["audit"], "audit_id", "title", "Audit", OPEN_AUDIT)
    ev = reg["evidence"]
    if not ev.empty:
        sub = ev[ev["status"].astype(str).str.upper().isin(UNVERIFIED_EVIDENCE)]
        for _, r in sub.head(5).iterrows():
            queue.append(
                {
                    "type": "Evidence",
                    "id": str(r.get("evidence_id", "")),
                    "title": str(r.get("title", ""))[:60],
                    "severity": str(r.get("severity", "")),
                }
            )

    if queue:
        st.dataframe(pd.DataFrame(queue), use_container_width=True, hide_index=True)
    else:
        st.caption("No open governance operations items in CSV registries.")

    st.markdown("#### Recent activity")
    tl = build_governance_timeline(reg).head(12)
    if tl.empty:
        st.caption("No governance timeline events yet.")
    else:
        st.dataframe(tl, use_container_width=True, hide_index=True)

    st.markdown("---")
