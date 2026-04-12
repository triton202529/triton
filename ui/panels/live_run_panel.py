# ui/panels/live_run_panel.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import pandas as pd
import streamlit as st

from services.runtime_io import safe_read_json


RUNTIME_VERIFY_PATH = Path("data/runtime/open_orders_verify.json")
RUNTIME_STATE_PATH = Path("data/runtime/runtime_state.json")


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _mins_ago(dt: Optional[datetime]) -> Optional[int]:
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    return int((now - dt).total_seconds() // 60)


def _badge(label: str, value: str, kind: str = "ok"):
    if kind == "ok":
        st.success(f"**{label}:** {value}")
    elif kind == "warn":
        st.warning(f"**{label}:** {value}")
    else:
        st.error(f"**{label}:** {value}")


def render_live_run_panel():
    st.title("🟢 Live Run (Phase 1.5)")
    st.caption("Operations & Safety view — scheduler heartbeat + open orders verification.")

    state = safe_read_json(RUNTIME_STATE_PATH)
    report = safe_read_json(RUNTIME_VERIFY_PATH)

    # ─────────────────────────────────────────
    # SECTION 1: Scheduler / Automation Heartbeat
    # ─────────────────────────────────────────
    st.subheader("Scheduler / Automation Heartbeat")

    if not state:
        st.warning("No runtime_state.json found yet.")
        st.code(
            "Expected: data/runtime/runtime_state.json\n"
            "Write one now:\n"
            '  python -m services.runtime_state --on --mode NORMAL --note "manual heartbeat"'
        )
    else:
        hb_ts = _parse_ts(state.get("ts_utc"))
        hb_mins = _mins_ago(hb_ts)

        enabled = state.get("automation_enabled")
        mode = (state.get("mode") or "UNKNOWN").upper()
        note = state.get("note") or ""
        ident = state.get("identity") or {}
        cycle = state.get("cycle") if isinstance(state.get("cycle"), dict) else {}

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if enabled is True:
                _badge("AUTOMATION", "ON", "ok")
            elif enabled is False:
                _badge("AUTOMATION", "OFF", "warn")
            else:
                _badge("AUTOMATION", "UNKNOWN", "warn")

        with c2:
            kind = "ok"
            if mode in ("LOCKDOWN",):
                kind = "bad"
            elif mode in ("DEFENSIVE",):
                kind = "warn"
            _badge("MODE", mode, kind)

        with c3:
            if hb_mins is None:
                _badge("HEARTBEAT", "Unknown", "warn")
            elif hb_mins <= 2:
                _badge("HEARTBEAT", f"{hb_mins} min ago", "ok")
            elif hb_mins <= 10:
                _badge("HEARTBEAT", f"{hb_mins} min ago", "warn")
            else:
                _badge("HEARTBEAT", f"{hb_mins} min ago", "bad")

        with c4:
            host = ident.get("host") or "—"
            pid = ident.get("pid") or "—"
            st.info(f"**Runner**\n\n- host: `{host}`\n- pid: `{pid}`")

        if note:
            st.caption(f"Note: {note}")

        # Cycle details
        if cycle:
            st.markdown("**Last Cycle**")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Cycle status", str(cycle.get("status") or "—"))
            cc2.metric(
                "Success", str(cycle.get("success")) if cycle.get("success") is not None else "—"
            )
            cc3.metric("Duration (sec)", str(cycle.get("duration_sec") or "—"))
            cc4.metric("Reason", str(cycle.get("reason") or "—"))

    st.markdown("---")

    # ─────────────────────────────────────────
    # SECTION 2: Open Orders Verify Report
    # ─────────────────────────────────────────
    st.subheader("Open Orders Verification")

    if not report:
        st.warning("No open_orders_verify.json found yet.")
        st.code(
            "Expected: data/runtime/open_orders_verify.json\n"
            "Generate it now:\n"
            "  python -m services.runtime_verify --write-report"
        )
        return

    ts = _parse_ts(report.get("ts"))
    mins = _mins_ago(ts)

    status = (report.get("status") or "UNKNOWN").upper()
    summary = report.get("summary") or ""

    counts = report.get("counts") or {}
    issues = report.get("issues") or {}
    policy = report.get("policy") or {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kind = "ok" if status == "OK" else ("warn" if status == "WARN" else "bad")
        _badge("VERIFY STATUS", status, kind)

    with c2:
        if mins is None:
            _badge("LAST VERIFY", "Unknown", "warn")
        elif mins <= 2:
            _badge("LAST VERIFY", f"{mins} min ago", "ok")
        elif mins <= 10:
            _badge("LAST VERIFY", f"{mins} min ago", "warn")
        else:
            _badge("LAST VERIFY", f"{mins} min ago", "bad")

    with c3:
        _badge("SUMMARY", summary or "—", "ok" if status == "OK" else "warn")

    with c4:
        exp_tif = policy.get("expect_tif", "—")
        stale_min = policy.get("stale_minutes", "—")
        stale_only_day = policy.get("stale_only_day", False)
        st.info(
            f"**Policy**\n\n"
            f"- expect_tif: `{exp_tif}`\n"
            f"- stale_minutes: `{stale_min}`\n"
            f"- stale_only_day: `{stale_only_day}`"
        )

    st.subheader("Orders Snapshot")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Open parents", int(counts.get("open_parents", 0)))
    k2.metric("Open legs", int(counts.get("open_legs", 0)))
    k3.metric("Stale", int(counts.get("stale", 0)))
    k4.metric("Dupes (groups)", int(counts.get("dupe_groups", 0)))
    k5.metric("Orphan legs", int(counts.get("orphan_legs", 0)))
    k6.metric("Non-GTC", int(counts.get("non_gtc", 0)))

    st.subheader("Issues Detail")
    tabs = st.tabs(
        [
            "Stale Orders",
            "Duplicates",
            "Non-GTC",
            "Orphan Legs",
            "Broken Parents",
            "Cancel Plan / Cancelled",
        ]
    )

    with tabs[0]:
        stale_list: List[Dict[str, Any]] = issues.get("stale") or []
        if not stale_list:
            st.success("No stale orders in this report.")
        else:
            df = pd.DataFrame(stale_list)
            preferred = [
                "symbol",
                "side",
                "type",
                "time_in_force",
                "order_class",
                "limit_price",
                "stop_price",
                "created_at",
                "id",
                "client_order_id",
            ]
            cols = [c for c in preferred if c in df.columns] + [
                c for c in df.columns if c not in preferred
            ]
            st.dataframe(df[cols], use_container_width=True)

    with tabs[1]:
        dup_groups = issues.get("duplicates") or []
        if not dup_groups:
            st.success("No duplicate order groups in this report.")
        else:
            for g in dup_groups:
                st.warning(
                    f"Fingerprint: `{g.get('fingerprint')}` — {len(g.get('orders', []))} orders"
                )
                df = pd.DataFrame(g.get("orders") or [])
                st.dataframe(df, use_container_width=True)

    with tabs[2]:
        non_gtc = issues.get("non_gtc") or []
        if not non_gtc:
            st.success("No non-GTC orders in this report.")
        else:
            df = pd.DataFrame(non_gtc)
            st.dataframe(df, use_container_width=True)

    with tabs[3]:
        orphan = issues.get("orphan_legs") or []
        if not orphan:
            st.success("No orphan legs detected.")
        else:
            df = pd.DataFrame(orphan)
            st.dataframe(df, use_container_width=True)

    with tabs[4]:
        broken = issues.get("broken_parents") or []
        missing = issues.get("missing_leg_counts") or []
        if not broken and not missing:
            st.success("No broken parents detected (filled parents missing legs).")
        else:
            if broken:
                st.error(f"Broken parents: {len(broken)}")
                st.dataframe(pd.DataFrame(broken), use_container_width=True)
            if missing:
                st.warning("Parents with insufficient leg count after fill:")
                st.dataframe(pd.DataFrame(missing), use_container_width=True)

    with tabs[5]:
        cancel_plan = report.get("cancel_plan") or []
        cancelled = report.get("cancelled") or []
        dry_run = report.get("dry_run", True)

        cA, cB, cC = st.columns(3)
        cA.metric("Cancel plan", len(cancel_plan))
        cB.metric("Cancelled", len(cancelled))
        cC.metric("Dry run", 1 if dry_run else 0)

        if cancel_plan:
            st.write("### Cancel Plan")
            st.dataframe(pd.DataFrame(cancel_plan), use_container_width=True)
        else:
            st.info("No cancels planned in this report.")

        if cancelled:
            st.write("### Cancelled Order IDs")
            st.code("\n".join(cancelled))
