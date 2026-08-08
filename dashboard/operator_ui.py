"""
operator_ui.py
Phase UIR-001 — presentation-only helpers for the Operator Console.

No execution, eligibility, governance, broker, or artifact mutation logic.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

import streamlit as st

# Presentation tokens (visual only — no operational meaning attached).
TOKENS = {
    "shell": "#0B0F14",
    "panel": "#11171F",
    "panel_2": "#161D26",
    "border": "#26313D",
    "text": "#E7ECF3",
    "muted": "#95A2B2",
    "blue": "#4EA1FF",
    "green": "#39C98A",
    "amber": "#F2B84B",
    "red": "#EF6461",
}


def escape(value: Any) -> str:
    """HTML-escape artifact / status values before unsafe_allow_html use."""
    return html.escape("" if value is None else str(value), quote=True)


def format_timestamp(value: Any, *, fallback: str = "UNKNOWN") -> str:
    """Best-effort display formatting; never invents a clock reading."""
    if value is None or value == "":
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return text


def inject_operator_theme() -> None:
    """Inject stable, low-specificity theme rules for the operator shell."""
    # Always refresh theme CSS so UIR repairs apply without stale session latch.
    t = TOKENS
    st.markdown(
        f"""
<style>
  .stApp {{
    background: {t["shell"]};
    color: {t["text"]};
  }}
  section[data-testid="stSidebar"] {{
    background: {t["panel"]};
    border-right: 1px solid {t["border"]};
  }}
  section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {{
    color: {t["text"]};
  }}
  /* Use available desktop width; avoid large empty gutters at 2048px. */
  .block-container {{
    padding-top: 1.0rem;
    padding-bottom: 1.5rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
    max-width: min(100%, 1920px);
  }}
  h1, h2, h3, h4, h5, h6, p, label, span, div {{
    font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  }}
  div[data-testid="stMetric"] {{
    background: {t["panel_2"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 0.35rem 0.55rem !important;
  }}
  div[data-testid="stMetricLabel"] p {{
    color: {t["muted"]} !important;
    font-size: 0.72rem !important;
  }}
  div[data-testid="stMetricValue"] {{
    color: {t["text"]} !important;
    font-size: 1.05rem !important;
  }}
  .triton-rail-brand {{
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: {t["text"]};
    margin: 0 0 0.15rem 0;
  }}
  .triton-rail-sub {{
    color: {t["muted"]};
    font-size: 0.78rem;
    margin: 0 0 0.65rem 0;
  }}
  .triton-cmd-header {{
    background: {t["panel"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 0.65rem 0.85rem;
    margin: 0 0 0.65rem 0;
  }}
  .triton-cmd-title {{
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0;
    color: {t["text"]};
  }}
  .triton-cmd-meta {{
    color: {t["muted"]};
    font-size: 0.82rem;
    margin-top: 0.2rem;
  }}
  .triton-safety-strip {{
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.4rem;
    margin: 0 0 0.55rem 0;
  }}
  .triton-safety-cell {{
    background: {t["panel"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 0.45rem 0.55rem;
  }}
  .triton-safety-cell.ok {{
    background: rgba(57, 201, 138, 0.08);
    border-color: {t["green"]};
  }}
  .triton-safety-cell.warn {{
    background: rgba(242, 184, 75, 0.10);
    border-color: {t["amber"]};
  }}
  .triton-safety-cell.crit {{
    background: rgba(239, 100, 97, 0.10);
    border-color: {t["red"]};
  }}
  .triton-safety-cell.muted {{
    background: {t["panel"]};
    border-color: {t["border"]};
  }}
  .triton-live-blocked-banner {{
    background: rgba(239, 100, 97, 0.10);
    border: 1px solid {t["red"]};
    border-radius: 4px;
    padding: 0.55rem 0.75rem;
    margin: 0 0 0.75rem 0;
    color: {t["text"]};
  }}
  .triton-live-blocked-banner .title {{
    color: {t["red"]};
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
  }}
  .triton-live-blocked-banner .note {{
    color: {t["muted"]};
    font-size: 0.78rem;
    margin-top: 0.2rem;
  }}
  .triton-safety-label {{
    color: {t["muted"]};
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .triton-safety-value {{
    font-size: 0.92rem;
    font-weight: 650;
    margin-top: 0.15rem;
    color: {t["text"]};
  }}
  .triton-safety-note {{
    margin-top: 0.25rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: {t["muted"]};
  }}
  .triton-safety-note.warn {{
    color: {t["amber"]};
    font-weight: 650;
  }}
  .triton-panel {{
    background: {t["panel"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 0.55rem 0.7rem;
    margin: 0 0 0.55rem 0;
    min-width: 0;
    max-width: 100%;
    overflow: visible;
    height: auto;
  }}
  .triton-panel-title {{
    font-size: 0.78rem;
    font-weight: 650;
    color: {t["muted"]};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0 0 0.35rem 0;
  }}
  /* Stacked label/value rows — avoids inline collision on long statuses. */
  .triton-kv {{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.08rem;
    font-size: 0.86rem;
    padding: 0.28rem 0;
    border-bottom: 1px solid rgba(38, 49, 61, 0.65);
    min-width: 0;
    max-width: 100%;
  }}
  .triton-kv:last-child {{ border-bottom: 0; padding-bottom: 0; }}
  .triton-kv .k {{
    color: {t["muted"]};
    font-size: 0.72rem;
    line-height: 1.25;
    max-width: 100%;
  }}
  .triton-kv .v {{
    color: {t["text"]};
    font-weight: 600;
    text-align: left;
    line-height: 1.3;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  .triton-badge {{
    display: inline-block;
    border: 1px solid {t["border"]};
    border-radius: 3px;
    padding: 0.08rem 0.4rem;
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.03em;
    background: {t["panel_2"]};
    color: {t["text"]};
  }}
  .triton-badge.ok {{ border-color: {t["green"]}; color: {t["green"]}; }}
  .triton-badge.warn {{ border-color: {t["amber"]}; color: {t["amber"]}; }}
  .triton-badge.crit {{ border-color: {t["red"]}; color: {t["red"]}; }}
  .triton-badge.info {{ border-color: {t["blue"]}; color: {t["blue"]}; }}
  .triton-badge.muted {{ border-color: {t["border"]}; color: {t["muted"]}; }}
  .triton-attention {{
    border: 1px solid {t["border"]};
    border-left-width: 3px;
    border-radius: 4px;
    background: {t["panel"]};
    padding: 0.5rem 0.65rem;
    margin: 0 0 0.45rem 0;
  }}
  .triton-attention.crit {{ border-left-color: {t["red"]}; }}
  .triton-attention.warn {{ border-left-color: {t["amber"]}; }}
  .triton-attention.info {{ border-left-color: {t["blue"]}; }}
  .triton-attention-title {{
    font-weight: 650;
    color: {t["text"]};
    font-size: 0.9rem;
  }}
  .triton-attention-body {{
    color: {t["muted"]};
    font-size: 0.82rem;
    margin-top: 0.2rem;
  }}
  .triton-empty {{
    background: {t["panel"]};
    border: 1px dashed {t["border"]};
    border-radius: 4px;
    padding: 1rem;
    color: {t["muted"]};
  }}
  .triton-ctrl-legend {{
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    font-size: 0.75rem;
    color: {t["muted"]};
    margin: 0 0 0.4rem 0;
  }}
  .triton-ctrl-legend span::before {{
    content: "";
    display: inline-block;
    width: 0.55rem;
    height: 0.55rem;
    margin-right: 0.3rem;
    border-radius: 1px;
    vertical-align: -1px;
  }}
  .triton-ctrl-legend .start::before {{ background: {t["blue"]}; }}
  .triton-ctrl-legend .stop::before {{ background: {t["red"]}; }}
  .triton-ctrl-legend .restart::before {{ background: {t["amber"]}; }}
  .triton-ctrl-legend .unavailable::before {{ background: {t["muted"]}; }}
  /* Stable keyed-container action colors (Streamlit st-key-*). */
  .st-key-triton_ctrl_start button {{
    background-color: rgba(78, 161, 255, 0.16) !important;
    border: 1px solid {t["blue"]} !important;
    color: {t["text"]} !important;
  }}
  .st-key-triton_ctrl_stop button {{
    background-color: rgba(239, 100, 97, 0.14) !important;
    border: 1px solid {t["red"]} !important;
    color: {t["text"]} !important;
  }}
  .st-key-triton_ctrl_restart button {{
    background-color: rgba(242, 184, 75, 0.14) !important;
    border: 1px solid {t["amber"]} !important;
    color: {t["text"]} !important;
  }}
  .st-key-triton_ctrl_start button:disabled,
  .st-key-triton_ctrl_stop button:disabled,
  .st-key-triton_ctrl_restart button:disabled {{
    opacity: 0.42 !important;
    border-color: {t["border"]} !important;
    background-color: {t["panel_2"]} !important;
    color: {t["muted"]} !important;
  }}
  @media (max-width: 1280px) {{
    .triton-safety-strip {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .block-container {{ padding-left: 0.75rem; padding-right: 0.75rem; }}
  }}
  @media (max-width: 900px) {{
    .triton-safety-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  }}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_triton_operator_theme_injected"] = True


def section_header(title: str, caption: str = "") -> None:
    safe_title = escape(title)
    safe_cap = escape(caption) if caption else ""
    cap_html = f'<div class="triton-cmd-meta">{safe_cap}</div>' if safe_cap else ""
    st.markdown(
        f'<div class="triton-panel"><div class="triton-panel-title">{safe_title}</div>{cap_html}</div>',
        unsafe_allow_html=True,
    )


def status_badge(label: str, *, tone: str = "muted") -> str:
    tone_key = tone if tone in ("ok", "warn", "crit", "info", "muted") else "muted"
    return f'<span class="triton-badge {tone_key}">{escape(label)}</span>'


def tone_for_status(value: Any) -> str:
    """Map displayed status text to a visual tone only (no reinterpretation)."""
    s = str(value or "").upper().strip()
    if not s or s in ("UNKNOWN", "MISSING", "N/A", "NONE"):
        return "muted"
    # Market closed is neutral — never a critical runtime failure cue.
    if s in ("CLOSED", "MARKET_CLOSED"):
        return "muted"
    # Live-execution BLOCKED is a required safety lock — always red, never green.
    if s in ("BLOCKED", "LIVE EXECUTION BLOCKED", "LIVE:BLOCKED"):
        return "crit"
    if s in (
        "RUNNING",
        "ARMED",
        "FRESH",
        "CONNECTED",
        "SUCCESS",
        "STARTED",
    ):
        return "ok"
    if s in (
        "STALE",
        "EXPIRED",
        "DEGRADED",
        "WARN",
        "WARNING",
        "AGING",
        "PIPELINE_DEGRADED",
        "PARTIAL",
    ):
        return "warn"
    if s in ("CRITICAL", "NOT BLOCKED", "INVALID", "FAILED", "ERROR"):
        return "crit"
    # Distinct stopped states stay distinct and non-healthy.
    if s in ("STOPPED", "FULLY_STOPPED", "NOT RUNNING", "DISCONNECTED"):
        return "info"
    return "muted"


def tone_for_live_execution(value: Any) -> str:
    """Live-execution display tone: BLOCKED is red; never green."""
    return tone_for_status(value)


def tone_for_market(value: Any) -> str:
    """Market-state tones: CLOSED is neutral, never a failure (red)."""
    s = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    if not s or s in ("UNKNOWN", "N/A", "NONE", "UNAVAILABLE"):
        return "muted"
    if s == "OPEN":
        return "ok"
    if s in ("CLOSED", "MARKET_CLOSED"):
        return "muted"
    if s in ("PREMARKET", "AFTER_HOURS", "AFTERHOURS", "POSTMARKET", "POST_MARKET"):
        return "info"
    return "muted"


def tone_for_process_status(value: Any) -> str:
    """Semantic tones for Continuous loop / Watchdog safety cards only."""
    s = str(value or "").strip().upper()
    if not s or s in ("UNKNOWN", "N/A", "UNAVAILABLE", "NONE"):
        return "muted"
    if s == "RUNNING":
        return "ok"
    if s in ("STOPPED", "NOT RUNNING", "MISSING"):
        return "crit"
    if s in ("STARTING", "STOPPING", "DEGRADED"):
        return "warn"
    return "muted"


def loop_process_card_state(
    *,
    process_running: Optional[bool],
    heartbeat_class: Any = None,
    status_label: Any = None,
) -> Dict[str, Any]:
    """Presentation-only Continuous/Watchdog card fields.

    Preserves the authoritative display label. Does not merge backend
    classifications. RUNNING + STALE heartbeat is never shown as unconditionally healthy.
    """
    if status_label is not None and str(status_label).strip() != "":
        label = str(status_label).strip()
    elif process_running is True:
        label = "RUNNING"
    elif process_running is False:
        label = "NOT RUNNING"
    else:
        label = "UNKNOWN"

    tone = tone_for_process_status(label)
    stale_warning = None
    hb = str(heartbeat_class or "").strip().upper()
    if label.upper() == "RUNNING" and hb == "STALE":
        # Keep authoritative RUNNING label; downgrade visual health + warn.
        tone = "warn"
        stale_warning = "Heartbeat STALE"
    return {
        "label": label,
        "tone": tone,
        "stale_warning": stale_warning,
        "heartbeat_class": hb or "UNKNOWN",
    }


# Matches the existing Operations banner threshold in view_results.render_heartbeat_freshness_banner.
WATCHDOG_HEARTBEAT_STALE_MINUTES = 60.0


def present_pipeline_heartbeat_label(*, freshness_class: Any) -> Dict[str, Any]:
    """Presentation-only: pipeline heartbeat.json class (never a watchdog aggregate)."""
    label = display_or_unknown(freshness_class)
    return {
        "channel": "pipeline",
        "source": "heartbeat.json",
        "label": label,
        "tone": tone_for_status(label),
        "age_minutes": None,
        "display": label,
    }


def present_watchdog_heartbeat_label(
    runtime_hb: Mapping[str, Any],
    *,
    stale_after_minutes: float = WATCHDOG_HEARTBEAT_STALE_MINUTES,
) -> Dict[str, Any]:
    """Presentation-only: watchdog_status.json heartbeat age → FRESH/STALE/UNKNOWN.

    Does not invent an aggregate runtime heartbeat. Does not alter backend age calculation.
    """
    age = runtime_hb.get("age_minutes")
    ts = runtime_hb.get("timestamp")
    source = str(runtime_hb.get("path") or "watchdog_status.json")
    if ts is None or age is None:
        return {
            "channel": "watchdog",
            "source": source,
            "label": "UNKNOWN",
            "tone": "muted",
            "age_minutes": None,
            "display": "UNKNOWN",
            "age_note": None,
        }
    try:
        age_f = float(age)
    except (TypeError, ValueError):
        return {
            "channel": "watchdog",
            "source": source,
            "label": "UNKNOWN",
            "tone": "muted",
            "age_minutes": None,
            "display": "UNKNOWN",
            "age_note": None,
        }
    label = "STALE" if age_f > float(stale_after_minutes) else "FRESH"
    age_note = f"{age_f:.0f} min old"
    return {
        "channel": "watchdog",
        "source": source,
        "label": label,
        "tone": tone_for_status(label),
        "age_minutes": age_f,
        "display": f"{label} ({age_note})",
        "age_note": age_note,
    }


def classify_portfolio_freshness_label(
    *,
    last_refresh_time: Any,
    fresh_minutes: float = 90.0,
    now: Optional[datetime] = None,
) -> str:
    """Presentation label for portfolio snapshot freshness (not heartbeat)."""
    if last_refresh_time is None or last_refresh_time == "":
        return "UNKNOWN"
    try:
        text = str(last_refresh_time).replace("Z", "+00:00")
        ts = datetime.fromisoformat(text)
        if ts.tzinfo is None:
            from datetime import timezone

            ts = ts.replace(tzinfo=timezone.utc)
        clock = now
        if clock is None:
            from datetime import timezone

            clock = datetime.now(timezone.utc)
        age_min = (clock - ts).total_seconds() / 60.0
        if age_min < 0:
            return "UNKNOWN"
        if age_min <= float(fresh_minutes):
            return "FRESH"
        return "STALE"
    except Exception:
        return "UNKNOWN"


_OBSOLETE_START_GUIDANCE = "dashboard start is not available yet"


def present_attention_recommendation(
    item: Mapping[str, Any],
    *,
    start_decision: Any = None,
    start_button_enabled: Any = None,
) -> Dict[str, Any]:
    """Presentation-only: never show obsolete Start guidance as current action."""
    raw_action = str(item.get("recommended_action") or "").strip()
    decision = str(start_decision or "").strip()
    enabled = bool(start_button_enabled)
    historical = None
    action = raw_action
    if _OBSOLETE_START_GUIDANCE in raw_action.lower():
        historical = raw_action
        if enabled or decision == "START_ALLOWED":
            action = (
                f"Current Start eligibility is {decision or 'START_ALLOWED'}. "
                "Use the protected Start control after confirmation."
            )
        elif decision:
            action = (
                f"Current Start eligibility is {decision}. "
                "Follow the protected Start control state; do not invent an alternate start path."
            )
        else:
            action = (
                "Review protected Start eligibility in Runtime controls. "
                "Do not treat historical guidance as current."
            )
    return {
        "current_action": action,
        "historical_action": historical,
        "start_decision": decision or "UNKNOWN",
        "start_button_enabled": enabled,
    }


def render_safety_strip(cells: Sequence[Mapping[str, Any]]) -> None:
    parts: List[str] = []
    for cell in cells:
        label = escape(cell.get("label") or "")
        value = escape(cell.get("value") or "UNKNOWN")
        tone = str(cell.get("tone") or tone_for_status(cell.get("value")))
        if tone not in ("ok", "warn", "crit", "info", "muted"):
            tone = "muted"
        note = cell.get("note")
        note_cls = "triton-safety-note"
        if note and "STALE" in str(note).upper():
            note_cls += " warn"
        note_html = f'<div class="{note_cls}">{escape(note)}</div>' if note else ""
        parts.append(
            f'<div class="triton-safety-cell {tone}">'
            f'<div class="triton-safety-label">{label}</div>'
            f'<div class="triton-safety-value">{status_badge(value, tone=tone)}</div>'
            f"{note_html}"
            f"</div>"
        )
    st.markdown(
        f'<div class="triton-safety-strip">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_compact_panel(title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Compact summary card with stacked label/value rows (no truncation)."""
    body = []
    for row in rows:
        k = escape(row.get("label") or "")
        raw = row.get("value")
        v = escape(raw if raw is not None else "UNKNOWN")
        body.append(
            f'<div class="triton-kv">'
            f'<span class="k">{k}</span>'
            f'<span class="v">{v}</span>'
            f"</div>"
        )
    st.markdown(
        f'<div class="triton-panel"><div class="triton-panel-title">{escape(title)}</div>'
        f'{"".join(body)}</div>',
        unsafe_allow_html=True,
    )


def render_command_header(
    *,
    title: str,
    context: str,
    market_state: str,
    evidence_ts: str,
) -> None:
    st.markdown(
        f"""
<div class="triton-cmd-header">
  <div class="triton-cmd-title">{escape(title)}</div>
  <div class="triton-cmd-meta">
    {escape(context)} · Market: <strong>{escape(market_state)}</strong>
    · Evidence: <strong>{escape(evidence_ts)}</strong>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str, reserved: Optional[Sequence[str]] = None) -> None:
    reserved_html = ""
    if reserved:
        items = "".join(f"<li>{escape(x)}</li>" for x in reserved)
        reserved_html = (
            f'<div style="margin-top:0.65rem;color:{TOKENS["muted"]};font-size:0.82rem;">'
            f"<div>Reserved reporting periods:</div><ul>{items}</ul></div>"
        )
    st.markdown(
        f'<div class="triton-empty"><div style="color:{TOKENS["text"]};font-weight:650;">'
        f"{escape(title)}</div>"
        f'<div style="margin-top:0.35rem;">{escape(body)}</div>{reserved_html}</div>',
        unsafe_allow_html=True,
    )


def render_attention_item(
    item: Mapping[str, Any],
    *,
    start_decision: Any = None,
    start_button_enabled: Any = None,
) -> None:
    sev = str(item.get("severity") or "INFO").upper()
    if sev in ("CRITICAL", "HIGH", "FATAL"):
        css = "crit"
        tone = "crit"
    elif sev in ("WARN", "WARNING", "MEDIUM"):
        css = "warn"
        tone = "warn"
    else:
        css = "info"
        tone = "info"
    title = escape(item.get("title") or "Attention item")
    explanation = escape(item.get("explanation") or "")
    presented = present_attention_recommendation(
        item,
        start_decision=start_decision,
        start_button_enabled=start_button_enabled,
    )
    action = escape(presented.get("current_action") or "Review evidence before acting.")
    st.markdown(
        f'<div class="triton-attention {css}">'
        f'<div class="triton-attention-title">{status_badge(sev, tone=tone)} {title}</div>'
        f'<div class="triton-attention-body">{explanation}</div>'
        f'<div class="triton-attention-body">Recommended safe action: {action}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Supporting evidence", expanded=False):
        st.caption(f"Evidence source: {item.get('evidence_source') or 'n/a'}")
        if presented.get("historical_action"):
            st.caption("Historical / obsolete recommendation (not current):")
            st.write(presented.get("historical_action"))
        st.caption(
            f"Authoritative Start eligibility (presentation overlay): "
            f"{presented.get('start_decision')} · button_enabled={presented.get('start_button_enabled')}"
        )
        st.json(
            {
                "severity": item.get("severity"),
                "title": item.get("title"),
                "explanation": item.get("explanation"),
                "recommended_action_raw": item.get("recommended_action"),
                "recommended_action_presented": presented.get("current_action"),
                "historical_recommendation": presented.get("historical_action"),
                "evidence_source": item.get("evidence_source"),
            }
        )


def display_or_unknown(value: Any) -> str:
    if value is None or value == "":
        return "UNKNOWN"
    return str(value)


def runtime_mode_label(status: Mapping[str, Any]) -> str:
    overall = display_or_unknown(status.get("overall_status"))
    return f"Runtime: {overall}"


def compact_system_status_lines(status: Mapping[str, Any]) -> Dict[str, str]:
    live = status.get("live_execution") or {}
    paper = status.get("paper_arm") or {}
    freshness = status.get("data_freshness") or {}
    return {
        "overall": display_or_unknown(status.get("overall_status")),
        "live": "BLOCKED" if live.get("blocked") else "NOT BLOCKED",
        "paper": display_or_unknown(paper.get("display_status")),
        "freshness": display_or_unknown(freshness.get("class")),
    }


def render_live_execution_blocked_banner() -> None:
    """Deprecated full-width banner (UIR-001B). Prefer the Live execution status card."""
    st.markdown(
        '<div class="triton-live-blocked-banner">'
        f'<div class="title">{escape("LIVE EXECUTION BLOCKED")}</div>'
        f'<div class="note">{escape("Required safety lock — live execution is unavailable.")}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def summarize_attention_items(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compact Operator Attention summary for the right utility panel."""
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "WARN": 0, "INFO": 0, "OTHER": 0}

    def _rank(item: Mapping[str, Any]) -> int:
        s = str(item.get("severity") or "").upper()
        if s in ("CRITICAL", "FATAL"):
            return 0
        if s == "HIGH":
            return 1
        if s in ("WARN", "WARNING", "MEDIUM"):
            return 2
        return 3

    ordered = sorted(list(items or []), key=_rank)
    for item in ordered:
        s = str(item.get("severity") or "INFO").upper()
        if s in ("CRITICAL", "FATAL"):
            counts["CRITICAL"] += 1
        elif s == "HIGH":
            counts["HIGH"] += 1
        elif s == "MEDIUM":
            counts["MEDIUM"] += 1
        elif s in ("WARN", "WARNING"):
            counts["WARN"] += 1
        elif s == "INFO":
            counts["INFO"] += 1
        else:
            counts["OTHER"] += 1
    top = ordered[0] if ordered else None
    return {
        "total": len(ordered),
        "counts": counts,
        "top_item": top,
        "ordered": ordered,
    }
