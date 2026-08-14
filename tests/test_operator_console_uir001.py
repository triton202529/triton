"""
UIR-001 / UIR-001A / UIR-001B — Operator Console presentation contracts.

Presentation-only. Uses source inspection and fixtures.
Does not start/stop runtime, call broker, create approvals, or mutate evidence.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from dashboard.operator_console import (
    DEFAULT_OPERATOR_PAGE,
    OPERATOR_PAGES,
    OPERATOR_RENDERERS,
    render_operations_page,
    render_reports_page,
)
from dashboard import operator_ui as oui

ROOT = Path(__file__).resolve().parents[1]


def test_operator_console_default_and_seven_page_order():
    assert DEFAULT_OPERATOR_PAGE == "Operations"
    assert OPERATOR_PAGES == [
        "Operations",
        "Activity",
        "Portfolio",
        "Reports",
        "Orders",
        "Risk & Alerts",
        "Diagnostics",
    ]
    assert list(OPERATOR_RENDERERS.keys()) == OPERATOR_PAGES
    assert "Reports" in OPERATOR_RENDERERS


def test_reports_renderer_delegates_to_uir002_engine():
    src = inspect.getsource(render_reports_page)
    assert "render_operator_reports_page" in src
    assert "Reports not available yet" not in src
    from dashboard import operator_reports as oreports

    page_src = inspect.getsource(oreports.render_operator_reports_page)
    assert "Daily" in page_src or "PERIOD_TYPES" in page_src
    banned = ("total_return", "sharpe", "cagr", "pnl_pct", "fake_return")
    lower = page_src.lower()
    for token in banned:
        assert token not in lower


def test_operations_reuses_exact_start_controls():
    import dashboard.operator_console as oc

    src = inspect.getsource(render_operations_page)
    assert "_render_start_controls(" in src
    # Controls must appear before Attention / recovery in the Operations body.
    assert src.index("_render_start_controls(") < src.index("_render_runtime_recovery_guidance()")
    # UIR-003C: pass the same evidence snapshot used by Operations cards.
    assert "status_snapshot=status" in src
    controls = inspect.getsource(oc._render_start_controls)
    stop_col = inspect.getsource(oc._render_stop_control_column)
    restart_col = inspect.getsource(oc._render_restart_control_column)
    assert "execute_start_triton(" in controls
    assert "execute_stop_triton(" in stop_col
    assert "execute_restart_triton(" in restart_col
    assert 'key="triton_ctrl_start"' in controls
    assert 'key="triton_ctrl_stop"' in controls
    assert 'key="triton_ctrl_restart"' in controls


def test_operations_page_load_does_not_submit_controls():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    load = src[src.index("def render_operations_page") : src.index("_render_start_controls(")]
    assert "execute_start_triton" not in load
    assert "execute_stop_triton" not in load
    assert "execute_restart_triton" not in load


def test_html_helpers_escape_values():
    assert oui.escape("<script>") == "&lt;script&gt;"
    assert "RUNNING" in oui.status_badge("RUNNING", tone="ok")
    assert "&lt;" in oui.status_badge("<x>", tone="muted")


def test_unknown_and_stale_not_mapped_healthy():
    assert oui.tone_for_status("UNKNOWN") == "muted"
    assert oui.tone_for_status("STALE") == "warn"
    assert oui.tone_for_status("STOPPED") != "ok"
    assert oui.tone_for_status("FULLY_STOPPED") != "ok"
    assert oui.tone_for_status("PIPELINE_DEGRADED") == "warn"
    assert oui.tone_for_status("CLOSED") == "muted"


def test_live_execution_blocked_never_maps_to_green():
    for label in ("BLOCKED", "LIVE EXECUTION BLOCKED", "LIVE:BLOCKED"):
        tone = oui.tone_for_status(label)
        assert tone == "crit", label
        assert tone != "ok", label
        assert oui.tone_for_live_execution(label) == "crit"
        badge = oui.status_badge(
            label if label != "LIVE EXECUTION BLOCKED" else "BLOCKED", tone=tone
        )
        assert "triton-badge crit" in badge
        assert "triton-badge ok" not in badge
    ops = inspect.getsource(render_operations_page)
    # UIR-001B: full-width banner removed; permanent Live card remains the safety surface.
    assert "render_live_execution_blocked_banner" not in ops
    assert 'st.success("LIVE EXECUTION BLOCKED")' not in ops
    assert "tone_for_live_execution(live_value)" in ops
    assert "Required safety lock" in ops
    assert '"Live execution"' in ops


def test_process_card_semantic_state_mapping():
    assert oui.tone_for_process_status("RUNNING") == "ok"
    assert oui.tone_for_process_status("STOPPED") == "crit"
    assert oui.tone_for_process_status("NOT RUNNING") == "crit"
    assert oui.tone_for_process_status("MISSING") == "crit"
    assert oui.tone_for_process_status("STARTING") == "warn"
    assert oui.tone_for_process_status("STOPPING") == "warn"
    assert oui.tone_for_process_status("DEGRADED") == "warn"
    assert oui.tone_for_process_status("UNKNOWN") == "muted"
    assert oui.tone_for_process_status(None) == "muted"


def test_loop_process_card_preserves_label_and_stale_warning():
    running = oui.loop_process_card_state(process_running=True, heartbeat_class="FRESH")
    assert running["label"] == "RUNNING"
    assert running["display"] == "RUNNING · 1 / 1 INSTANCE"
    assert running["tone"] == "ok"
    assert running["stale_warning"] is None

    # UIR-003C: exactly one process with stale heartbeat presents as STALE, not healthy RUNNING.
    stale = oui.loop_process_card_state(process_running=True, heartbeat_class="STALE")
    assert stale["label"] == "STALE"
    assert stale["display"] == "STALE · 1 / 1 INSTANCE"
    assert stale["tone"] == "warn"
    assert stale["stale_warning"] == "Heartbeat STALE"

    stopped = oui.loop_process_card_state(process_running=False)
    assert stopped["label"] == "STOPPED"
    assert stopped["display"] == "STOPPED · 0 / 1 INSTANCE"
    assert stopped["tone"] == "crit"

    unknown = oui.loop_process_card_state(process_running=None)
    assert unknown["label"] == "UNKNOWN"
    assert "COUNT UNAVAILABLE" in unknown["display"]
    assert unknown["tone"] == "muted"

    # Exact provided labels are not reinterpreted/merged.
    for label, tone in (
        ("STOPPED", "crit"),
        ("MISSING", "crit"),
        ("STARTING", "warn"),
        ("STOPPING", "warn"),
        ("DEGRADED", "warn"),
    ):
        card = oui.loop_process_card_state(process_running=None, status_label=label)
        assert card["label"] == label
        assert card["tone"] == tone


def test_operations_safety_strip_has_six_cards_exact_order():
    import re

    src = inspect.getsource(render_operations_page)
    strip = src.split("oui.render_safety_strip(", 1)[1].split("\n    )\n", 1)[0]
    labels = re.findall(r'"label":\s*"([^"]+)"', strip)
    assert labels == [
        "Live execution",
        "Paper authorization",
        "Market",
        "Continuous loop",
        "Watchdog",
        "Portfolio data freshness",
    ]
    # Generic Runtime card removed from the strip.
    assert '{"label": "Runtime"' not in src
    runtime_panel = src.split('oui.render_compact_panel(\n            "Runtime"', 1)[1].split(
        "with s2:", 1
    )[0]
    assert "Continuous loop" not in runtime_panel
    assert "Overall runtime" in runtime_panel
    assert "Pipeline heartbeat" in runtime_panel
    assert "Next expected cycle" in runtime_panel
    assert "Runtime heartbeat" not in runtime_panel


def test_runtime_summary_card_stacked_layout_preserves_full_values():
    """Runtime summary uses stacked rows and never truncates authoritative values."""
    import re

    src = inspect.getsource(render_operations_page)
    runtime_panel = src.split('oui.render_compact_panel(\n            "Runtime"', 1)[1].split(
        "with s2:", 1
    )[0]
    labels = re.findall(r'"label":\s*"([^"]+)"', runtime_panel)
    assert labels == [
        "Overall runtime",
        "Trading cycle",
        "Session context",
        "Pipeline heartbeat",
        "Next expected cycle",
    ]
    assert "overall" in runtime_panel
    assert "session_phase" in runtime_panel
    assert 'pipeline_hb["display"]' in runtime_panel
    assert "next_expected" in runtime_panel
    # Values are passed through unabbreviated — no slicing / truncation helpers.
    assert "[: " not in runtime_panel
    assert ".truncate" not in runtime_panel.lower()

    panel_src = inspect.getsource(oui.render_compact_panel)
    assert "triton-kv" in panel_src
    assert "escape(" in panel_src
    # Full authoritative strings must survive rendering unchanged.
    html = oui.render_compact_panel.__doc__ or ""
    assert "no truncation" in html.lower() or "stacked" in html.lower()

    theme = inspect.getsource(oui.inject_operator_theme)
    assert "flex-direction: column" in theme
    assert "overflow-wrap: anywhere" in theme
    assert "height: auto" in theme
    # No fragile Streamlit-generated class selectors for the summary cards.
    assert "st-emotion" not in theme.split(".triton-panel", 1)[1].split(".triton-badge", 1)[0]

    # Direct HTML contract: long statuses remain intact.
    from unittest.mock import patch

    captured: list[str] = []

    def _capture(markdown, *args, **kwargs):
        captured.append(str(markdown))

    with patch.object(oui.st, "markdown", side_effect=_capture):
        oui.render_compact_panel(
            "Runtime",
            [
                {"label": "Overall runtime", "value": "FULLY_STOPPED"},
                {
                    "label": "Pipeline heartbeat",
                    "value": "STALE (1396 minutes)",
                },
                {
                    "label": "Next expected cycle",
                    "value": "2026-08-08T20:15:00+00:00",
                },
            ],
        )
    assert captured, "expected compact panel markdown"
    out = captured[0]
    assert "FULLY_STOPPED" in out
    assert "STALE (1396 minutes)" in out
    assert "2026-08-08T20:15:00+00:00" in out
    assert "Overall runtime" in out
    assert "Pipeline heartbeat" in out
    assert "Next expected cycle" in out
    assert out.index("Overall runtime") < out.index("FULLY_STOPPED")
    assert 'class="k"' in out and 'class="v"' in out


def test_market_closed_is_neutral_not_failure():
    assert oui.tone_for_market("CLOSED") == "muted"
    assert oui.tone_for_market("CLOSED") != "crit"
    assert oui.tone_for_market("CLOSED") != "ok"
    assert oui.tone_for_market("OPEN") == "ok"
    assert oui.tone_for_market("PREMARKET") == "info"
    assert oui.tone_for_market("AFTER_HOURS") == "info"
    assert oui.tone_for_market("UNKNOWN") == "muted"
    assert oui.tone_for_market(None) == "muted"


def test_operations_compacts_attention_and_removes_full_width_banner():
    src = inspect.getsource(render_operations_page)
    assert "Operator Attention details —" in src
    assert "summarize_attention_items" in src
    assert "expanded=False" in src
    # Right panel must not permanently loop-render every attention card.
    right = src.split("with right:", 1)[1]
    assert "for item in" not in right
    assert "Current action status" in right
    import dashboard.operator_console as oc

    recovery = inspect.getsource(oc._render_runtime_recovery_guidance)
    assert "Runtime recovery details" in recovery
    assert "Recovery evidence summary" in recovery
    assert "Recovery technical evidence" in recovery
    assert "expanded=False" in recovery


def test_pipeline_and_watchdog_heartbeats_are_not_conflated():
    pipeline = oui.present_pipeline_heartbeat_label(freshness_class="FRESH")
    assert pipeline["channel"] == "pipeline"
    assert pipeline["source"] == "heartbeat.json"
    assert pipeline["label"] == "FRESH"

    watchdog = oui.present_watchdog_heartbeat_label(
        {"timestamp": object(), "age_minutes": 1396.0, "path": "watchdog_status.json"}
    )
    assert watchdog["channel"] == "watchdog"
    assert watchdog["label"] == "STALE"
    assert watchdog["age_minutes"] == 1396.0
    assert "1396" in watchdog["display"]
    # Stale watchdog must not produce a generic FRESH runtime label.
    assert watchdog["label"] != "FRESH"
    assert "runtime" not in watchdog["display"].lower()

    # Pipeline FRESH and watchdog STALE can coexist without aggregation.
    assert pipeline["label"] == "FRESH"
    assert watchdog["label"] == "STALE"
    assert pipeline["channel"] != watchdog["channel"]


def test_stale_watchdog_does_not_yield_generic_runtime_fresh_label():
    src = inspect.getsource(render_operations_page)
    assert "Runtime heartbeat" not in src
    assert "present_pipeline_heartbeat_label" in src
    assert "present_watchdog_heartbeat_label" in src
    assert "load_runtime_heartbeat" in src
    # Continuous uses pipeline label; watchdog process card uses watchdog label.
    assert "heartbeat_class=pipeline_hb" in src
    assert "heartbeat_class=watchdog_hb" in src


def test_portfolio_freshness_remains_independently_sourced():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 8, 8, 18, 0, 0, tzinfo=timezone.utc)
    fresh = oui.classify_portfolio_freshness_label(
        last_refresh_time=(now - timedelta(minutes=10)).isoformat(),
        fresh_minutes=90.0,
        now=now,
    )
    stale = oui.classify_portfolio_freshness_label(
        last_refresh_time=(now - timedelta(minutes=200)).isoformat(),
        fresh_minutes=90.0,
        now=now,
    )
    assert fresh == "FRESH"
    assert stale == "STALE"
    # Independent of heartbeat presentation helpers.
    pipeline = oui.present_pipeline_heartbeat_label(freshness_class="STALE")
    watchdog = oui.present_watchdog_heartbeat_label(
        {"timestamp": now, "age_minutes": 5.0, "path": "watchdog_status.json"}
    )
    assert fresh == "FRESH"
    assert pipeline["label"] == "STALE"
    assert watchdog["label"] == "FRESH"


def test_obsolete_start_guidance_not_shown_when_start_allowed():
    item = {
        "severity": "HIGH",
        "title": "Continuous loop missing",
        "explanation": "No continuous-loop process matching run_continuous_loop.ps1 was detected.",
        "recommended_action": (
            "Use approved runtime start scripts or Task Scheduler backup; "
            "dashboard Start is not available yet."
        ),
        "evidence_source": "services.runtime_process_detector",
    }
    presented = oui.present_attention_recommendation(
        item, start_decision="START_ALLOWED", start_button_enabled=True
    )
    assert "dashboard Start is not available yet" not in presented["current_action"]
    assert presented["historical_action"] is not None
    assert "START_ALLOWED" in presented["current_action"]
    # When Start is refused, still do not present obsolete "not available yet" as current.
    refused = oui.present_attention_recommendation(
        item, start_decision="START_REFUSED_ALREADY_RUNNING", start_button_enabled=False
    )
    assert "dashboard Start is not available yet" not in refused["current_action"].lower()
    assert refused["historical_action"] is not None


def test_portfolio_freshness_separate_from_unknown():
    assert oui.classify_portfolio_freshness_label(last_refresh_time=None) == "UNKNOWN"
    assert oui.classify_portfolio_freshness_label(last_refresh_time="") == "UNKNOWN"
    assert oui.classify_portfolio_freshness_label(last_refresh_time="not-a-date") == "UNKNOWN"


def test_layout_uses_available_width_not_narrow_cap():
    theme = inspect.getsource(oui.inject_operator_theme)
    assert "max-width: min(100%, 1920px)" in theme
    assert "max-width: 1440px" not in theme


def test_technical_json_collapsed_by_default():
    src = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    ops = src[
        src.index("def render_operations_page") : src.index("def _render_runtime_recovery_guidance")
    ]
    assert 'st.expander("Technical evidence", expanded=False)' in ops
    coop = src[
        src.index("def _render_cooperative_shutdown_status") : src.index(
            "def render_operations_page"
        )
    ]
    assert 'st.expander("Request / guard JSON", expanded=False)' in coop
    assert 'st.expander("Acknowledgements / progress JSON", expanded=False)' in coop
    assert "st.json(" in coop  # evidence still present


def test_presentation_module_has_no_controller_calls():
    tree = ast.parse((ROOT / "dashboard" / "operator_ui.py").read_text(encoding="utf-8"))
    banned = {
        "execute_start_triton",
        "execute_stop_triton",
        "execute_restart_triton",
        "issue_stop_preflight",
        "issue_restart_preflight",
        "AlpacaBroker",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            assert name not in banned


def test_view_results_preserves_mode_switch_and_legacy():
    src = (ROOT / "view_results.py").read_text(encoding="utf-8", errors="replace")
    assert "PAGE_REGISTRY" in src and "SECTIONS" in src
    assert "cc_console_mode_radio" in src
    assert "dashboard_nav_mode" in src
    assert "Advanced / Legacy Dashboard" in src
    assert "OPERATOR_RENDERERS" in src
    assert '"Reports"' in src
    assert "list(OPERATOR_PAGES)" in src
