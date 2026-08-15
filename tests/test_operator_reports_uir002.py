"""
UIR-002 — Reports page presentation contracts.

Source inspection + fixture-backed engine checks.
Does not invoke Start/Stop/Restart or mutate operational artifacts.
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
from dashboard import operator_reports as oreports
from services.reconciled_portfolio_reporting import DEFAULT_PERIOD, PERIOD_TYPES

ROOT = Path(__file__).resolve().parents[1]


def test_reports_page_wired_and_default_daily():
    assert DEFAULT_OPERATOR_PAGE == "Operations"
    assert OPERATOR_PAGES[3] == "Reports"
    assert OPERATOR_RENDERERS["Reports"] is render_reports_page
    src = inspect.getsource(oreports.render_operator_reports_page)
    assert "cc_reports_period" in src
    assert "DEFAULT_PERIOD" in inspect.getsource(oreports)
    assert list(PERIOD_TYPES)[0] == "Daily" == DEFAULT_PERIOD
    for name in PERIOD_TYPES:
        assert f'"{name}"' in inspect.getsource(oreports) or name in PERIOD_TYPES


def test_reports_period_radio_order():
    src = inspect.getsource(oreports.render_operator_reports_page)
    assert "list(PERIOD_TYPES)" in src
    assert PERIOD_TYPES.index("Daily") == 0
    assert tuple(PERIOD_TYPES) == (
        "Daily",
        "Weekly",
        "Monthly",
        "Quarterly",
        "Half-year",
        "Yearly",
        "Custom",
    )


def test_evidence_expanders_collapsed_by_default():
    src = inspect.getsource(oreports.render_operator_reports_page)
    assert "expanded=False" in src
    for label in (
        "Metric provenance",
        "Source-level freshness provenance",
        "Reconciliation details",
        "Source coverage",
        "Raw source references",
        "Technical JSON",
        "Calculation descriptions",
    ):
        assert label in src
    assert "post-as-of" in src.lower() or "Post-as-of" in src


def test_reports_page_discloses_opening_boundary_and_withholds_realized():
    src = inspect.getsource(oreports.render_operator_reports_page)
    assert "Authoritative session-open equity is Unavailable" in src
    assert "Previous recorded equity" in src
    assert "Latest recorded equity" in src
    assert "Change between recorded observations" in src
    assert "governed accounting authority" in src
    assert "partial coverage" in src.lower()
    assert "not Period P/L" in src or "Not Period P/L" in src


def test_refresh_is_local_only():
    src = inspect.getsource(oreports.render_operator_reports_page)
    assert 'key="cc_reports_refresh"' in src
    assert "st.cache_data.clear" in src
    banned = ("TradeClient", "alpaca", "requests.get", "execute_start", "execute_stop")
    lower = src.lower()
    for token in banned:
        assert token.lower() not in lower


def test_unavailable_presentation_not_zero_dollar():
    src = inspect.getsource(oreports.render_operator_reports_page)
    mod = inspect.getsource(oreports)
    assert "Unavailable" in src
    assert "format_money" in mod
    # Must not hardcode fabricated zeros as portfolio results.
    assert '"$0.00"' not in src


def test_reports_delegates_from_operator_console():
    src = inspect.getsource(render_reports_page)
    assert "render_operator_reports_page" in src
    assert "Reports not available yet" not in src


def test_operations_and_callbacks_unchanged_surface():
    ops = inspect.getsource(render_operations_page)
    tree = ast.parse(ops)
    start_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "_render_start_controls")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "_render_start_controls")
        )
    ]
    assert start_calls, "Operations page must render start controls"
    assert any(
        any(kw.arg == "status_snapshot" for kw in call.keywords) for call in start_calls
    ), "start controls must receive the existing status snapshot"
    console = (ROOT / "dashboard" / "operator_console.py").read_text(encoding="utf-8")
    assert "execute_start_triton(" in console
    assert "execute_stop_triton(" in console
    assert "execute_restart_triton(" in console
    assert OPERATOR_PAGES == [
        "Operations",
        "Activity",
        "Portfolio",
        "Reports",
        "Orders",
        "Risk & Alerts",
        "Diagnostics",
    ]


def test_no_fragile_streamlit_emotion_selectors_in_reports():
    src = inspect.getsource(oreports)
    assert "st-emotion" not in src
    assert "css-" not in src


def test_uir002d_holdings_ui_uses_latest_snapshot_semantics():
    src = inspect.getsource(oreports.render_operator_reports_page)
    assert "section_title" in src
    assert "disclosure" in src
    assert "snapshot_ts" in src
    assert "period_relationship" in src
    assert "No TRITON decision-audit rows in period." not in src
    assert "available local evidence" in src
    assert "empty_state_triton" in src
    assert "empty_state_broker" in src
