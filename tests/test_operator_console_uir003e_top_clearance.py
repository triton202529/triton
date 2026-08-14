"""UIR-003E follow-up — Streamlit toolbar top clearance (presentation only)."""

from __future__ import annotations

import inspect
import re

from dashboard import operator_ui as oui


def test_operator_theme_uses_single_toolbar_clearance_on_block_container():
    theme = inspect.getsource(oui.inject_operator_theme)
    assert "--triton-toolbar-clearance: 4rem" in theme
    assert "padding-top: var(--triton-toolbar-clearance) !important" in theme
    # Must not reintroduce the short padding that tucked content under the toolbar.
    assert "padding-top: 1.0rem" not in theme
    assert "padding-top: 1rem" not in theme
    # Clearance is container-level — one padding-top declaration for the rule block.
    assert theme.count("padding-top:") == 1


def test_operator_theme_does_not_hide_streamlit_toolbar():
    theme = inspect.getsource(oui.inject_operator_theme)
    forbidden = (
        "stDecoration",
        "toolbarMode",
        "hideToolbar",
        "header[data-testid",
        "display: none",
        "visibility: hidden",
    )
    for token in forbidden:
        assert token not in theme


def test_operator_theme_has_no_pull_under_toolbar_overrides():
    theme = inspect.getsource(oui.inject_operator_theme)
    assert not re.search(r"margin-top:\s*-\d", theme)
    assert "translateY(" not in theme
    assert "position: absolute" not in theme
    # Sticky top must clear the toolbar, not pin to viewport 0.
    assert "top: var(--triton-toolbar-clearance)" in theme
    assert not re.search(r"(?<!margin-)top:\s*0(?:\s|;|$)", theme)


def test_view_results_compact_css_matches_clearance_contract():
    from pathlib import Path

    src = Path("view_results.py").read_text(encoding="utf-8")
    # Early compact CSS block.
    head = src.split("COMPACT UI CSS", 1)[1].split("</style>", 1)[0]
    assert "--triton-toolbar-clearance" in head
    assert "padding-top: var(--triton-toolbar-clearance)" in head
    assert "top:var(--triton-toolbar-clearance)" in head.replace(" ", "")
    sticky_tops = re.findall(r"(?:triton-topwrap|triton-clock)\{[^}]*top:([^;]+);", head.replace("\n", ""))
    assert sticky_tops
    assert all("triton-toolbar-clearance" in t for t in sticky_tops)
