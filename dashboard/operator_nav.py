"""Operator navigation groups for UI-A. Presentation only."""

from __future__ import annotations

from typing import Dict, List, Tuple

from dashboard.operator_console import OPERATOR_PAGES

# Overview is the landing page. Existing operator pages remain reachable.
NAV_GROUPS: List[Tuple[str, List[str]]] = [
    ("Overview", ["Overview"]),
    ("Portfolio", ["Portfolio"]),
    ("Trading", ["Orders", "Activity"]),
    ("Performance", ["Reports"]),
    ("Operations", ["Operations", "Risk & Alerts"]),
    ("System", ["Diagnostics"]),
]

UI_A_PAGES: List[str] = [page for _, pages in NAV_GROUPS for page in pages]
DEFAULT_UI_A_PAGE = "Overview"

# Preserve the contract list for tests that still import OPERATOR_PAGES.
CONTRACT_OPERATOR_PAGES = list(OPERATOR_PAGES)


def nav_group_for(page: str) -> str:
    for group, pages in NAV_GROUPS:
        if page in pages:
            return group
    return "Overview"


def all_destinations() -> List[str]:
    return list(UI_A_PAGES)


def page_help() -> Dict[str, str]:
    return {
        "Overview": "Runtime, paper mode, and what needs attention.",
        "Portfolio": "Value, history, and current holdings.",
        "Orders": "Unified paper-order ledger.",
        "Activity": "What happened, in order.",
        "Reports": "Period performance from local evidence.",
        "Operations": "Start, stop, and authorization.",
        "Risk & Alerts": "Risk posture and active alerts.",
        "Diagnostics": "Technical evidence and hashes.",
    }
