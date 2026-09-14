"""
Command Center single navigation + paper readiness wording tests.

Isolated fixtures, fixed UTC clock. No Streamlit server, no broker calls,
no runtime launch, no artifact mutation on import.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "view_results.py"
CONFIG = ROOT / ".streamlit" / "config.toml"
STATUS_MOD = ROOT / "services" / "operator_console_status.py"
PAGES = ROOT / "pages"

FIXED_NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)

# Landing-page radio order from dashboard/operator_nav.py (Overview is the landing page).
CANONICAL_OPERATOR_PAGES = [
    "Overview",
    "Portfolio",
    "Performance",
    "Orders",
    "Activity",
    "Reports",
    "Operations",
    "Risk & Alerts",
    "Diagnostics",
]

# Legacy Command Center constant in view_results.py (includes Overview; not the radio list).
COMMAND_CENTER_DESTINATIONS = [
    "Overview",
    "Operations",
    "Activity",
    "Portfolio",
    "Reports",
    "Orders",
    "Risk & Alerts",
    "Diagnostics",
    "Advanced Dashboard",
]


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _seed(tmp_path: Path, *, paper: bool = True, arm: str = "EXPIRED") -> Path:
    results = tmp_path / "data" / "results"
    live = tmp_path / "data" / "live"
    results.mkdir(parents=True)
    live.mkdir(parents=True)
    _write_json(
        results / "governance_authorization.json",
        {
            "generated_at": "2026-08-01T22:29:39Z",
            "paper_execution_permitted": paper,
            "live_execution_permitted": False,
            "overall_authorization": False,
        },
    )
    _write_json(
        results / "execution_readiness.json",
        {
            "generated_at": "2026-08-01T22:29:39Z",
            "mode": "PAPER_ONLY",
            "paper_execution_permitted": paper,
            "live_execution_permitted": False,
            "readiness_status": "NOT_READY",
        },
    )
    if arm == "EXPIRED":
        _write_json(
            live / "paper_arm_confirm.json",
            {
                "timestamp": "2026-07-31T13:11:23Z",
                "mode": "ASSISTED",
                "expires_at": "2026-07-31T23:11:23Z",
                "allow_execute": True,
            },
        )
    elif arm == "VALID":
        _write_json(
            live / "paper_arm_confirm.json",
            {
                "timestamp": FIXED_NOW.isoformat(),
                "mode": "ASSISTED",
                "expires_at": (FIXED_NOW + timedelta(hours=1)).isoformat(),
                "allow_execute": True,
            },
        )
    elif arm == "MISSING":
        pass
    elif arm == "MALFORMED":
        (live / "paper_arm_confirm.json").write_text("{not-json", encoding="utf-8")
    return tmp_path


def test_native_multipage_nav_disabled_via_supported_config():
    text = CONFIG.read_text(encoding="utf-8")
    assert "[client]" in text
    assert re.search(r"showSidebarNavigation\s*=\s*false", text)
    # Must not wipe server/browser settings
    assert 'address = "127.0.0.1"' in text
    assert "port = 8501" in text


def test_custom_command_center_nav_present_and_destinations_once():
    nav = (ROOT / "dashboard" / "operator_nav.py").read_text(encoding="utf-8")
    groups = re.search(
        r"^NAV_GROUPS:.*?=\s*\[(.*?)\]\s*^UI_A_PAGES",
        nav,
        re.S | re.M,
    )
    assert groups
    pages: list[str] = []
    for inner in re.findall(r'\("[^"]+",\s*\[(.*?)\]\)', groups.group(1), re.S):
        pages.extend(re.findall(r'"([^"]+)"', inner))
    assert pages == CANONICAL_OPERATOR_PAGES
    assert len(pages) == len(set(pages))
    assert "Overview" in pages and "Performance" in pages
    assert re.search(r'DEFAULT_UI_A_PAGE\s*=\s*"Overview"', nav)

    src = VIEW.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"COMMAND_CENTER_DESTINATIONS:\s*List\[str\]\s*=\s*\[(.*?)\]",
        src,
        re.S,
    )
    assert m
    items = re.findall(r'"([^"]+)"', m.group(1))
    assert items == COMMAND_CENTER_DESTINATIONS
    assert len(items) == len(set(items))
    assert items[0] == "Overview"
    assert "st.radio(" in src
    assert "all_destinations()" in src or "ui_a_pages" in src
    assert '"Destination"' in src or "Destination" in src or '"Page"' in src
    assert "OPERATOR_RENDERERS" in src


def test_page_files_and_routes_preserved():
    assert PAGES.is_dir()
    assert (PAGES / "manual_order_desk.py").is_file()
    text = CONFIG.read_text(encoding="utf-8")
    # Config disables native nav only; it does not rewrite page routes.
    assert "manual_order_desk" not in text
    assert "showSidebarNavigation" in text


def test_sidebar_preserved_no_css_js_hacks():
    src = VIEW.read_text(encoding="utf-8", errors="replace")
    cfg = CONFIG.read_text(encoding="utf-8")
    assert "with st.sidebar" in src or "st.sidebar" in src
    assert "display:none" not in cfg.lower()
    assert "<script" not in cfg.lower()
    idx = src.find("build_paper_readiness_wording")
    assert idx > 0
    wording_region = src[idx : idx + 1200]
    assert "stylesheet" not in wording_region.lower()
    assert "<script" not in wording_region.lower()
    assert "querySelector" not in wording_region


def test_no_iframe_subprocess_second_server_in_wording_path():
    src = VIEW.read_text(encoding="utf-8", errors="replace")
    # Sidebar readiness block around build_paper_readiness_wording
    idx = src.find("build_paper_readiness_wording")
    assert idx > 0
    block = src[idx - 200 : idx + 1200]
    assert "iframe" not in block.lower()
    assert "subprocess" not in block
    assert "8502" not in block
    assert "st_autorefresh" not in block
    assert "AlpacaBroker" not in block


def test_authorization_and_arm_load_from_separate_contracts(tmp_path: Path):
    from services.operator_console_status import build_paper_readiness_wording

    root = _seed(tmp_path, paper=True, arm="EXPIRED")
    prw = build_paper_readiness_wording(root=root, now=FIXED_NOW)
    assert prw["authorization_and_arm_are_distinct"] is True
    assert "governance_authorization" in prw["standing_authorization_source"]
    assert "paper_arm_confirm" in prw["paper_arm_source"]
    assert prw["standing_authorization_status"] == "PERMITTED"
    assert prw["paper_arm_status"] == "EXPIRED"


def test_valid_auth_expired_arm_display_separately(tmp_path: Path):
    from services.operator_console_status import build_paper_readiness_wording

    root = _seed(tmp_path, paper=True, arm="EXPIRED")
    prw = build_paper_readiness_wording(root=root, now=FIXED_NOW)
    assert prw["standing_authorization_label"] == "Standing paper authorization"
    assert prw["paper_arm_label"] == "Paper execution arm"
    assert prw["standing_authorization_status"] == "PERMITTED"
    assert prw["paper_arm_status"] == "EXPIRED"
    assert prw["paper_arm_expires_at"]
    assert "EXPIRED" in prw["paper_arm_status"]


def test_valid_authorization_never_ready_to_trade(tmp_path: Path):
    from services.operator_console_status import build_paper_readiness_wording

    root = _seed(tmp_path, paper=True, arm="VALID")
    prw = build_paper_readiness_wording(root=root, now=FIXED_NOW)
    assert prw["standing_authorization_status"] == "PERMITTED"
    assert prw["ready_to_trade"] is False
    assert prw["order_ready"] is False
    blob = json.dumps(prw)
    assert "Ready to trade" not in blob
    assert "READY_TO_TRADE" not in blob


def test_authorization_alone_never_order_ready(tmp_path: Path):
    from services.operator_console_status import build_paper_readiness_wording

    root = _seed(tmp_path, paper=True, arm="EXPIRED")
    prw = build_paper_readiness_wording(root=root, now=FIXED_NOW)
    assert prw["authorization_alone_permits_submission"] is False
    assert prw["order_ready"] is False


def test_expired_arm_explicitly_labeled(tmp_path: Path):
    from services.operator_console_status import build_paper_readiness_wording

    root = _seed(tmp_path, paper=False, arm="EXPIRED")
    prw = build_paper_readiness_wording(root=root, now=FIXED_NOW)
    assert prw["paper_arm_status"] == "EXPIRED"


def test_missing_or_malformed_evidence_unknown(tmp_path: Path):
    from services.operator_console_status import build_paper_readiness_wording

    root = _seed(tmp_path, paper=True, arm="MISSING")
    # Remove auth artifacts
    (root / "data" / "results" / "governance_authorization.json").unlink()
    (root / "data" / "results" / "execution_readiness.json").unlink()
    prw = build_paper_readiness_wording(root=root, now=FIXED_NOW)
    assert prw["standing_authorization_status"] == "UNKNOWN"
    assert prw["paper_arm_status"] == "MISSING"

    root2 = _seed(tmp_path / "m2", paper=True, arm="MALFORMED")
    prw2 = build_paper_readiness_wording(root=root2, now=FIXED_NOW)
    assert prw2["paper_arm_status"] in ("UNKNOWN", "INVALID", "EXPIRED", "MISSING")


def test_explanatory_copy_requires_arm_and_gates(tmp_path: Path):
    from services.operator_console_status import (
        PAPER_READINESS_EXPLANATORY_COPY,
        build_paper_readiness_wording,
    )

    root = _seed(tmp_path, paper=True, arm="EXPIRED")
    prw = build_paper_readiness_wording(root=root, now=FIXED_NOW)
    copy = prw["explanatory_copy"]
    assert copy == PAPER_READINESS_EXPLANATORY_COPY
    assert "separate" in copy.lower()
    assert "cannot submit" in copy.lower()
    assert "paper arm" in copy.lower()
    assert "broker-dispatch" in copy.lower() or "broker-dispatch" in copy
    assert "Live execution remains blocked" in copy


def test_live_execution_remains_visibly_blocked(tmp_path: Path):
    from services.operator_console_status import build_paper_readiness_wording

    root = _seed(tmp_path, paper=True, arm="EXPIRED")
    prw = build_paper_readiness_wording(root=root, now=FIXED_NOW)
    assert prw["live_execution_blocked"] is True
    assert "BLOCKED" in prw["live_execution_display"]


def test_no_handlers_during_wording_build(tmp_path: Path):
    from unittest.mock import patch

    from services.operator_console_status import build_paper_readiness_wording

    root = _seed(tmp_path, paper=True, arm="EXPIRED")
    with patch("services.broker_alpaca.AlpacaBroker") as broker, patch(
        "services.operator_runtime_controller.execute_start_triton"
    ) as start, patch("services.operator_cooperative_shutdown.create_shutdown_request") as shut:
        build_paper_readiness_wording(root=root, now=FIXED_NOW)
        assert broker.call_count == 0
        assert start.call_count == 0
        assert shut.call_count == 0


def test_import_causes_no_file_mutation(tmp_path: Path):
    import importlib

    import services.operator_console_status as mod

    before = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    importlib.reload(mod)
    after = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_operator_renderers_and_selected_renderer_pattern():
    src = VIEW.read_text(encoding="utf-8", errors="replace")
    assert "OPERATOR_RENDERERS" in src
    assert "OPERATOR_RENDERERS.get(page)" in src
    assert "_render_cc_operations_broker_order_evidence" in src
    assert "_render_cc_orders_broker_order_evidence" in src


def test_phase2_broker_freshness_intact():
    from services.operator_console_status import (
        BROKER_FRESH_MINUTES,
        classify_paper_broker_rest_evidence,
    )

    assert BROKER_FRESH_MINUTES == 120.0
    cls = classify_paper_broker_rest_evidence(
        broker_connected=True,
        evidence_timestamp=FIXED_NOW.isoformat(),
        now=FIXED_NOW,
    )
    assert cls["state"] == "CONNECTED"
    assert "Paper broker REST" in cls["display_label"]
    stale = classify_paper_broker_rest_evidence(
        broker_connected=True,
        evidence_timestamp=(FIXED_NOW - timedelta(hours=3)).isoformat(),
        now=FIXED_NOW,
    )
    assert stale["state"] == "STALE"


def test_order_evidence_separation_intact(tmp_path: Path):
    from services.operator_console_status import build_orders_summary

    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    (results / "open_orders_snapshot.csv").write_text(
        "snapshot_ts,symbol,status,id\n", encoding="utf-8"
    )
    import os

    os.utime(results / "open_orders_snapshot.csv", (FIXED_NOW.timestamp(), FIXED_NOW.timestamp()))
    (results / "paper_execution_audit.csv").write_text(
        "timestamp,symbol,submitted_order_id,status\n" "2026-06-08T14:08:16Z,XLK,oid-1,SUBMITTED\n",
        encoding="utf-8",
    )
    summary = build_orders_summary(root=tmp_path, now=FIXED_NOW)
    assert "current_open_orders" in summary
    assert "submission_history" in summary
    assert summary["open_orders"] == []
    assert len(summary["submission_history"]["submitted_rows"]) == 1


def test_view_results_uses_new_labels():
    src = VIEW.read_text(encoding="utf-8", errors="replace")
    assert "Standing paper authorization" in src or "standing_authorization_label" in src
    assert "Paper execution arm" in src or "paper_arm_label" in src
    assert "Paper authorization: **{'YES'" not in src
    assert "Ready to trade" not in src
    assert "build_paper_readiness_wording" in src


def test_ast_status_module_no_order_submission():
    tree = ast.parse(STATUS_MOD.read_text(encoding="utf-8"))
    banned = {"submit_order", "cancel_order", "set_paper_arm_confirm"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            assert name not in banned


def test_no_css_javascript_navigation_workaround_in_config_or_new_helpers():
    cfg = CONFIG.read_text(encoding="utf-8").lower()
    assert "style>" not in cfg
    assert "<script" not in cfg
    assert "display: none" not in cfg
    status_src = STATUS_MOD.read_text(encoding="utf-8")
    idx = status_src.find("def build_paper_readiness_wording")
    assert idx > 0
    chunk = status_src[idx : idx + 4000]
    assert "stylesheet" not in chunk.lower()
    assert "<script" not in chunk.lower()
