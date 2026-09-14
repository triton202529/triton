"""BP1E2 isolated dashboard hotfix commit-scope checks. Test-only. No production writes."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dashboard.freshness_presentation import (
    FROZEN_PREVIEW_LABEL,
    bind_freshness_presentation,
    reset_freshness_presentation,
    shared_freshness_view,
)
from dashboard.order_presentation import load_authoritative_presentation
from dashboard.preview_data import expected_ledger_sha256 as preview_expected_hash
from dashboard.preview_data import isolation_active
from tests.fixtures.bm5.authoritative import (
    EXPECTED_LEDGER_SHA256 as BM5_SHA256,
    materialize_authoritative_fixture,
)
from tests.test_command_center_navigation_readiness_wording import CANONICAL_OPERATOR_PAGES

ISO_ROOT = Path(__file__).resolve().parents[1]
CALENDAR = ISO_ROOT / "services" / "market_calendar.py"
LEDGER = ISO_ROOT / "services" / "paper_order_ledger.py"
NAV = ISO_ROOT / "dashboard" / "operator_nav.py"
CONSOLE = ISO_ROOT / "dashboard" / "operator_console.py"
CALENDAR_SHA256 = "338ba5b09572312a4486fa75f28e22bd181dbef59dc897279b25f8938526a0b6"
LEDGER_SHA256 = "3b52d05e01571aad698a0ef93a16ec4f56694595df4afc77004c5bb4d30abb3d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_canonical_operator_pages_include_overview_and_performance_in_order():
    nav = NAV.read_text(encoding="utf-8")
    groups = re.search(r"^NAV_GROUPS:.*?=\s*\[(.*?)\]\s*^UI_A_PAGES", nav, re.S | re.M)
    assert groups
    pages: list[str] = []
    for inner in re.findall(r'\("[^"]+",\s*\[(.*?)\]\)', groups.group(1), re.S):
        pages.extend(re.findall(r'"([^"]+)"', inner))
    assert pages == CANONICAL_OPERATOR_PAGES
    assert pages[0] == "Overview"
    assert "Performance" in pages
    assert pages == [
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


def test_protected_calendar_and_ledger_hashes_unchanged():
    assert _sha256(CALENDAR) == CALENDAR_SHA256
    assert _sha256(LEDGER) == LEDGER_SHA256


def test_operations_page_load_slice_finds_keyword_start_controls_call():
    src = CONSOLE.read_text(encoding="utf-8")
    ops_start = src.index("def render_operations_page")
    call = src.index("_render_start_controls(", ops_start)
    load = src[ops_start:call]
    assert "execute_stop_triton" not in load
    assert "_render_start_controls(status_snapshot=" in src[call : call + 80]


def test_production_orders_remain_unpinned_and_fail_closed(tmp_path: Path):
    materialize_authoritative_fixture(tmp_path)
    assert isolation_active(tmp_path) is False
    assert preview_expected_hash(tmp_path) is None
    counts = load_authoritative_presentation(root=tmp_path, expected_sha256=None)
    assert counts["ledger_sha256"] == BM5_SHA256
    assert "Plan" not in {row.get("side") for row in counts.get("rows") or []}


def test_freshness_bind_agrees_and_frozen_preview_is_isolation_only():
    reset_freshness_presentation()
    status = {"data_freshness": {"class": "AGING_AFTER_CLOSE", "heartbeat_timestamp": "ts"}}
    bound = bind_freshness_presentation(status, isolated=False)
    prod = shared_freshness_view(status, isolated=False)
    iso = shared_freshness_view(status, isolated=True)
    assert bound["evaluated_at"] == prod["evaluated_at"] == "ts"
    assert prod["value"] != FROZEN_PREVIEW_LABEL
    assert iso["value"] == FROZEN_PREVIEW_LABEL


def test_restart_fixture_seeds_temp_paper_arm_only():
    restart_src = (ISO_ROOT / "tests" / "test_operator_protected_restart_do006.py").read_text(
        encoding="utf-8"
    )
    assert 'live / "paper_arm_confirm.json"' in restart_src
    assert "services/operator_restart_action.py" not in restart_src
    now = datetime.now(timezone.utc)
    assert (now + timedelta(hours=2)) > now
