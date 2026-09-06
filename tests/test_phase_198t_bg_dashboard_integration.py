"""Phase 198T-BG dashboard presentation contracts (no Streamlit server)."""

from __future__ import annotations

from pathlib import Path

from dashboard.paper_sell_arm_controls import map_sell_arm_visual_status
from services.paper_order_ledger import consumed_authority_message
from services.paper_sell_arm_console import CONFIRM_SUCCESS, _dispatch_visibility_fields


def test_consumed_banner_never_says_no_orders_submitted():
    vis = {
        "submitted": 4,
        "skipped": 1,
        "failed": 0,
        "filled": 4,
        "filled_proceeds": 100.0,
        "has_dispatch": True,
    }
    msg = consumed_authority_message(vis)
    assert "No orders were submitted" not in msg
    assert "filled" in msg.lower()


def test_confirm_success_copy_still_exists_for_unused_create():
    assert "No orders were submitted" in CONFIRM_SUCCESS


def test_visual_completed_uses_dispatch_message():
    visual = map_sell_arm_visual_status(
        {
            "freshness": "STALE",
            "challenge_state": "CONSUMED",
            "authority_state": "CONSUMED",
            "plan_hash": "p",
            "batch_hash": "b",
            "snapshot": {"latest_authority": {"plan_hash": "p", "batch_hash": "b"}},
            "last_dispatch_message": "Paper SELL authority consumed: 4 submitted, 1 skipped, 0 failed. Open Orders or Activity for execution status.",
        }
    )
    assert visual["code"] == "COMPLETED"
    assert "No orders were submitted" not in visual["reason"]
    assert "4 submitted" in visual["reason"]


def test_dispatch_visibility_helper_fail_closed(tmp_path: Path):
    fields = _dispatch_visibility_fields(
        authority_state="CONSUMED",
        authority_id="missing",
        results_root=tmp_path,
    )
    assert fields["suppress_no_orders_submitted"] is False
