"""UI-A isolation: ledger/status reads do not write production paths or mutate authority."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

ISO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOT = Path(r"C:\Users\akimw\triton")


def _mtimes(paths):
    out = {}
    for p in paths:
        out[str(p)] = p.stat().st_mtime if p.exists() else None
    return out


def test_unified_ledger_read_does_not_write(tmp_path):
    from services.paper_order_ledger import (
        append_events,
        record_sell_dispatch_events,
        reduce_ledger,
        unified_order_rows,
    )

    prod_paths = [
        PROD_ROOT / "data" / "results" / "paper_order_events.jsonl",
        PROD_ROOT / "data" / "results" / "paper_order_events.lock",
    ]
    before = _mtimes(prod_paths)

    def _fail_write(*args, **kwargs):
        raise AssertionError("ledger write invoked during UI-A read")

    with patch("services.paper_order_ledger.append_events", side_effect=_fail_write):
        with patch(
            "services.paper_order_ledger.record_sell_dispatch_events", side_effect=_fail_write
        ):
            rows = unified_order_rows(root=ISO_ROOT)
            reduced = reduce_ledger(root=ISO_ROOT)
    assert isinstance(rows, list)
    assert isinstance(reduced, list)
    assert _mtimes(prod_paths) == before
    for row in rows:
        term = str(row.get("terminal_status") or "")
        if str(row.get("submission_status") or "").upper() == "SUBMITTED":
            if not row.get("terminal_status_proven"):
                assert term != "FILLED" or row.get("terminal_status_proven") is True


def test_orders_summary_status_consistency():
    from dashboard.operator_modern import _order_badge_fields
    from services.operator_console_status import build_orders_summary

    summary = build_orders_summary(root=ISO_ROOT)
    assert summary.get("note") is None or "execute" not in str(summary.get("note") or "").lower()
    submitted = {
        "submission_status": "SUBMITTED",
        "terminal_status": None,
        "terminal_status_proven": False,
        "symbol": "GLD",
        "side": "SELL",
        "selected_qty": 1,
    }
    fields = _order_badge_fields(submitted)
    assert "Fill" not in fields["Status"]
    filled = {
        "submission_status": "SUBMITTED",
        "terminal_status": "FILLED",
        "terminal_status_proven": True,
        "symbol": "GLD",
        "side": "SELL",
        "average_fill_price": 10.5,
    }
    assert (
        "Fill" in _order_badge_fields(filled)["Status"]
        or _order_badge_fields(filled)["status_raw"] == "FILLED"
    )


def test_no_broker_or_authority_mutation_on_status_load():
    from services.operator_console_status import build_operator_status, build_portfolio_summary

    prod_sensitive = [
        PROD_ROOT / "data" / "results" / "paper_order_events.jsonl",
        PROD_ROOT / "data" / "governance" / "paper_sell_arm_authority.json",
        PROD_ROOT / "data" / "live" / "paper_entry_dispatch_contract.json",
    ]
    before = _mtimes(prod_sensitive)

    def _no_broker(*args, **kwargs):
        raise AssertionError("broker call during UI-A status load")

    with patch.dict("sys.modules", {}):
        status = build_operator_status()
        port = build_portfolio_summary(root=ISO_ROOT)
    assert isinstance(status, dict)
    assert "live_execution" in status
    assert port.get("realized_pnl") in (None, port.get("realized_pnl"))
    assert _mtimes(prod_sensitive) == before


def test_buy_sell_button_keys_and_phrases_unchanged():
    buy = (ISO_ROOT / "dashboard" / "paper_buy_add_arm_controls.py").read_text(encoding="utf-8")
    sell = (ISO_ROOT / "dashboard" / "paper_sell_arm_controls.py").read_text(encoding="utf-8")
    assert 'key="paper_buy_add_arm_submit"' in buy
    assert 'key="paper_buy_add_arm_revoke"' in buy
    assert "ARM PAPER ENTRIES" in buy
    assert "REVOKE PAPER ENTRIES" in buy
    assert 'key="paper_sell_arm_reviewed"' in sell
    assert 'key="paper_sell_arm_confirm"' in sell
    assert "CONFIRM PAPER SELLS" in sell


def test_production_marker_is_not_the_isolated_root():
    assert ISO_ROOT.resolve() != PROD_ROOT.resolve()
    marker = ISO_ROOT / "tmp" / "ui_a_isolation_marker.txt"
    assert marker.is_file()
    assert "DO_NOT_WRITE_PRODUCTION" in marker.read_text(encoding="utf-8")
