"""
Command Center broker freshness + order evidence contract tests.

Isolated fixtures, fixed UTC clock. No Streamlit server, no broker network,
no continuous/watchdog runtime launch, no file mutation on import.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "view_results.py"
STATUS_MOD = ROOT / "services" / "operator_console_status.py"

FIXED_NOW = datetime(2026, 8, 3, 4, 0, 0, tzinfo=timezone.utc)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _seed_minimal_root(tmp_path: Path) -> Path:
    results = tmp_path / "data" / "results"
    live = tmp_path / "data" / "live"
    results.mkdir(parents=True)
    live.mkdir(parents=True)
    _write_json(
        results / "execution_readiness.json",
        {
            "live_execution_permitted": False,
            "mode": "PAPER_ONLY",
            "paper_execution_permitted": True,
        },
    )
    _write_json(
        results / "governance_authorization.json",
        {"live_execution_permitted": False, "paper_execution_permitted": True},
    )
    _write_json(results / "heartbeat.json", {"timestamp": FIXED_NOW.isoformat()})
    return tmp_path


def test_fresh_broker_connected_true_produces_connected(tmp_path: Path):
    from services.operator_console_status import (
        classify_paper_broker_rest_evidence,
        build_operator_status,
    )

    cls = classify_paper_broker_rest_evidence(
        broker_connected=True,
        evidence_timestamp=FIXED_NOW.isoformat(),
        now=FIXED_NOW,
    )
    assert cls["state"] == "CONNECTED"
    assert cls["connected"] is True
    assert "Paper broker REST: Connected" == cls["display_label"]
    assert cls["streaming_assessed"] is False

    root = _seed_minimal_root(tmp_path)
    _write_json(
        root / "data" / "results" / "watchdog_status.json",
        {"timestamp": FIXED_NOW.isoformat(), "broker_connected": True},
    )
    status = build_operator_status(root=root, now=FIXED_NOW)
    assert status["broker"]["state"] == "CONNECTED"
    assert status["broker"]["connected"] is True


def test_fresh_broker_connected_false_produces_disconnected():
    from services.operator_console_status import classify_paper_broker_rest_evidence

    cls = classify_paper_broker_rest_evidence(
        broker_connected=False,
        evidence_timestamp=(FIXED_NOW - timedelta(minutes=5)).isoformat(),
        now=FIXED_NOW,
    )
    assert cls["state"] == "DISCONNECTED"
    assert cls["connected"] is False
    assert cls["display_label"] == "Paper broker REST: Disconnected"


def test_evidence_older_than_120_minutes_produces_stale():
    from services.operator_console_status import (
        BROKER_FRESH_MINUTES,
        classify_paper_broker_rest_evidence,
    )

    old = FIXED_NOW - timedelta(minutes=BROKER_FRESH_MINUTES + 1)
    cls = classify_paper_broker_rest_evidence(
        broker_connected=True,
        evidence_timestamp=old.isoformat(),
        now=FIXED_NOW,
    )
    assert cls["state"] == "STALE"
    assert cls["connected"] is None
    assert cls["freshness"] == "STALE"


def test_missing_evidence_produces_unknown():
    from services.operator_console_status import classify_paper_broker_rest_evidence

    cls = classify_paper_broker_rest_evidence(
        broker_connected=True,
        evidence_timestamp=None,
        now=FIXED_NOW,
    )
    assert cls["state"] == "UNKNOWN"
    assert cls["connected"] is None


def test_malformed_timestamp_never_produces_connected():
    from services.operator_console_status import classify_paper_broker_rest_evidence

    cls = classify_paper_broker_rest_evidence(
        broker_connected=True,
        evidence_timestamp="not-a-timestamp",
        now=FIXED_NOW,
    )
    assert cls["state"] == "UNKNOWN"
    assert cls["state"] != "CONNECTED"


def test_future_timestamp_never_produces_connected():
    from services.operator_console_status import classify_paper_broker_rest_evidence

    future = FIXED_NOW + timedelta(hours=6)
    cls = classify_paper_broker_rest_evidence(
        broker_connected=True,
        evidence_timestamp=future.isoformat(),
        now=FIXED_NOW,
    )
    assert cls["state"] == "UNKNOWN"
    assert cls["state"] != "CONNECTED"


def test_stopped_runtime_does_not_override_fresh_broker_evidence(tmp_path: Path):
    from services.operator_console_status import build_operator_status

    root = _seed_minimal_root(tmp_path)
    _write_json(
        root / "data" / "results" / "watchdog_status.json",
        {"timestamp": FIXED_NOW.isoformat(), "broker_connected": True},
    )
    # No fixture_process_state.json → fixture root treats runtimes as stopped.
    status = build_operator_status(root=root, now=FIXED_NOW)
    assert status["watchdog"]["process_running"] is False
    assert status["continuous"]["process_running"] is False
    assert status["broker"]["state"] == "CONNECTED"


def test_badge_identifies_paper_rest_not_streaming():
    from services.operator_console_status import classify_paper_broker_rest_evidence

    cls = classify_paper_broker_rest_evidence(
        broker_connected=True,
        evidence_timestamp=FIXED_NOW.isoformat(),
        now=FIXED_NOW,
    )
    assert "Paper broker REST" in cls["display_label"]
    assert cls["streaming_assessed"] is False
    assert cls["streaming_connected"] is None
    assert "stream" in cls["meaning"].lower()
    assert "not assessed" in cls["meaning"].lower()


def test_no_direct_broker_method_during_status_build(tmp_path: Path):
    from services.operator_console_status import build_operator_status, build_orders_summary

    root = _seed_minimal_root(tmp_path)
    _write_json(
        root / "data" / "results" / "watchdog_status.json",
        {"timestamp": FIXED_NOW.isoformat(), "broker_connected": True},
    )
    (root / "data" / "results" / "open_orders_snapshot.csv").write_text(
        "snapshot_ts,symbol,status,id\n", encoding="utf-8"
    )
    with patch("services.broker_alpaca.AlpacaBroker") as broker_cls:
        build_operator_status(root=root, now=FIXED_NOW)
        build_orders_summary(root=root, now=FIXED_NOW)
        assert broker_cls.call_count == 0


def test_fresh_zero_open_orders_message(tmp_path: Path):
    from services.operator_console_status import build_orders_summary

    root = _seed_minimal_root(tmp_path)
    path = root / "data" / "results" / "open_orders_snapshot.csv"
    path.write_text("snapshot_ts,symbol,status,id\n", encoding="utf-8")
    # Force mtime via rewrite after setting — use snapshot_ts in empty file: no rows,
    # so evidence falls back to mtime; patch classify clock via now=FIXED_NOW and touch mtime.
    import os

    ts = FIXED_NOW.timestamp()
    os.utime(path, (ts, ts))
    summary = build_orders_summary(root=root, now=FIXED_NOW)
    cur = summary["current_open_orders"]
    assert cur["freshness"] == "FRESH"
    assert cur["count"] == 0
    assert cur["message"] == "No current open orders."
    assert cur["rows"] == []


def test_stale_order_evidence_does_not_claim_zero(tmp_path: Path):
    from services.operator_console_status import build_orders_summary

    root = _seed_minimal_root(tmp_path)
    path = root / "data" / "results" / "open_orders_snapshot.csv"
    path.write_text("snapshot_ts,symbol,status,id\n", encoding="utf-8")
    import os

    old = (FIXED_NOW - timedelta(hours=5)).timestamp()
    os.utime(path, (old, old))
    summary = build_orders_summary(root=root, now=FIXED_NOW)
    cur = summary["current_open_orders"]
    assert cur["freshness"] == "STALE"
    assert cur["count"] is None
    assert cur["message"] != "No current open orders."
    assert "stale" in (cur["message"] or "").lower()


def test_historical_submitted_only_in_submission_history(tmp_path: Path):
    from services.operator_console_status import build_orders_summary

    root = _seed_minimal_root(tmp_path)
    oo = root / "data" / "results" / "open_orders_snapshot.csv"
    oo.write_text("snapshot_ts,symbol,status,id\n", encoding="utf-8")
    import os

    os.utime(oo, (FIXED_NOW.timestamp(), FIXED_NOW.timestamp()))
    audit = root / "data" / "results" / "paper_execution_audit.csv"
    audit.write_text(
        "timestamp,symbol,submitted_order_id,status\n" "2026-06-08T14:08:16Z,XLK,oid-1,SUBMITTED\n",
        encoding="utf-8",
    )
    summary = build_orders_summary(root=root, now=FIXED_NOW)
    assert summary["current_open_orders"]["rows"] == []
    assert summary["open_orders"] == []
    submitted = summary["submission_history"]["submitted_rows"]
    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "XLK"
    assert submitted[0]["status"] == "SUBMITTED"
    assert submitted[0]["ui_classification"] == "Historical submission"
    assert "Open" not in submitted[0]["ui_classification"]
    assert "Working" not in submitted[0]["ui_classification"]


def test_submitted_not_classified_as_open_or_working(tmp_path: Path):
    from services.operator_console_status import build_orders_summary

    root = _seed_minimal_root(tmp_path)
    (root / "data" / "results" / "paper_execution_audit.csv").write_text(
        "timestamp,symbol,submitted_order_id,status\n"
        "2026-06-08T14:08:16Z,TSLA,oid-2,SUBMITTED\n",
        encoding="utf-8",
    )
    summary = build_orders_summary(root=root, now=FIXED_NOW)
    for row in summary["submission_history"]["submitted_rows"]:
        assert row["ui_classification"] == "Historical submission"
        assert row["final_broker_status"] == "Final broker status unavailable."


def test_unknown_final_status_preserved(tmp_path: Path):
    from services.operator_console_status import build_orders_summary

    root = _seed_minimal_root(tmp_path)
    (root / "data" / "results" / "paper_execution_audit.csv").write_text(
        "timestamp,symbol,submitted_order_id,status\n" "2026-06-08T14:08:16Z,QQQ,oid-3,SUBMITTED\n",
        encoding="utf-8",
    )
    summary = build_orders_summary(root=root, now=FIXED_NOW)
    row = summary["submission_history"]["submitted_rows"][0]
    assert row["final_broker_status"] == "Final broker status unavailable."
    assert "404" not in row["final_broker_status"]
    assert row["final_broker_status"] not in ("FILLED", "CANCELLED", "REJECTED")


def test_separate_evidence_contracts(tmp_path: Path):
    from services.operator_console_status import build_orders_summary

    root = _seed_minimal_root(tmp_path)
    summary = build_orders_summary(root=root, now=FIXED_NOW)
    assert "current_open_orders" in summary
    assert "submission_history" in summary
    assert summary["current_open_orders"] is not summary["submission_history"]
    note = summary.get("note") or ""
    assert "separate" in note.lower()


def test_operator_renderers_reused_and_selected_only():
    src = VIEW.read_text(encoding="utf-8", errors="replace")
    assert "OPERATOR_RENDERERS" in src
    assert "OPERATOR_RENDERERS.get(page)" in src
    assert "_render_cc_operations_broker_order_evidence" in src
    assert "_render_cc_orders_broker_order_evidence" in src
    # Activity/Portfolio/Risk still go through OPERATOR_RENDERERS
    assert "OPERATOR_RENDERERS.get(page)" in src


def test_protected_handlers_not_invoked_by_status_loaders(tmp_path: Path):
    from services.operator_console_status import build_operator_status, build_orders_summary

    root = _seed_minimal_root(tmp_path)
    _write_json(
        root / "data" / "results" / "watchdog_status.json",
        {"timestamp": FIXED_NOW.isoformat(), "broker_connected": True},
    )
    with patch("services.operator_runtime_controller.execute_start_triton") as p1, patch(
        "services.operator_cooperative_shutdown.create_shutdown_request"
    ) as p3:
        build_operator_status(root=root, now=FIXED_NOW)
        build_orders_summary(root=root, now=FIXED_NOW)
        assert p1.call_count == 0
        assert p3.call_count == 0


def test_import_causes_no_file_mutation(tmp_path: Path):
    before = {p.name: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    import importlib
    import services.operator_console_status as mod

    importlib.reload(mod)
    after = {p.name: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_no_iframe_subprocess_second_server_or_refresh_loop_added():
    src = VIEW.read_text(encoding="utf-8", errors="replace")
    # Narrow: new evidence helpers must not introduce these patterns.
    ops_fn = src[
        src.index("def _render_cc_operations_broker_order_evidence") : src.index(
            "def _render_cc_orders_broker_order_evidence"
        )
    ]
    orders_fn = src[
        src.index("def _render_cc_orders_broker_order_evidence") : src.index("def main()")
    ]
    for block in (ops_fn, orders_fn):
        assert "iframe" not in block.lower()
        assert "subprocess" not in block
        assert "8502" not in block
        assert "st_autorefresh" not in block
        assert "AlpacaBroker" not in block


def test_authority_gates_and_live_block_remain():
    src = VIEW.read_text(encoding="utf-8", errors="replace")
    assert "LIVE EXECUTION BLOCKED" in src
    assert "_render_start_controls" in src
    status_src = STATUS_MOD.read_text(encoding="utf-8", errors="replace")
    assert "live_execution_permitted" in status_src
    assert "BROKER_FRESH_MINUTES" in status_src
    assert "classify_paper_broker_rest_evidence" in status_src


def test_ast_no_broker_calls_in_status_module():
    tree = ast.parse(STATUS_MOD.read_text(encoding="utf-8"))
    banned = {"get_account", "list_orders", "submit_order", "cancel_order", "AlpacaBroker"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            assert name not in banned
