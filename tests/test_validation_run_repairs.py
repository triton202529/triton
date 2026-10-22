"""Validation-run repairs. Temporary fixtures only. October 19 evidence is read, never written."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "data/governance/basis_unproven_hold.json",
    "data/governance/paper_sell_arm_authority.json",
    "data/results/daily_refresh_fetch_receipt.json",
    "data/results/daily_prediction_refresh_stage_receipts_daily_prediction_refresh_20261019_183000.json",
    "data/results/daily_prediction_refresh_summary.json",
    "data/results/performance_accounting_events.jsonl",
    "data/results/watchdog_alerts.json",
    "data/results/paper_order_events.jsonl",
    "data/live/operator_closeout/2026-10-19.json",
    "data/live/live_armed.json",
    "data/live/live_confirm.json",
    "logs/daily_refresh_20261019_183000.log",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _protected() -> dict:
    return {rel: _sha256(ROOT / rel) for rel in PROTECTED}


def test_october_19_fetch_only_stays_incomplete():
    from services.daily_refresh_completion import (
        CLASS_FETCH_ONLY,
        evaluate_daily_refresh_completion,
    )

    before = _protected()
    verdict = evaluate_daily_refresh_completion(ROOT, "2026-10-19")
    assert verdict["classification"] == CLASS_FETCH_ONLY
    assert verdict["ok"] is False
    assert verdict["detail"] == CLASS_FETCH_ONLY
    fetch_only = evaluate_daily_refresh_completion(
        ROOT,
        "2026-10-19",
        receipt_doc={"target_session": "2026-10-19", "passed": True},
    )
    assert fetch_only["classification"] == CLASS_FETCH_ONLY
    assert fetch_only["ok"] is False
    assert _protected() == before


def test_session_budget_survives_a_new_invocation_tag(tmp_path: Path):
    from services.session_entry_budget import reserve_entries

    session = "2026-10-19"
    first = reserve_entries(
        tmp_path,
        session,
        [
            {"symbol": "PG", "action": "OPEN_NEW", "held_at_open": False},
            {"symbol": "USO", "action": "OPEN_NEW", "held_at_open": False},
            {"symbol": "WMT", "action": "OPEN_NEW", "held_at_open": False},
            {"symbol": "UNH", "action": "OPEN_NEW", "held_at_open": False},
            {"symbol": "MA", "action": "ADD", "held_at_open": True},
        ],
        invocation_tag="exec_trades_20261019_142329",
    )
    assert first["ok"] is True
    second = reserve_entries(
        tmp_path,
        session,
        [
            {"symbol": "PG", "action": "ADD", "held_at_open": True},
            {"symbol": "USO", "action": "ADD", "held_at_open": True},
            {"symbol": "WMT", "action": "ADD", "held_at_open": True},
            {"symbol": "UNH", "action": "ADD", "held_at_open": True},
            {"symbol": "MA", "action": "ADD", "held_at_open": True},
        ],
        invocation_tag="exec_trades_20261019_194108",
    )
    assert second["ok"] is False
    refused = {row["symbol"]: row["reason"] for row in second["refusals"]}
    assert refused["PG"] == "opened_today_cannot_add"
    assert refused["USO"] == "opened_today_cannot_add"
    assert refused["WMT"] == "opened_today_cannot_add"
    assert refused["UNH"] == "opened_today_cannot_add"
    assert refused["MA"] == "session_symbol_already_used"
    stored = json.loads(
        (tmp_path / "data" / "live" / "session_entry_budget" / f"{session}.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["symbol"] for row in stored["reservations"]] == [
        "PG",
        "USO",
        "WMT",
        "UNH",
        "MA",
    ]
    assert stored["reservations"][0]["invocation_tag"] == "exec_trades_20261019_142329"


def test_budget_read_failure_refuses_and_reentry_is_narrow(tmp_path: Path):
    from services.session_entry_budget import budget_path, reserve_entries

    path = budget_path(tmp_path, "2026-10-19")
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    refused = reserve_entries(
        tmp_path,
        "2026-10-19",
        [{"symbol": "MA", "action": "ADD", "held_at_open": True}],
        invocation_tag="exec_trades_later",
    )
    assert refused["ok"] is False
    assert refused["reason"] == "budget_unreadable"

    root = tmp_path / "fresh"
    opened = reserve_entries(
        root,
        "2026-10-19",
        [{"symbol": "PG", "action": "OPEN_NEW", "held_at_open": False}],
        invocation_tag="cycle-1",
    )
    assert opened["ok"] is True
    blocked = reserve_entries(
        root,
        "2026-10-19",
        [{"symbol": "PG", "action": "OPEN_NEW", "held_at_open": False}],
        invocation_tag="cycle-2",
    )
    assert blocked["reason"] == "session_symbol_already_used"
    reentry = reserve_entries(
        root,
        "2026-10-19",
        [
            {
                "symbol": "PG",
                "action": "OPEN_NEW",
                "held_at_open": False,
                "terminal_sell_proven": True,
                "broker_flat": True,
            }
        ],
        invocation_tag="cycle-3",
    )
    assert reentry["ok"] is True


def test_nonexecutable_nvda_authority_submits_nothing(monkeypatch):
    from services.paper_sell_arm_runtime_enforcement import (
        build_pre_plan_planned_orders,
    )

    calls = []

    class Broker:
        def list_positions(self):
            calls.append("list_positions")
            return [{"symbol": "NVDA", "qty": "1", "qty_available": "1"}]

        def submit_order(self, *args, **kwargs):
            calls.append("submit_order")

    monkeypatch.setattr(
        "services.place_live_orders.get_ref_price",
        lambda *_args, **_kwargs: 219.62,
    )
    authority = {
        "authorized_actions": [
            {"symbol": "NVDA", "action": "EXIT", "max_qty": 1.0, "max_notional": 217.58}
        ],
        "per_symbol_max_quantity": {"NVDA": 1.0},
        "per_symbol_max_notional": {"NVDA": 217.58},
        "total_batch_max_notional": 217.58,
    }
    ok, reason, planned = build_pre_plan_planned_orders(authority, Broker())
    assert ok is False
    assert reason == "SELL_AUTHORITY_NOT_EXECUTABLE"
    assert planned == ()
    assert calls == ["list_positions"]


def test_broker_held_nvda_stays_long():
    from services.apply_signal_lifecycle import _normalize_lifecycle_dataframe

    frame = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "position_state": "FLAT",
                "lifecycle_action": "EXIT",
                "stance": "EXIT",
            }
        ]
    )
    _normalize_lifecycle_dataframe(frame, broker_long={"NVDA"}, broker_evidence="fresh")
    assert frame.loc[0, "position_state"] == "LONG"
    assert frame.loc[0, "lifecycle_action"] == "EXIT"
    assert frame.loc[0, "stance"] == "EXIT"


def test_drawdown_blocks_only_after_its_timestamp():
    from services.governance_degraded_mode import (
        classify_governance_mode,
        entry_blocked_by_active_drawdown,
    )

    alerts = [
        {
            "alert_type": "EXCESS_POSITION_DRAWDOWN",
            "status": "ACTIVE",
            "subject": "NVDA",
            "first_seen": "2026-10-19T19:44:12Z",
        }
    ]
    before = datetime(2026, 10, 19, 19, 0, tzinfo=timezone.utc)
    after = datetime(2026, 10, 19, 19, 50, tzinfo=timezone.utc)
    assert entry_blocked_by_active_drawdown(alerts, before) is False
    assert entry_blocked_by_active_drawdown(alerts, after) is True
    classification = classify_governance_mode(
        gov_doc={"escalation_state": "YELLOW"},
        cpe_doc={
            "escalation_state": "YELLOW",
            "escalation_reason": ["EXCESS_POSITION_DRAWDOWN"],
        },
        governance_authorized=True,
    )
    assert classification["governance_mode"] == "GREEN"
    assert classification["risk_increasing_actions_blocked"] is True


def test_watchdog_long_pass_is_live_and_dead_or_hung_blocks():
    from services.governance_execution_readiness import _check_watchdog_health
    from services.risk_watchdog import (
        classify_watchdog_liveness,
        watchdog_liveness_blocks_entries,
    )

    now = datetime(2026, 10, 19, 20, 0, tzinfo=timezone.utc)
    completed = classify_watchdog_liveness(
        cycle_started_at="2026-10-19T19:40:00Z",
        last_completed_at="2026-10-19T19:55:00Z",
        owner_pid_alive=True,
        lock_present=True,
        lock_owner_alive=True,
        now=now,
    )
    assert completed == "OK"
    assert watchdog_liveness_blocks_entries(completed) is False
    dead = classify_watchdog_liveness(
        cycle_started_at="2026-10-19T19:40:00Z",
        last_completed_at="2026-10-19T19:30:00Z",
        owner_pid_alive=False,
        lock_present=False,
        lock_owner_alive=None,
        now=now,
    )
    hung = classify_watchdog_liveness(
        cycle_started_at="2026-10-19T19:30:00Z",
        last_completed_at="2026-10-19T19:20:00Z",
        owner_pid_alive=True,
        lock_present=True,
        lock_owner_alive=True,
        now=now,
    )
    stale_lock = classify_watchdog_liveness(
        cycle_started_at="2026-10-19T19:55:00Z",
        last_completed_at="2026-10-19T19:56:00Z",
        owner_pid_alive=False,
        lock_present=True,
        lock_owner_alive=False,
        now=now,
    )
    assert dead == "DEAD"
    assert hung == "HUNG"
    assert stale_lock == "DEAD"
    for classification in (dead, hung, stale_lock):
        assert watchdog_liveness_blocks_entries(classification) is True
        ok, _reason = _check_watchdog_health("OK", [], liveness=classification)
        assert ok is False


def test_dashboard_nonce_and_open_port_block(tmp_path: Path, monkeypatch):
    from services.dashboard_cooperative_shutdown import (
        acknowledge_dashboard_shutdown,
        ack_matches_request,
        request_dashboard_shutdown,
    )
    from services.operator_closeout_control import (
        STATE_PUBLISHING,
        STATE_STREAMLIT_STOP_FAILED,
        run_publishing_worker,
    )

    request = request_dashboard_shutdown(
        tmp_path, expected_pid=15, port=8501, session_date="2026-10-20"
    )
    mismatch = acknowledge_dashboard_shutdown(tmp_path, pid=15, presented_nonce="wrong")
    assert mismatch["ok"] is False
    assert mismatch["reason"] == "shutdown_nonce_mismatch"
    assert not (tmp_path / "data" / "live" / "dashboard_shutdown_ack.json").exists()
    matched = acknowledge_dashboard_shutdown(tmp_path, pid=15, presented_nonce=request["nonce"])
    assert matched["ok"] is True
    assert ack_matches_request(tmp_path) is True

    receipt_path = tmp_path / "data" / "live" / "operator_closeout" / "2026-10-20.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps({"state": STATE_PUBLISHING, "session_date": "2026-10-20"}),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        "services.dashboard_cooperative_shutdown.ack_matches_request",
        lambda _root: True,
    )
    result = run_publishing_worker(
        root=tmp_path,
        receipt_path=receipt_path,
        runtime_inspector=lambda _root: {
            "runtime_stopped": True,
            "watchdog_stopped": True,
            "streamlit_stopped": False,
            "ports_closed": False,
            "freshness_writer_idle": True,
            "freshness_lock_present": False,
        },
        git_inspector=lambda _root: {"git_identity_valid": True, "index_empty": True},
        processes=[{"pid": 15, "command_line": "python -m streamlit run view_results.py"}],
        killer=lambda pid: calls.append(("kill", pid)),
        publisher=lambda _root, _receipt: calls.append("publish"),
        own_pid=999,
    )
    assert result["state"] == STATE_STREAMLIT_STOP_FAILED
    assert result["reason"] == "dashboard_or_ports_still_present"
    assert calls == []


def test_portfolio_certified_and_provisional_panels_stay_separate(monkeypatch):
    import dashboard.operator_modern as modern

    calls = []

    class Column:
        def metric(self, label, value, *args, **kwargs):
            calls.append(f"{label}={value}")

    class Streamlit:
        def markdown(self, text, *args, **kwargs):
            calls.append(str(text))

        def caption(self, text, *args, **kwargs):
            calls.append(str(text))

        def columns(self, count):
            return [Column() for _ in range(int(count))]

    monkeypatch.setattr(modern, "st", Streamlit())
    modern._portfolio_metrics(
        {
            "cash": 38700.09,
            "market_value": 53210.83,
            "total_portfolio_value": 91910.92,
            "unrealized_pnl": -10.43,
            "as_of_et": "2026-10-19 16:05 ET",
        },
        [],
        certified={
            "available": True,
            "portfolio_value": "$91,691.63",
            "cash": "$75,271.72",
            "market_value": "$16,419.91",
            "as_of_et": "2026-10-16 16:00 ET",
        },
    )
    rendered = "\n".join(calls)
    certified_at = rendered.index("Latest Certified Close")
    provisional_at = rendered.index("Current Provisional Paper Snapshot")
    assert certified_at < provisional_at
    assert "Certified cash=$75,271.72" in rendered
    assert "Provisional cash=" in rendered
    assert "Provisional unrealized P/L=" in rendered
    assert "2026-10-16 16:00 ET" in rendered
    assert "2026-10-19 16:05 ET" in rendered


def test_basis_unproven_hold_is_unchanged():
    policy = json.loads(
        (ROOT / "data/governance/basis_unproven_hold.json").read_text(encoding="utf-8")
    )
    assert policy["policy_id"] == "basis-unproven-hold-voo-xlk-20261008"
    assert policy["status"] == "ACTIVE"
    assert {row["symbol"] for row in policy["symbols"]} == {"VOO", "XLK"}
    assert "BUY" in policy["blocked_actions"]
    assert "SELL" in policy["blocked_actions"]
    assert _sha256(ROOT / "data/governance/basis_unproven_hold.json") == (
        "c575d1a5a461704278943784e08b1f96cc53755cc7f66992b566d361e4ee0a55"
    )
