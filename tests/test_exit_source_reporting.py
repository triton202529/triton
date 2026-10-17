"""Display-only exit-source labels for broker-activity sell rows."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from dashboard.operator_reports import broker_activity_display_rows
from services.exit_source_reporting import (
    EXIT_SOURCE_MANUAL,
    EXIT_SOURCE_TRITON,
    EXIT_SOURCE_UNLINKED,
    MANUAL_PROFIT_TAKING_REASON,
    ExitSourceIndex,
)
from services.reconciled_portfolio_reporting import (
    ReportRequest,
    _build_activity,
    build_reconciled_report,
)

ROOT = Path(__file__).resolve().parents[1]

OCTOBER_SELLS = {
    "8aba8b08-2836-4f0b-9b91-efa7dbf76a3a": ("SPCX", EXIT_SOURCE_TRITON, "8"),
    "84ba9dc1-fba6-434e-8958-079debd1a56b": ("AMD", EXIT_SOURCE_TRITON, "2"),
    "bf26f7d0-e678-4dd2-8500-a0bf7d23b27c": ("NVDA", EXIT_SOURCE_TRITON, "3"),
    "6db1af0b-7cbf-4837-8976-f6d39dc63694": ("INTC", EXIT_SOURCE_TRITON, "12"),
    "507112c1-8984-4450-8c67-f9d1bca5a8ab": ("BITO", EXIT_SOURCE_MANUAL, "450"),
    "7443d112-0191-4eb5-89dd-69b59b950577": ("BITO", EXIT_SOURCE_MANUAL, "400"),
    "99bd5524-40c6-439b-8600-533d937140dc": ("BITO", EXIT_SOURCE_MANUAL, "19"),
}

PROTECTED = (
    "data/results/paper_order_events.jsonl",
    "data/results/recent_orders.csv",
    "data/results/trade_outcomes.csv",
    "data/results/paper_execution_audit.csv",
    "data/governance/execution_gate_decisions.json",
    "data/governance/execution_gate_evidence.json",
    "data/governance/execution_approval_requests.json",
    "data/governance/paper_sell_arm_authority.json",
    "data/governance/paper_sell_arm_consumption.json",
    "data/governance/paper_sell_arm_dispatch_permit.json",
    "data/governance/paper_sell_arm_dispatch_intent.json",
    "data/governance/paper_sell_arm_state.json",
    "data/live/paper_sell_arm.sqlite3",
    "data/governance/manual_portfolio_exit/manual_portfolio_exit.sqlite3",
    "data/governance/manual_portfolio_exit/requests/0f628c831b6e31d43583ceb610e13db1ed32c1bd727ad32e5856d5e6e0a2215f.json",
    "data/governance/manual_portfolio_exit/requests/191fcd3e5a3bac36cc0e5200c9ab4c91bc00cbb51a9f3398ad2e20fbaf1afd16.json",
    "data/governance/manual_portfolio_exit/requests/eedd181cd423bc6382ee97ed9f01778cbd03ba6d95172bd1e06a1221bc9e670a.json",
    "data/governance/manual_portfolio_exit/dispatch_terminal/sell-dispatch-f40ccbd63cf54e52.json",
    "data/governance/manual_portfolio_exit/dispatch_terminal/sell-dispatch-ed4565837b154c44.json",
    "data/governance/manual_portfolio_exit/dispatch_terminal/sell-dispatch-d17896a20e9846af.json",
    "data/governance/manual_portfolio_exit/order_identities/507112c1-8984-4450-8c67-f9d1bca5a8ab.json",
    "data/governance/manual_portfolio_exit/order_identities/7443d112-0191-4eb5-89dd-69b59b950577.json",
    "data/governance/manual_portfolio_exit/order_identities/99bd5524-40c6-439b-8600-533d937140dc.json",
)

BROKER = "broker-spcx-1"
CLIENT = "triton-preplan-SPCX-EXIT-abc"
AUTH = "sell-arm-audit-test"
DISP = "sell-dispatch-test"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protected_hashes(root: Path = ROOT) -> dict:
    hashes = {}
    for rel in PROTECTED:
        path = root / rel
        hashes[rel] = _sha256(path) if path.is_file() else None
    return hashes


def _names(path: Path) -> list:
    if not path.is_dir():
        return []
    return sorted(item.name for item in path.iterdir())


def _write_orders(root: Path, rows: list) -> None:
    path = root / "data" / "results" / "recent_orders.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "client_order_id",
        "symbol",
        "side",
        "status",
        "filled_qty",
        "filled_at",
        "submitted_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _order(**overrides) -> dict:
    row = {
        "id": BROKER,
        "client_order_id": CLIENT,
        "symbol": "SPCX",
        "side": "sell",
        "status": "filled",
        "filled_qty": "8",
        "filled_at": "2026-10-16T14:28:01+00:00",
        "submitted_at": "2026-10-16T14:28:00+00:00",
    }
    row.update(overrides)
    return row


def _write_triton_sqlite(
    root: Path,
    *,
    authority_state: str = "CONSUMED",
    consuming: str = DISP,
    dispatch_state: str = "SUBMITTED",
    broker_ref: str = BROKER,
    client: str = CLIENT,
    qty: str = "8",
    symbol: str = "SPCX",
    max_qty: str = "8",
    authority_id: str = AUTH,
    dispatch_id: str = DISP,
) -> None:
    path = root / "data" / "live" / "paper_sell_arm.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    bounds = [{"action": "EXIT", "max_qty": float(max_qty), "symbol": symbol}]
    payload = {"authorized_actions": bounds}
    intent = {
        "requested": [
            {
                "action": "EXIT",
                "client_order_id": client,
                "qty": float(qty),
                "symbol": symbol,
            }
        ]
    }
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE sell_authorities (
              authority_id TEXT, state TEXT, consuming_dispatch_id TEXT,
              canonical_batch_json TEXT, payload_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE sell_dispatches (
              dispatch_id TEXT, authority_id TEXT, state TEXT,
              broker_ref TEXT, intent_json TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO sell_authorities VALUES (?,?,?,?,?)",
            (
                authority_id,
                authority_state,
                consuming,
                json.dumps(bounds),
                json.dumps(payload),
            ),
        )
        conn.execute(
            "INSERT INTO sell_dispatches VALUES (?,?,?,?,?)",
            (dispatch_id, authority_id, dispatch_state, broker_ref, json.dumps(intent)),
        )
        conn.commit()
    finally:
        conn.close()


def _write_gate(
    root: Path,
    *,
    decision: str = "ALLOW",
    client: str = CLIENT,
    symbol: str = "SPCX",
    quantity: str = "",
) -> None:
    path = root / "data" / "governance" / "execution_gate_decisions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": [
            {
                "execution_identifier": client,
                "execution_gate_decision": decision,
                "allowed": decision == "ALLOW",
                "symbol(s)": symbol,
                "quantity": quantity,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ledger_event(**overrides) -> dict:
    event = {
        "event_type": "ACTION_SUBMITTED",
        "source_pipeline": "preplan_authorized_sell",
        "authority_id": AUTH,
        "dispatch_id": DISP,
        "symbol": "SPCX",
        "side": "SELL",
        "selected_qty": 8,
        "client_order_id": CLIENT,
        "broker_order_id": BROKER,
        "broker_terminal_status": None,
        "filled_qty": None,
        "terminal_status_proven": False,
    }
    event.update(overrides)
    return event


def _write_ledger(root: Path, events: list) -> None:
    path = root / "data" / "results" / "paper_order_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(event) for event in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _triton_events() -> list:
    return [
        _ledger_event(),
        _ledger_event(
            event_type="ORDER_TERMINAL_RECONCILED",
            source_pipeline="recent_orders_snapshot",
            broker_terminal_status="FILLED",
            filled_qty=8,
            terminal_status_proven=True,
        ),
    ]


def _write_triton_chain(root: Path) -> None:
    _write_triton_sqlite(root)
    _write_gate(root)
    _write_ledger(root, _triton_events())


def _classify(root: Path, **overrides) -> str:
    order = _order(**overrides)
    return ExitSourceIndex.load(root).classify(
        broker_order_id=order["id"],
        client_order_id=order["client_order_id"],
        symbol=order["symbol"],
        filled_qty=order["filled_qty"],
    )


def _write_manual_chain(
    root: Path, *, reason: str = MANUAL_PROFIT_TAKING_REASON, qty: str = "450"
) -> None:
    authority_id = "manual-auth-test"
    dispatch_id = "sell-dispatch-manual"
    broker = "broker-bito-1"
    client = "manual-BITO-test"
    path = root / "data" / "governance" / "manual_portfolio_exit" / "manual_portfolio_exit.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    intent = {
        "broker_order_id": broker,
        "requested": [{"action": "EXIT", "qty": float(qty), "side": "sell", "symbol": "BITO"}],
    }
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE sell_authorities (
              authority_id TEXT, state TEXT, consuming_dispatch_id TEXT,
              canonical_batch_json TEXT, payload_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE sell_dispatches (
              dispatch_id TEXT, authority_id TEXT, state TEXT,
              broker_ref TEXT, intent_json TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO sell_authorities VALUES (?,?,?,?,?)",
            (authority_id, "CONSUMED", dispatch_id, "[]", "{}"),
        )
        conn.execute(
            "INSERT INTO sell_dispatches VALUES (?,?,?,?,?)",
            (dispatch_id, authority_id, "SUBMITTED", broker, json.dumps(intent)),
        )
        conn.commit()
    finally:
        conn.close()
    request = {
        "authority_id": authority_id,
        "dispatch_id": dispatch_id,
        "broker_order_id": broker,
        "client_order_id": client,
        "symbol": "BITO",
        "selected_qty": qty,
        "reason": reason,
    }
    receipt = {
        "authority_id": authority_id,
        "dispatch_id": dispatch_id,
        "broker_order_id": broker,
        "client_order_id": client,
        "symbol": "BITO",
        "selected_qty": qty,
        "reconciliation_state": "terminal_reconciled",
    }
    requests = root / "data" / "governance" / "manual_portfolio_exit" / "requests"
    receipts = root / "data" / "governance" / "manual_portfolio_exit" / "dispatch_terminal"
    requests.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    (requests / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (receipts / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    _write_ledger(
        root,
        [
            _ledger_event(
                source_pipeline="manual_portfolio_exit",
                authority_id=authority_id,
                dispatch_id=dispatch_id,
                symbol="BITO",
                selected_qty=float(qty),
                client_order_id=client,
                broker_order_id=broker,
            ),
            _ledger_event(
                event_type="ORDER_TERMINAL_RECONCILED",
                source_pipeline="recent_orders_snapshot",
                authority_id=authority_id,
                dispatch_id=dispatch_id,
                symbol="BITO",
                selected_qty=float(qty),
                filled_qty=float(qty),
                client_order_id=client,
                broker_order_id=broker,
                broker_terminal_status="FILLED",
                terminal_status_proven=True,
            ),
        ],
    )


def test_october_sells_receive_expected_exit_source():
    before = _protected_hashes()
    live_names = _names(ROOT / "data" / "live")
    manual_names = _names(ROOT / "data" / "governance" / "manual_portfolio_exit")
    report = build_reconciled_report(
        ReportRequest(
            period_type="Custom",
            custom_start=date(2026, 10, 15),
            custom_end=date(2026, 10, 16),
        ),
        root=ROOT,
        now=datetime(2026, 10, 17, 16, 0, tzinfo=timezone.utc),
    )
    assert report["writes_performed"] is False
    assert _protected_hashes() == before
    assert _names(ROOT / "data" / "live") == live_names
    assert _names(ROOT / "data" / "governance" / "manual_portfolio_exit") == manual_names
    rows = {row["order_id"]: row for row in report["activity"]["broker_orders"]}
    assert len(OCTOBER_SELLS) == 7
    for order_id, (symbol, label, qty) in OCTOBER_SELLS.items():
        row = rows[order_id]
        assert row["symbol"] == symbol
        assert row["side"] == "sell"
        assert row["filled_qty"] == qty
        assert row["exit_source"] == label
    nvda = rows["bf26f7d0-e678-4dd2-8500-a0bf7d23b27c"]
    assert nvda["filled_qty"] == "3"
    assert nvda["exit_source"] == EXIT_SOURCE_TRITON


def test_attribution_counts_stay_independent_of_exit_source():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_orders(
            root,
            [
                _order(id="linked-prefix", client_order_id="triton-only-prefix"),
                _order(id="plain-sell", client_order_id="broker-native", symbol="AAPL"),
                _order(id="buy-1", client_order_id="triton-buy", side="buy", symbol="MSFT"),
            ],
        )
        activity = _build_activity(root=root, start=date(2026, 10, 16), end=date(2026, 10, 16))
    by_id = {row["order_id"]: row for row in activity["broker_orders"]}
    assert by_id["linked-prefix"]["attribution"] == "TRITON_LINKED"
    assert by_id["linked-prefix"]["exit_source"] == EXIT_SOURCE_UNLINKED
    assert by_id["plain-sell"]["attribution"] == "UNLINKED_BROKER_ACTIVITY"
    assert by_id["plain-sell"]["exit_source"] == EXIT_SOURCE_UNLINKED
    assert activity["counts"]["triton_linked"] == 2
    assert activity["counts"]["unlinked_broker_activity"] == 1
    assert "exit_source" not in by_id["buy-1"]


def test_prefix_only_evidence_remains_unlinked():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_gate(root, client="triton-prefix-only")
        _write_ledger(
            root,
            [
                _ledger_event(
                    source_pipeline="place_live_orders",
                    client_order_id="triton-prefix-only",
                    authority_id=None,
                    dispatch_id=None,
                )
            ],
        )
        assert (
            _classify(root, client_order_id="triton-prefix-only", id="missing-broker")
            == EXIT_SOURCE_UNLINKED
        )
        assert _classify(root, client_order_id="manual-prefix-only") == EXIT_SOURCE_UNLINKED


def test_complete_chains_and_fail_closed_mismatches():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_triton_chain(root)
        assert _classify(root) == EXIT_SOURCE_TRITON
        _write_triton_sqlite(root, max_qty="4", qty="3")
        _write_ledger(
            root,
            [
                _ledger_event(selected_qty=3),
                _ledger_event(
                    event_type="ORDER_TERMINAL_RECONCILED",
                    broker_terminal_status="FILLED",
                    filled_qty=3,
                    selected_qty=3,
                    terminal_status_proven=True,
                ),
            ],
        )
        assert _classify(root, filled_qty="3") == EXIT_SOURCE_TRITON

    mismatches = []

    def check(name: str, mutate) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_triton_chain(root)
            mutate(root)
            mismatches.append((name, _classify(root)))

    check("authority_unused", lambda root: _write_triton_sqlite(root, authority_state="UNUSED"))
    check(
        "dispatch_not_submitted",
        lambda root: _write_triton_sqlite(root, dispatch_state="INTENT_COMMITTED"),
    )
    check(
        "dispatch_id_mismatch",
        lambda root: _write_triton_sqlite(root, consuming="sell-dispatch-other"),
    )
    check("broker_mismatch", lambda root: _write_triton_sqlite(root, broker_ref="other-broker"))
    check("client_mismatch", lambda root: _write_triton_sqlite(root, client="triton-other"))
    check("symbol_mismatch", lambda root: _write_triton_sqlite(root, symbol="AMD"))
    check("quantity_mismatch", lambda root: _write_triton_sqlite(root, qty="7"))
    check("bound_exceeded", lambda root: _write_triton_sqlite(root, max_qty="4"))
    check(
        "gate_missing",
        lambda root: (root / "data" / "governance" / "execution_gate_decisions.json").unlink(),
    )
    check("gate_deny", lambda root: _write_gate(root, decision="DENY"))
    check("gate_symbol_mismatch", lambda root: _write_gate(root, symbol="AMD"))
    check(
        "submitted_pipeline_mismatch",
        lambda root: _write_ledger(
            root,
            [
                _ledger_event(source_pipeline="place_live_orders"),
                _triton_events()[1],
            ],
        ),
    )
    check("terminal_missing", lambda root: _write_ledger(root, [_ledger_event()]))
    check(
        "terminal_unproven",
        lambda root: _write_ledger(
            root,
            [
                _ledger_event(),
                _ledger_event(
                    event_type="ORDER_TERMINAL_RECONCILED",
                    broker_terminal_status="FILLED",
                    filled_qty=8,
                    terminal_status_proven=False,
                ),
            ],
        ),
    )
    check(
        "terminal_client_mismatch",
        lambda root: _write_ledger(
            root,
            [
                _ledger_event(),
                _ledger_event(
                    event_type="ORDER_TERMINAL_RECONCILED",
                    client_order_id="triton-other",
                    broker_terminal_status="FILLED",
                    filled_qty=8,
                    terminal_status_proven=True,
                ),
            ],
        ),
    )
    for name, label in mismatches:
        assert label == EXIT_SOURCE_UNLINKED, name

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_manual_chain(root)
        assert (
            ExitSourceIndex.load(root).classify(
                broker_order_id="broker-bito-1",
                client_order_id="manual-BITO-test",
                symbol="BITO",
                filled_qty="450",
            )
            == EXIT_SOURCE_MANUAL
        )
        request = (
            root / "data" / "governance" / "manual_portfolio_exit" / "requests" / "request.json"
        )
        payload = json.loads(request.read_text(encoding="utf-8"))
        payload["reason"] = "profit taking"
        request.write_text(json.dumps(payload), encoding="utf-8")
        assert (
            ExitSourceIndex.load(root).classify(
                broker_order_id="broker-bito-1",
                client_order_id="manual-BITO-test",
                symbol="BITO",
                filled_qty="450",
            )
            == EXIT_SOURCE_UNLINKED
        )
        payload["reason"] = MANUAL_PROFIT_TAKING_REASON
        payload["client_order_id"] = "manual-other"
        request.write_text(json.dumps(payload), encoding="utf-8")
        assert (
            ExitSourceIndex.load(root).classify(
                broker_order_id="broker-bito-1",
                client_order_id="manual-BITO-test",
                symbol="BITO",
                filled_qty="450",
            )
            == EXIT_SOURCE_UNLINKED
        )


def test_buys_have_no_exit_source_and_column_renders():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_triton_chain(root)
        _write_orders(
            root,
            [
                _order(),
                _order(id="buy-1", side="buy", symbol="MSFT", client_order_id="triton-buy"),
                _order(id="close-1", side="sell_to_close", symbol="INTC"),
            ],
        )
        activity = _build_activity(root=root, start=date(2026, 10, 16), end=date(2026, 10, 16))
    by_id = {row["order_id"]: row for row in activity["broker_orders"]}
    assert "exit_source" not in by_id["buy-1"]
    assert by_id[BROKER]["exit_source"] == EXIT_SOURCE_TRITON
    assert by_id["close-1"]["exit_source"] == EXIT_SOURCE_UNLINKED
    displayed = broker_activity_display_rows(activity["broker_orders"])
    assert all("Exit source" in row for row in displayed)
    buy = next(row for row in displayed if row.get("Symbol") == "MSFT")
    assert buy["Exit source"] == "\u2014"
    sell = next(row for row in displayed if row.get("Symbol") == "SPCX")
    assert sell["Exit source"] == EXIT_SOURCE_TRITON


def test_report_build_does_not_write_files():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_orders(root, [_order(side="buy", symbol="MSFT")])
        (root / "data" / "results" / "trade_outcomes.csv").write_text(
            "symbol,realized_pl\nMSFT,10\n",
            encoding="utf-8",
        )
        before = {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in root.rglob("*")
            if path.is_file()
        }
        report = build_reconciled_report(
            ReportRequest(period_type="Daily", anchor_date=date(2026, 10, 16)),
            root=root,
            now=datetime(2026, 10, 16, 21, 0, tzinfo=timezone.utc),
        )
        after = {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in root.rglob("*")
            if path.is_file()
        }
    assert report["writes_performed"] is False
    assert after == before
    assert (
        "trade_outcomes.csv is a local FIFO derivative"
        in report["metrics"]["realized_pl"]["reason"]
    )
