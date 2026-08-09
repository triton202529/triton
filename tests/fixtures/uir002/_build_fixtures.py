"""One-shot fixture builder for UIR-002 tests (not imported by production)."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    base = ROOT / "complete_daily" / "data" / "results"
    write_csv(
        base / "portfolio_history.csv",
        ["date", "cash", "market_value", "total_value"],
        [
            {"date": "2026-08-06", "cash": "10000", "market_value": "5000", "total_value": "15000"},
            {"date": "2026-08-07", "cash": "10000", "market_value": "5200", "total_value": "15200"},
        ],
    )
    write_csv(
        base / "positions_snapshot.csv",
        ["snapshot_ts", "date", "symbol", "qty", "market_value", "cost_basis", "unrealized_pl"],
        [
            {
                "snapshot_ts": "2026-08-07T20:00:00Z",
                "date": "2026-08-07",
                "symbol": "AAA",
                "qty": "10",
                "market_value": "5200",
                "cost_basis": "5000",
                "unrealized_pl": "200",
            }
        ],
    )
    write_csv(
        base / "recent_orders.csv",
        [
            "id",
            "client_order_id",
            "symbol",
            "qty",
            "filled_qty",
            "filled_avg_price",
            "side",
            "status",
            "submitted_at",
            "filled_at",
            "canceled_at",
        ],
        [
            {
                "id": "o1",
                "client_order_id": "triton-abc",
                "symbol": "AAA",
                "qty": "10",
                "filled_qty": "10",
                "filled_avg_price": "520",
                "side": "buy",
                "status": "filled",
                "submitted_at": "2026-08-07T14:00:00Z",
                "filled_at": "2026-08-07T14:01:00Z",
                "canceled_at": "",
            }
        ],
    )
    write_csv(
        base / "trade_outcomes.csv",
        ["symbol", "outcome_type", "realized_pl", "closed_at", "opened_at"],
        [
            {
                "symbol": "BBB",
                "outcome_type": "REALIZED",
                "realized_pl": "12.5",
                "closed_at": "2026-08-07T15:00:00+00:00",
                "opened_at": "2026-08-01T15:00:00+00:00",
            }
        ],
    )
    write_csv(
        base / "paper_execution_audit.csv",
        ["timestamp", "symbol", "action", "qty", "submitted_order_id", "status"],
        [
            {
                "timestamp": "2026-08-07T14:00:00+00:00",
                "symbol": "AAA",
                "action": "ENTRY",
                "qty": "10",
                "submitted_order_id": "o1",
                "status": "submitted",
            }
        ],
    )

    (ROOT / "no_data" / "data" / "results").mkdir(parents=True, exist_ok=True)

    base = ROOT / "partial_coverage" / "data" / "results"
    write_csv(
        base / "portfolio_history.csv",
        ["date", "cash", "market_value", "total_value"],
        [{"date": "2026-08-01", "cash": "100", "market_value": "100", "total_value": "200"}],
    )

    base = ROOT / "conflict_totals" / "data" / "results"
    write_csv(
        base / "portfolio_history.csv",
        ["date", "cash", "market_value", "total_value"],
        [{"date": "2026-08-07", "cash": "100", "market_value": "100", "total_value": "250"}],
    )

    base = ROOT / "orders_mix" / "data" / "results"
    write_csv(
        base / "portfolio_history.csv",
        ["date", "cash", "market_value", "total_value"],
        [{"date": "2026-08-07", "cash": "1", "market_value": "1", "total_value": "2"}],
    )
    write_csv(
        base / "recent_orders.csv",
        [
            "id",
            "client_order_id",
            "symbol",
            "qty",
            "filled_qty",
            "filled_avg_price",
            "side",
            "status",
            "submitted_at",
            "filled_at",
            "canceled_at",
        ],
        [
            {
                "id": "f1",
                "client_order_id": "triton-f",
                "symbol": "AAA",
                "qty": "10",
                "filled_qty": "10",
                "filled_avg_price": "1",
                "side": "buy",
                "status": "filled",
                "submitted_at": "2026-08-07T14:00:00Z",
                "filled_at": "2026-08-07T14:01:00Z",
                "canceled_at": "",
            },
            {
                "id": "p1",
                "client_order_id": "other",
                "symbol": "BBB",
                "qty": "10",
                "filled_qty": "4",
                "filled_avg_price": "2",
                "side": "buy",
                "status": "partially_filled",
                "submitted_at": "2026-08-07T14:00:00Z",
                "filled_at": "2026-08-07T14:02:00Z",
                "canceled_at": "",
            },
            {
                "id": "c1",
                "client_order_id": "x",
                "symbol": "CCC",
                "qty": "5",
                "filled_qty": "0",
                "filled_avg_price": "",
                "side": "sell",
                "status": "canceled",
                "submitted_at": "2026-08-07T14:00:00Z",
                "filled_at": "",
                "canceled_at": "2026-08-07T14:05:00Z",
            },
            {
                "id": "r1",
                "client_order_id": "y",
                "symbol": "DDD",
                "qty": "5",
                "filled_qty": "0",
                "filled_avg_price": "",
                "side": "buy",
                "status": "rejected",
                "submitted_at": "2026-08-07T14:00:00Z",
                "filled_at": "",
                "canceled_at": "",
            },
            {
                "id": "u1",
                "client_order_id": "broker-only",
                "symbol": "EEE",
                "qty": "1",
                "filled_qty": "1",
                "filled_avg_price": "9",
                "side": "buy",
                "status": "filled",
                "submitted_at": "2026-08-07T14:00:00Z",
                "filled_at": "2026-08-07T14:03:00Z",
                "canceled_at": "",
            },
        ],
    )

    base = ROOT / "stale_sources" / "data" / "results"
    write_csv(
        base / "portfolio_history.csv",
        ["date", "cash", "market_value", "total_value"],
        [{"date": "2026-08-01", "cash": "1", "market_value": "1", "total_value": "2"}],
    )

    # Weekend-dated history row must not become a market session.
    base = ROOT / "weekend_history" / "data" / "results"
    write_csv(
        base / "portfolio_history.csv",
        ["date", "cash", "market_value", "total_value"],
        [
            {"date": "2026-08-07", "cash": "100", "market_value": "100", "total_value": "200"},
            {"date": "2026-08-08", "cash": "100", "market_value": "110", "total_value": "210"},
        ],
    )
    write_csv(
        base / "positions_snapshot.csv",
        ["snapshot_ts", "date", "symbol", "qty", "market_value", "cost_basis", "unrealized_pl"],
        [
            {
                "snapshot_ts": "2026-07-01T12:00:00Z",
                "date": "2026-07-01",
                "symbol": "AAA",
                "qty": "1",
                "market_value": "1",
                "cost_basis": "1",
                "unrealized_pl": "0",
            }
        ],
    )
    print("ok", ROOT)


if __name__ == "__main__":
    main()
