"""Reprice must replace only the shares still resting, not the original order size."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from services.manage_open_orders import _unfilled_order_qty
from services.reprice_open_orders import _iter_eligible_stale_buys


def test_unfilled_qty_subtracts_partial_fill():
    assert _unfilled_order_qty({"qty": "20", "filled_qty": "8"}) == 12
    assert _unfilled_order_qty({"qty": 20, "filled_qty": 0}) == 20
    assert _unfilled_order_qty({"qty": 20, "filled_qty": 20}) == 0
    assert _unfilled_order_qty({"qty": 20, "filled_qty": ""}) == 20
    assert _unfilled_order_qty({"qty": 20, "filled_qty": float("nan")}) == 20
    assert _unfilled_order_qty({"quantity": 15, "filled_quantity": 4}) == 11
    # Fractional remainder must not round back up to another share.
    assert _unfilled_order_qty({"qty": 10, "filled_qty": 9.2}) == 0


def test_stale_partial_buy_is_eligible_for_remainder_only():
    now = datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)
    submitted = (now - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = pd.DataFrame(
        [
            {
                "order_id": "partial",
                "symbol": "SPY",
                "side": "buy",
                "status": "partially_filled",
                "qty": 20,
                "filled_qty": 8,
                "limit_price": 500.0,
                "submitted_at": submitted,
            },
            {
                "order_id": "fresh",
                "symbol": "QQQ",
                "side": "buy",
                "status": "new",
                "qty": 5,
                "filled_qty": 0,
                "limit_price": 480.0,
                "submitted_at": submitted,
            },
            {
                "order_id": "sell-stop",
                "symbol": "AAPL",
                "side": "sell",
                "status": "new",
                "qty": 3,
                "filled_qty": 0,
                "limit_price": 200.0,
                "submitted_at": submitted,
            },
        ]
    )
    rows = _iter_eligible_stale_buys(df, now=now, stale_minutes=30, session_map={})
    by_id = {r["order_id"]: r for r in rows}
    assert set(by_id) == {"partial", "fresh"}
    assert by_id["partial"]["qty"] == 12
    assert by_id["fresh"]["qty"] == 5


def test_reprice_modules_submit_unfilled_qty():
    root = Path(__file__).resolve().parents[1]
    for name in ("reprice_open_orders.py", "reprice_order_ladder.py"):
        text = (root / "services" / name).read_text(encoding="utf-8")
        assert "_unfilled_order_qty" in text
        assert 'int(float(r.get("qty")' not in text
