"""
Phase 198I-FIX-CLOSEOUT-EVIDENCE-CORRECTION-5

Broker-route closure: paper SELL cancel is a guarded pure cancellation and
does not require a reviewed-SELL dispatch permit.
"""

from __future__ import annotations

from typing import Any, Dict

from services.broker_alpaca import AlpacaBroker


class _FakeBrokerForCancel(AlpacaBroker):
    """Minimal stand-in: no network; overrides order fetch/delete only."""

    def __init__(self, side: str = "sell", symbol: str = "TSLA", qty: float = 1.0):
        # Do not call AlpacaBroker.__init__ (would require credentials/config).
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self._side = side
        self._symbol = symbol
        self._qty = qty
        self.deleted: list[str] = []
        self.get_calls: list[str] = []

    def get_order(self, order_id: str) -> Dict[str, Any]:
        self.get_calls.append(str(order_id))
        return {
            "id": order_id,
            "side": self._side,
            "symbol": self._symbol,
            "qty": self._qty,
            "status": "open",
        }

    def list_orders(self, status: str = "open", nested: bool = True, **kwargs):
        return [self.get_order("oid-1")]

    def _delete(self, path: str) -> Dict[str, Any]:
        self.deleted.append(path)
        return {"status": "canceled", "path": path}


def test_paper_sell_cancel_blocked_without_permit(monkeypatch):
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    _ACTIVE_PERMIT.set(None)
    b = _FakeBrokerForCancel(side="sell")
    out = b.cancel_order("oid-sell-1")
    assert out.get("status") == "canceled"
    assert b.deleted == ["/v2/orders/oid-sell-1"]
    assert b.get_calls == ["oid-sell-1"]


def test_paper_buy_cancel_not_sell_gated():
    b = _FakeBrokerForCancel(side="buy")
    out = b.cancel_order("oid-buy-1")
    assert out.get("status") == "canceled"
    assert b.deleted == ["/v2/orders/oid-buy-1"]


def test_paper_sell_cancel_all_blocked_without_permit(monkeypatch):
    from services.paper_sell_arm_runtime_enforcement import _ACTIVE_PERMIT

    monkeypatch.setenv("TRITON_ENABLE_LIVE_EXECUTION", "0")
    monkeypatch.setenv("TRITON_LIVE_EXECUTION", "0")
    _ACTIVE_PERMIT.set(None)
    b = _FakeBrokerForCancel(side="sell")
    out = b.cancel_all_orders()
    assert b.deleted == ["/v2/orders"]
    assert out.get("status") == "canceled" or out is not None


def test_raw_cancel_all_script_disabled_fail_closed():
    """Correction-7: raw scripts/cancel_all.py must not bypass cancel gates."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "cancel_all.py").read_text(
        encoding="utf-8"
    )
    assert "DISABLED_FAIL_CLOSED" in src
    assert "raise SystemExit" in src
    assert "AlpacaBroker" in src


def test_app_tradingclient_cancel_all_disabled_fail_closed():
    """Correction-7: app.py Cancel ALL must not invoke TradingClient.cancel_orders."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "DISABLED_FAIL_CLOSED" in src
    assert "c.cancel_orders();print('Canceled ALL open orders')" not in src


def test_opening_sequence_uses_broker_cancel_all():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "opening_sequence.py").read_text(
        encoding="utf-8"
    )
    assert "AlpacaBroker" in src
    assert "cancel_all_orders()" in src
    assert "c.cancel_orders()" not in src
