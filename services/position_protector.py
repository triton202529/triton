# services/position_protector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from services.broker_alpaca import AlpacaBroker, AlpacaError


Number = Union[int, float]


@dataclass
class ProtectResult:
    ok: bool
    message: str
    order: Optional[Dict[str, Any]] = None


def _round_price(x: Number) -> float:
    return round(float(x), 4)


def place_bracket(
    *,
    broker: AlpacaBroker,
    symbol: str,
    side: str,
    qty: Optional[Number] = None,
    notional: Optional[Number] = None,
    entry_type: str = "market",
    tif: str = "day",
    entry_limit_price: Optional[Number] = None,
    stop_loss_price: Number,
    take_profit_price: Number,
    client_order_id: Optional[str] = None,
) -> ProtectResult:
    """
    Create ONE bracket order (parent + 2 legs) in Alpaca.

    - If qty is provided => qty-based bracket
    - If notional is provided => notional bracket (Alpaca supports notional on parent; legs are price-based)

    IMPORTANT:
    - For entry_type='market', entry_limit_price must be None/ignored.
    - For entry_type='limit', entry_limit_price is required.
    """

    symbol = (symbol or "").upper().strip()
    side = (side or "").lower().strip()
    entry_type = (entry_type or "market").lower().strip()
    tif = (tif or "day").lower().strip()

    if not symbol:
        return ProtectResult(False, "Symbol is required.")
    if side not in ("buy", "sell"):
        return ProtectResult(False, "Side must be buy or sell.")
    if entry_type not in ("market", "limit"):
        return ProtectResult(False, "Entry type must be market or limit.")
    if tif not in ("day", "gtc"):
        return ProtectResult(False, "Time-in-force must be day or gtc.")

    if qty is None and notional is None:
        return ProtectResult(False, "Provide either qty or notional.")
    if qty is not None and notional is not None:
        return ProtectResult(False, "Provide qty OR notional (not both).")

    if entry_type == "limit" and (entry_limit_price is None or float(entry_limit_price) <= 0):
        return ProtectResult(False, "Entry limit price is required for limit entry.")

    sl = float(stop_loss_price)
    tp = float(take_profit_price)
    if sl <= 0 or tp <= 0:
        return ProtectResult(False, "Stop loss and take profit must be > 0.")

    # Basic sanity: for buys, SLP should be below, TP above. For sells, reverse.
    # (We keep it as a safety warning style — but enforce hard because this is a manual risk tool.)
    if side == "buy":
        if not (sl < tp):
            return ProtectResult(False, "For BUY bracket: stop loss must be < take profit.")
    else:
        if not (tp < sl):
            return ProtectResult(False, "For SELL bracket: take profit must be < stop loss.")

    try:
        take_profit = {"limit_price": _round_price(tp)}
        stop_loss = {"stop_price": _round_price(sl)}

        # Qty-based bracket
        if qty is not None:
            q = float(qty)
            if q <= 0:
                return ProtectResult(False, "Qty must be > 0.")

            # Use your generic submit_order because it already supports bracket payloads
            payload = broker.submit_order(
                symbol=symbol,
                qty=q,
                side=side,
                order_type=entry_type,
                time_in_force=tif,
                limit_price=_round_price(entry_limit_price) if entry_type == "limit" else None,
                client_order_id=client_order_id,
                order_class="bracket",
                take_profit=take_profit,
                stop_loss=stop_loss,
                extended_hours=False,  # keep bracket simple/safe
            )
            return ProtectResult(True, "Bracket submitted.", payload)

        # Notional-based bracket (THP-H002 — gated broker API)
        n = float(notional)  # type: ignore[arg-type]
        if n <= 0:
            return ProtectResult(False, "Notional must be > 0.")

        resp = broker.submit_bracket_order_notional(
            symbol=symbol,
            notional=n,
            side=side,
            take_profit=take_profit,
            stop_loss=stop_loss,
            entry_type=entry_type,
            entry_limit_price=_round_price(entry_limit_price) if entry_type == "limit" else None,
            time_in_force=tif,
            client_order_id=client_order_id,
        )
        return ProtectResult(True, "Notional bracket submitted.", resp)

    except AlpacaError as e:
        return ProtectResult(False, f"Alpaca error: {e}")
    except Exception as e:
        return ProtectResult(False, f"Unexpected error: {e}")
