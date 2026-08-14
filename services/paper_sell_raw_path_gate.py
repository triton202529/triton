"""
paper_sell_raw_path_gate.py
Phase 198I-FIX-CLOSEOUT — fail-closed gate for non-AlpacaBroker sell POST paths.

Any paper sell that reaches a raw REST helper must still hold a hardened
dispatch permit issued after durable consumption.
"""

from __future__ import annotations

from typing import Any, Optional


def gate_raw_paper_sell(
    *,
    side: str,
    symbol: Optional[str] = None,
    qty: Optional[Any] = None,
    source: str = "raw_paper_sell_path",
    mode: str = "paper",
    client_order_id: Optional[str] = None,
) -> None:
    from services.paper_sell_arm_runtime_enforcement import (
        enforce_paper_sell_broker_submission,
    )

    q: Optional[float] = None
    if qty is not None and str(qty) != "":
        try:
            q = float(qty)
        except Exception:
            q = None
    enforce_paper_sell_broker_submission(
        mode=mode,
        side=side,
        source=source,
        symbol=symbol,
        qty=q,
        client_order_id=client_order_id,
    )
