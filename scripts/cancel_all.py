"""
Cancel-all helper.

DISABLED fail-closed: the previous raw REST cancel-all path bypassed
AlpacaBroker.cancel_all_orders paper-SELL cancel gating.

Operators must use the hardened broker boundary:
  AlpacaBroker(mode="paper").cancel_all_orders()
"""

raise SystemExit(
    "DISABLED_FAIL_CLOSED: scripts/cancel_all.py raw cancel-all is disabled. "
    "Use services.broker_alpaca.AlpacaBroker.cancel_all_orders() so paper SELL "
    "cancels require a hardened dispatch permit."
)
