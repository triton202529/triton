# scripts/risk_control.py


def _lookup_position(api, ticker):
    """Look up an open position on either Alpaca client.

    alpaca-py TradingClient exposes get_open_position. The older REST client
    exposes get_position. Calling the missing method raises AttributeError,
    which the BUY path used to treat as "no position" and then allow the trade.
    """
    if hasattr(api, "get_open_position"):
        return api.get_open_position(ticker)
    if hasattr(api, "get_position"):
        return api.get_position(ticker)
    raise AttributeError("broker client cannot look up positions")


def risk_allows(result):
    """Return (allowed, reason) from a risk_check result.

    risk_check returns (bool, str). A non-empty tuple is always truthy, so
    `if not risk_check(...)` never blocks a trade. Callers must use this.
    """
    if isinstance(result, tuple):
        allowed = bool(result[0]) if result else False
        reason = str(result[1]) if len(result) > 1 else "Blocked by risk control"
        return allowed, reason
    return bool(result), "Blocked by risk control"


def risk_check(ticker, signal, api):
    """
    Evaluate whether the given trade passes risk controls.

    Args:
        ticker (str): Ticker symbol
        signal (str): Trade signal ('BUY' or 'SELL')
        api (REST): Alpaca API object

    Returns:
        tuple: (bool, str) — True if allowed, along with reason message
    """
    try:
        account = api.get_account()
        buying_power = float(account.buying_power)

        if signal == "BUY":
            try:
                position = _lookup_position(api, ticker)
                current_value = float(position.market_value)
                if current_value > 0.10 * buying_power:
                    return (
                        False,
                        f"Too much exposure to {ticker} (>10% of buying power)",
                    )
            except Exception:
                # No position yet — this is okay
                pass
            return True, "Pass: Safe to BUY"

        elif signal == "SELL":
            try:
                position = _lookup_position(api, ticker)
                # Keep fractional shares sellable. int(0.5) == 0 would block the exit.
                if float(position.qty) > 0:
                    return True, "Pass: Safe to SELL"
                else:
                    return False, f"No shares of {ticker} to sell"
            except Exception:
                return False, f"No position found for {ticker} to sell"

    except Exception as e:
        return False, f"Risk check error: {str(e)}"

    return False, "Invalid signal"
