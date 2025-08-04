# scripts/risk_control.py

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
                position = api.get_position(ticker)
                current_value = float(position.market_value)
                if current_value > 0.10 * buying_power:
                    return False, f"Too much exposure to {ticker} (>10% of buying power)"
            except:
                # No position yet — this is okay
                pass
            return True, "Pass: Safe to BUY"

        elif signal == "SELL":
            try:
                position = api.get_position(ticker)
                if int(float(position.qty)) > 0:
                    return True, "Pass: Safe to SELL"
                else:
                    return False, f"No shares of {ticker} to sell"
            except:
                return False, f"No position found for {ticker} to sell"

    except Exception as e:
        return False, f"Risk check error: {str(e)}"

    return False, "Invalid signal"
