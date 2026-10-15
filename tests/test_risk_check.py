"""Risk checks must block over-exposed buys on the alpaca-py client."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from risk_control import risk_allows, risk_check


class _Account:
    def __init__(self, buying_power="1000"):
        self.buying_power = buying_power


class _Position:
    def __init__(self, market_value="200", qty="1"):
        self.market_value = market_value
        self.qty = qty


class AlpacaPyClient:
    """TradingClient stand-in: get_open_position exists, get_position does not."""

    def __init__(self, position=None, missing=False):
        self.position = position
        self.missing = missing
        self.lookups = []

    def get_account(self):
        return _Account()

    def get_open_position(self, ticker):
        self.lookups.append(ticker)
        if self.missing or self.position is None:
            raise RuntimeError("position does not exist")
        return self.position


class LegacyRestClient:
    def __init__(self, position):
        self.position = position

    def get_account(self):
        return _Account()

    def get_position(self, _ticker):
        return self.position


def test_tuple_denial_is_not_treated_as_allowed():
    client = AlpacaPyClient(_Position(market_value="200", qty="2"))
    result = risk_check("AAPL", "BUY", client)
    assert result  # the bug: a (False, reason) tuple is truthy
    allowed, reason = risk_allows(result)
    assert allowed is False
    assert "AAPL" in reason
    assert client.lookups == ["AAPL"]


def test_buy_blocked_when_position_exceeds_ten_percent_of_buying_power():
    client = AlpacaPyClient(_Position(market_value="150", qty="1"))
    allowed, _reason = risk_allows(risk_check("AAPL", "BUY", client))
    assert allowed is False


def test_buy_allowed_when_position_is_within_cap():
    client = AlpacaPyClient(_Position(market_value="50", qty="1"))
    allowed, _reason = risk_allows(risk_check("AAPL", "BUY", client))
    assert allowed is True


def test_buy_allowed_when_no_position_exists():
    client = AlpacaPyClient(missing=True)
    allowed, _reason = risk_allows(risk_check("AAPL", "BUY", client))
    assert allowed is True


def test_sell_allowed_for_fractional_position():
    client = AlpacaPyClient(_Position(market_value="50", qty="0.5"))
    allowed, _reason = risk_allows(risk_check("SPY", "SELL", client))
    assert allowed is True


def test_sell_blocked_when_no_position():
    client = AlpacaPyClient(missing=True)
    allowed, reason = risk_allows(risk_check("SPY", "SELL", client))
    assert allowed is False
    assert "No position" in reason


def test_legacy_rest_client_still_blocks_overexposed_buy():
    client = LegacyRestClient(_Position(market_value="500", qty="10"))
    allowed, _reason = risk_allows(risk_check("MSFT", "BUY", client))
    assert allowed is False


def test_executor_uses_risk_allows():
    source = Path("scripts/auto_execute_signals.py").read_text(encoding="utf-8")
    assert "if not risk_check(" not in source
    assert "risk_allows(result)" in source
    assert "risk_check(alpaca_symbol, side_text, client)" in source
