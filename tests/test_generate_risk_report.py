# tests/test_generate_risk_report.py
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on sys.path so 'services' package can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.enhanced_portfolio_manager import EnhancedPortfolioManager


def test_numeric_and_string_positions():
    mgr = EnhancedPortfolioManager()
    mgr.current_positions = {
        "AAPL": {"market_value": 50000},  # numeric in dict
        "SPY": "10000",  # numeric as string
        "BROKEN": "SPY",  # invalid string
    }
    mgr.portfolio_value = 100000
    rpt = mgr._generate_risk_report(pd.Timestamp("2025-01-01"), mgr.portfolio_value)

    assert "current_weights" in rpt
    cw = rpt["current_weights"]
    assert "BROKEN" not in cw
    assert "AAPL" in cw and "SPY" in cw
    total = sum(abs(v) for v in cw.values())
    assert total > 0.999 and total < 1.001


def test_quantity_price_position():
    mgr = EnhancedPortfolioManager()
    mgr.current_positions = {
        "TSLA": {"quantity": 10, "price": 200.0},  # should be 2000
        "MSFT": {"market_value": 1800},  # numeric
    }
    mgr.portfolio_value = 3800
    rpt = mgr._generate_risk_report(pd.Timestamp("2025-01-02"), mgr.portfolio_value)
    cw = rpt["current_weights"]
    assert "TSLA" in cw and "MSFT" in cw
    assert abs(cw["TSLA"] - (2000.0 / (2000.0 + 1800.0))) < 1e-6


def test_zero_or_nan_portfolio_value_guard():
    mgr = EnhancedPortfolioManager()
    mgr.current_positions = {"A": {"market_value": 1000}}
    mgr.portfolio_value = 0  # invalid
    rpt = mgr._generate_risk_report(pd.Timestamp("2025-01-03"), mgr.portfolio_value)
    assert "current_weights" in rpt
    assert isinstance(rpt["current_weights"], dict)


def test_no_positions():
    mgr = EnhancedPortfolioManager()
    mgr.current_positions = {}
    mgr.portfolio_value = 100000
    rpt = mgr._generate_risk_report(pd.Timestamp("2025-01-04"), mgr.portfolio_value)
    assert rpt["num_positions"] == 0
    assert rpt["current_weights"] == {}
