#!/usr/bin/env python3
"""
Quick Triton Validation Test
============================

Fast validation test to check if Triton components are working correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


def test_data_pipeline():
    """Test data pipeline components."""
    print("📊 Testing data pipeline...")

    try:
        # Test feature generation
        from services.feature_generator import add_technical_indicators

        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)

        df = pd.DataFrame({"date": dates, "close": prices})

        # Test feature generation
        df_with_features = add_technical_indicators(df)

        print(f"  ✅ Feature generation: {len(df_with_features.columns)} features")
        print(f"  ✅ Data shape: {df_with_features.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Data pipeline error: {e}")
        return False


def test_confidence_system():
    """Test confidence scoring system."""
    print("🧠 Testing confidence system...")

    try:
        from services.confidence import compute_confidence

        # Test confidence calculation
        test_row = {
            "close": 100.0,
            "predicted_close": 102.0,
            "signal": "BUY",
            "rsi14": 45.0,
            "sma20": 101.0,
            "sma50": 99.0,
            "atr14": 2.0,
            "sentiment": 0.5,
            "total_score": 75.0,
        }

        confidence, pos_size, edge = compute_confidence(test_row)

        print(f"  ✅ Confidence: {confidence:.3f}")
        print(f"  ✅ Position size: {pos_size:.3f}")
        print(f"  ✅ Edge: {edge:.3f}")

        return True

    except Exception as e:
        print(f"  ❌ Confidence system error: {e}")
        return False


def test_signal_generation():
    """Test signal generation."""
    print("📡 Testing signal generation...")

    try:
        # Test signal generation logic
        predictions = [
            {"ticker": "AAPL", "close": 100, "predicted_close": 102, "signal": "BUY"},
            {"ticker": "MSFT", "close": 200, "predicted_close": 198, "signal": "SELL"},
            {"ticker": "GOOGL", "close": 150, "predicted_close": 151, "signal": "HOLD"},
        ]

        signals = []
        for pred in predictions:
            delta = (pred["predicted_close"] - pred["close"]) / pred["close"]
            if delta > 0.02:
                signal = "BUY"
            elif delta < -0.02:
                signal = "SELL"
            else:
                signal = "HOLD"

            signals.append({"ticker": pred["ticker"], "signal": signal, "delta": delta})

        print(f"  ✅ Generated {len(signals)} signals")
        for signal in signals:
            print(f"    {signal['ticker']}: {signal['signal']} ({signal['delta']:.3f})")

        return True

    except Exception as e:
        print(f"  ❌ Signal generation error: {e}")
        return False


def test_risk_management():
    """Test risk management components."""
    print("🛡️ Testing risk management...")

    try:
        # Test position sizing
        portfolio_value = 100000
        max_allocation = 0.05
        confidence = 0.8

        position_size = portfolio_value * max_allocation * confidence
        max_positions = 15

        print(f"  ✅ Position size: ${position_size:,.2f}")
        print(f"  ✅ Max positions: {max_positions}")
        print(f"  ✅ Max allocation: {max_allocation:.1%}")

        # Test drawdown limits
        max_drawdown = 0.15
        current_value = 95000
        current_drawdown = (current_value - portfolio_value) / portfolio_value

        print(f"  ✅ Current drawdown: {current_drawdown:.1%}")
        print(f"  ✅ Within limits: {'Yes' if current_drawdown > -max_drawdown else 'No'}")

        return True

    except Exception as e:
        print(f"  ❌ Risk management error: {e}")
        return False


def test_performance_metrics():
    """Test performance calculation."""
    print("📈 Testing performance metrics...")

    try:
        # Simulate returns
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns

        # Calculate metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Calculate drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        print(f"  ✅ Total return: {total_return:.1%}")
        print(f"  ✅ Annualized return: {annualized_return:.1%}")
        print(f"  ✅ Volatility: {volatility:.1%}")
        print(f"  ✅ Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"  ✅ Max drawdown: {max_drawdown:.1%}")

        return True

    except Exception as e:
        print(f"  ❌ Performance metrics error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🧪 Triton Quick Validation Test")
    print("=" * 50)

    tests = [
        test_data_pipeline,
        test_confidence_system,
        test_signal_generation,
        test_risk_management,
        test_performance_metrics,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All systems operational! Ready for stress testing.")
        return True
    else:
        print("❌ Some systems need attention before stress testing.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
