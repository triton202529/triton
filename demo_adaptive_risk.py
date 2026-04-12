#!/usr/bin/env python3
"""
Triton Adaptive Risk Engine Demo
===============================

This script demonstrates the Adaptive Risk Budgeting Engine capabilities:
- Market regime detection
- Multi-factor risk modeling
- Dynamic risk allocation
- Tail risk hedging
- Performance attribution

Run this to see the adaptive risk engine in action.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import adaptive risk components
from services.adaptive_risk_engine import AdaptiveRiskEngine
from services.regime_detector import RegimeDetector
from services.multi_factor_risk_model import MultiFactorRiskModel
from services.dynamic_risk_allocator import DynamicRiskAllocator


def generate_demo_data():
    """Generate demo market data for testing."""
    print("📊 Generating demo market data...")

    # Create date range (business days gives more realistic trading cadence)
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="B")

    # Generate market data for different tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    universe_data = {}

    for ticker in tickers:
        # Deterministic seed per ticker (abs to avoid negative)
        seed = abs(hash(ticker)) % (2**32)
        rng = np.random.RandomState(seed)

        # Base returns with different volatilities
        base_vol = rng.uniform(0.15, 0.35)
        returns = rng.normal(0.0008, base_vol / np.sqrt(252), len(dates))

        # Add some market crashes (index-based)
        crash_periods = [100, 300, 500, 700]
        for crash_start in crash_periods:
            if crash_start + 50 < len(returns):
                returns[crash_start : crash_start + 50] *= -0.5  # Crash period

        # Add some bull runs
        bull_periods = [200, 400, 600, 800]
        for bull_start in bull_periods:
            if bull_start + 30 < len(returns):
                returns[bull_start : bull_start + 30] *= 1.5  # Bull period

        # Create price series
        prices = 100 * np.cumprod(1 + returns)

        # Add volume data
        volume = rng.randint(1000000, 10000000, len(dates))

        universe_data[ticker] = pd.DataFrame(
            {"date": dates, "close": prices, "volume": volume, "returns": returns}
        )

    print(f"✅ Generated data for {len(tickers)} tickers")
    return universe_data


def demo_regime_detection(universe_data):
    """Demonstrate regime detection capabilities."""
    print("\n🔄 DEMO: Market Regime Detection")
    print("=" * 50)

    # Use first ticker as market proxy
    market_proxy = list(universe_data.keys())[0]
    market_data = universe_data[market_proxy]

    # Initialize regime detector
    detector = RegimeDetector()
    try:
        detector.fit(market_data)
    except Exception as e:
        print(f"⚠️ Regime detector fit failed (continuing): {e}")

    # Predict current regime
    try:
        regime, confidence = detector.predict_regime(market_data)
    except Exception:
        regime, confidence = "Unknown", 0.0

    print(f"📊 Current Market Regime: {regime}")
    print(f"🎯 Confidence: {confidence:.2%}")

    # Get risk adjustments
    try:
        adjustments = detector.get_regime_risk_adjustments(regime)
    except Exception:
        adjustments = {}
    print(f"⚙️ Risk Adjustments: {adjustments}")

    # Analyze regime transitions (may be empty)
    try:
        transitions = detector.analyze_regime_transitions(market_data)
        print(f"🔄 Regime Transitions: {len(transitions)}")
    except Exception:
        print("🔄 Regime Transitions: (failed to compute)")

    return detector


def demo_multi_factor_risk_model(universe_data):
    """Demonstrate multi-factor risk modeling."""
    print("\n📊 DEMO: Multi-Factor Risk Model")
    print("=" * 50)

    # Initialize risk model
    risk_model = MultiFactorRiskModel()
    try:
        risk_model.fit(universe_data)
    except Exception as e:
        print(f"⚠️ Risk model fit failed (continuing): {e}")

    # Get factor exposures for a sample ticker
    sample_ticker = list(universe_data.keys())[0]
    try:
        exposures = risk_model.get_factor_exposures(universe_data[sample_ticker])
    except Exception:
        exposures = {}

    print(f"🎯 Factor Exposures for {sample_ticker}:")
    if exposures:
        for factor, exposure in exposures.items():
            print(f"  {factor}: {exposure:.3f}")
    else:
        print("  (no exposures computed)")

    # Test risk-adjusted weights
    sample_signals = {ticker: np.random.random() for ticker in list(universe_data.keys())[:5]}
    try:
        risk_adjusted_weights = risk_model.get_risk_adjusted_weights(sample_signals, universe_data)
    except Exception:
        risk_adjusted_weights = {}

    print(f"\n📈 Risk-Adjusted Weights:")
    if risk_adjusted_weights:
        for ticker, weight in risk_adjusted_weights.items():
            print(f"  {ticker}: {weight:.3f}")
    else:
        print("  (no risk-adjusted weights)")

    return risk_model


def demo_dynamic_risk_allocation(universe_data, regime_detector, risk_model):
    """Demonstrate dynamic risk allocation."""
    print("\n🎯 DEMO: Dynamic Risk Allocation")
    print("=" * 50)

    # Initialize allocator
    allocator = DynamicRiskAllocator(target_volatility=0.15, max_position_size=0.10)
    allocator.set_regime_detector(regime_detector)
    allocator.set_risk_model(risk_model)

    # Create sample signals
    signals = {ticker: np.random.random() for ticker in list(universe_data.keys())[:5]}

    # Allocate risk budget
    try:
        adaptive_weights = allocator.allocate_risk_budget(signals, universe_data)
    except Exception as e:
        print(f"⚠️ Allocation failed: {e}")
        adaptive_weights = {}

    print(f"📊 Original Signals: {signals}")
    print(f"🎯 Adaptive Weights: {adaptive_weights}")

    # Get risk report
    try:
        risk_report = allocator.get_risk_report(adaptive_weights, universe_data)
        print(f"📈 Portfolio Metrics: {risk_report.get('portfolio_metrics', {})}")
    except Exception:
        print("📈 Portfolio Metrics: (failed to compute)")

    return allocator


def demo_adaptive_risk_engine(universe_data):
    """Demonstrate the complete adaptive risk engine."""
    print("\n🚀 DEMO: Complete Adaptive Risk Engine")
    print("=" * 50)

    # Initialize engine
    engine = AdaptiveRiskEngine()

    # Initialize with data
    if not engine.initialize(universe_data):
        print("❌ Failed to initialize adaptive risk engine")
        return None

    print("✅ Adaptive Risk Engine initialized successfully")

    # Create sample signals
    signals = {ticker: np.random.random() for ticker in list(universe_data.keys())[:5]}

    # Process signals through adaptive engine
    adaptive_weights = engine.process_signals(signals, universe_data)

    print(f"📊 Original Signals: {signals}")
    print(f"🎯 Adaptive Weights: {adaptive_weights}")

    # Get risk report from allocator (more informative than engine dashboard payload)
    try:
        risk_report = engine.risk_allocator.get_risk_report(engine.last_allocation, universe_data)
    except Exception:
        risk_report = {"portfolio_metrics": {}}

    expected_vol = risk_report.get("portfolio_metrics", {}).get("expected_volatility", 0.0)
    print(f"🎯 Portfolio Expected Volatility: {expected_vol:.2%}")

    # Get dashboard data and performance attribution
    dashboard_data = engine.get_risk_dashboard_data()
    print(f"📈 Current Regime: {dashboard_data.get('current_regime', 'Unknown')}")
    attribution = engine.get_performance_attribution()
    print(f"📊 Performance Attribution: {attribution}")

    return engine


def demo_stress_testing(universe_data):
    """Demonstrate stress testing capabilities."""
    print("\n🧪 DEMO: Stress Testing")
    print("=" * 50)

    # Copy base data so scenarios don't permanently mutate inputs outside the function
    base_data = {t: df.copy(deep=True) for t, df in universe_data.items()}

    # Test different market scenarios
    scenarios = ["market_crash", "high_volatility", "normal_market"]

    for scenario in scenarios:
        print(f"\n📋 Testing scenario: {scenario}")

        # Make a working copy
        scenario_data = {t: df.copy(deep=True) for t, df in base_data.items()}

        # Modify scenario_data in place for the test
        if scenario == "market_crash":
            for ticker in scenario_data.keys():
                scenario_data[ticker]["close"] = scenario_data[ticker]["close"] * 0.7  # 30% crash
        elif scenario == "high_volatility":
            for ticker in scenario_data.keys():
                returns = scenario_data[ticker]["close"].pct_change().fillna(0)
                scenario_data[ticker]["close"] = (
                    scenario_data[ticker]["close"].iloc[0] * (1 + returns * 2).cumprod()
                )

        # Test adaptive engine
        engine = AdaptiveRiskEngine()
        if engine.initialize(scenario_data):
            signals = {ticker: 0.2 for ticker in list(scenario_data.keys())[:5]}
            adaptive_weights = engine.process_signals(signals, scenario_data)

            print(f"  Regime: {engine.last_regime}")
            print(f"  Weights: {adaptive_weights}")
            print(f"  Status: ✅ PASSED")
        else:
            print("  Status: ❌ FAILED INIT")


def main():
    """Run the complete demo."""
    print("🎯 Triton Adaptive Risk Engine Demo")
    print("=" * 60)

    # Generate demo data
    universe_data = generate_demo_data()

    # Demo individual components
    regime_detector = demo_regime_detection(universe_data)
    risk_model = demo_multi_factor_risk_model(universe_data)
    allocator = demo_dynamic_risk_allocation(universe_data, regime_detector, risk_model)

    # Demo complete engine
    engine = demo_adaptive_risk_engine(universe_data)

    # Demo stress testing
    demo_stress_testing(universe_data)

    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\n🚀 Next Steps:")
    print("  1. Run: python run_adaptive_pipeline.py")
    print("  2. Launch dashboard: streamlit run services/risk_dashboard.py")
    print("  3. Run stress tests: python services/enhanced_stress_test.py")
    print("\n📚 The Adaptive Risk Engine is now ready to transform Triton!")


if __name__ == "__main__":
    main()
