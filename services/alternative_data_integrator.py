#!/usr/bin/env python3
"""
Alternative Data Integrator for Triton

Integrates multiple alternative data sources:
- Options flow data
- Insider trading (Form 4 filings)
- Social media sentiment
- Satellite imagery (economic activity)
- Credit card data
- Web traffic data
- ESG scores

Institutional hedge funds rely heavily on alternative data for alpha generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests
import warnings

warnings.filterwarnings("ignore")


class AlternativeDataIntegrator:
    """
    Alternative Data Integration Engine.

    Aggregates and normalizes signals from alternative data sources.
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None, verbose: bool = False):
        """
        Initialize alternative data integrator.

        Args:
            api_keys: API keys for data providers
            verbose: Enable verbose logging
        """
        self.api_keys = api_keys or {}
        self.verbose = verbose
        self.data_cache = {}

    def _log(self, *args, **kwargs):
        """Logging helper."""
        if self.verbose:
            print(*args, **kwargs)

    def get_options_flow(self, ticker: str, lookback_days: int = 5) -> Dict[str, float]:
        """
        Get unusual options activity signals.

        Tracks large, unusual options trades that may signal informed trading.

        Returns:
            Signal strength based on options flow
        """
        self._log(f"📊 Fetching options flow for {ticker}")

        # Simulated options flow (in production, connect to data provider)
        # Key metrics:
        # - Put/Call ratio
        # - Large block trades
        # - Unusual volume
        # - OI changes

        signal = {
            "put_call_ratio": np.random.uniform(0.5, 1.5),
            "unusual_volume_score": np.random.uniform(0, 1),
            "block_trades_score": np.random.uniform(0, 1),
            "oi_change_score": np.random.uniform(-1, 1),
            "aggregate_signal": 0,
        }

        # Aggregate signal
        weights = [0.3, 0.3, 0.2, 0.2]
        components = [
            1 - abs(signal["put_call_ratio"] - 1),  # Neutral = 1
            signal["unusual_volume_score"],
            signal["block_trades_score"],
            (signal["oi_change_score"] + 1) / 2,  # Normalize to 0-1
        ]

        signal["aggregate_signal"] = np.average(components, weights=weights)

        return signal

    def get_insider_trading(self, ticker: str, lookback_days: int = 30) -> Dict[str, any]:
        """
        Get insider trading signals from Form 4 filings.

        Insider purchases often signal undervaluation.
        Insider sales can be neutral (liquidity) or bearish.

        Returns:
            Insider trading signal
        """
        self._log(f"👔 Fetching insider trading for {ticker}")

        # Simulated insider data (in production, scrape SEC EDGAR)
        num_buys = np.random.poisson(2)
        num_sells = np.random.poisson(3)
        buy_value = np.random.uniform(0, 5000000) if num_buys > 0 else 0
        sell_value = np.random.uniform(0, 3000000) if num_sells > 0 else 0

        # Net insider sentiment
        net_value = buy_value - sell_value * 0.3  # Weight sells less
        max_value = 5000000
        sentiment = np.clip(net_value / max_value, -1, 1)

        return {
            "num_buys": num_buys,
            "num_sells": num_sells,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "net_sentiment": sentiment,
            "signal_strength": (sentiment + 1) / 2,  # Normalize to 0-1
        }

    def get_social_sentiment(self, ticker: str) -> Dict[str, float]:
        """
        Aggregate social media sentiment.

        Sources: Twitter, Reddit (WSB), StockTwits, etc.

        Returns:
            Social sentiment scores
        """
        self._log(f"💬 Fetching social sentiment for {ticker}")

        # Simulated social sentiment (in production, use APIs)
        twitter_score = np.random.uniform(-1, 1)
        reddit_score = np.random.uniform(-1, 1)
        stocktwits_score = np.random.uniform(-1, 1)

        # Weighted aggregate
        aggregate = np.average(
            [twitter_score, reddit_score, stocktwits_score], weights=[0.4, 0.4, 0.2]
        )

        return {
            "twitter_sentiment": twitter_score,
            "reddit_sentiment": reddit_score,
            "stocktwits_sentiment": stocktwits_score,
            "aggregate_sentiment": aggregate,
            "signal_strength": (aggregate + 1) / 2,  # Normalize to 0-1
        }

    def get_satellite_data(self, ticker: str) -> Dict[str, float]:
        """
        Get satellite-derived economic activity signals.

        Examples:
        - Parking lot traffic (retail)
        - Shipping activity (logistics)
        - Agricultural yields (commodities)
        - Construction activity (real estate)

        Returns:
            Satellite-based signals
        """
        self._log(f"🛰️ Fetching satellite data for {ticker}")

        # Simulated satellite data
        activity_score = np.random.uniform(0, 1)
        trend_score = np.random.uniform(-1, 1)

        return {
            "activity_level": activity_score,
            "activity_trend": trend_score,
            "signal_strength": (activity_score + (trend_score + 1) / 2) / 2,
        }

    def get_web_traffic(self, ticker: str) -> Dict[str, float]:
        """
        Get web traffic and app usage data.

        Useful for consumer companies.

        Returns:
            Web traffic signals
        """
        self._log(f"🌐 Fetching web traffic for {ticker}")

        # Simulated web traffic data
        traffic_growth = np.random.uniform(-0.2, 0.5)
        engagement_score = np.random.uniform(0, 1)

        return {
            "traffic_growth": traffic_growth,
            "engagement_score": engagement_score,
            "signal_strength": (traffic_growth + 0.2) / 0.7 * engagement_score,
        }

    def get_credit_card_data(self, ticker: str) -> Dict[str, float]:
        """
        Get credit card transaction data (if available).

        Real-time revenue proxy for retailers.

        Returns:
            Credit card signals
        """
        self._log(f"💳 Fetching credit card data for {ticker}")

        # Simulated credit card data
        spend_growth = np.random.uniform(-0.1, 0.3)
        transaction_volume_change = np.random.uniform(-0.15, 0.25)

        return {
            "spend_growth": spend_growth,
            "transaction_volume_change": transaction_volume_change,
            "signal_strength": (spend_growth + transaction_volume_change) / 2 + 0.5,
        }

    def get_esg_scores(self, ticker: str) -> Dict[str, float]:
        """
        Get ESG (Environmental, Social, Governance) scores.

        Important for institutional investors with ESG mandates.

        Returns:
            ESG scores
        """
        self._log(f"🌱 Fetching ESG scores for {ticker}")

        # Simulated ESG data
        environmental_score = np.random.uniform(0, 100)
        social_score = np.random.uniform(0, 100)
        governance_score = np.random.uniform(0, 100)

        aggregate_score = np.mean([environmental_score, social_score, governance_score])

        return {
            "environmental_score": environmental_score,
            "social_score": social_score,
            "governance_score": governance_score,
            "aggregate_esg_score": aggregate_score,
            "signal_strength": aggregate_score / 100,
        }

    def aggregate_alternative_signals(self, ticker: str) -> Dict[str, float]:
        """
        Aggregate all alternative data signals for a ticker.

        Returns:
            Combined alternative data signal
        """
        self._log(f"🎯 Aggregating alternative data for {ticker}")

        # Collect all signals
        signals = {
            "options_flow": self.get_options_flow(ticker),
            "insider_trading": self.get_insider_trading(ticker),
            "social_sentiment": self.get_social_sentiment(ticker),
            "satellite_data": self.get_satellite_data(ticker),
            "web_traffic": self.get_web_traffic(ticker),
            "credit_card": self.get_credit_card_data(ticker),
            "esg": self.get_esg_scores(ticker),
        }

        # Extract signal strengths
        signal_strengths = {
            source: data.get("signal_strength", 0.5) for source, data in signals.items()
        }

        # Weighted aggregate (weights based on data reliability)
        weights = {
            "options_flow": 0.20,
            "insider_trading": 0.20,
            "social_sentiment": 0.15,
            "satellite_data": 0.15,
            "web_traffic": 0.10,
            "credit_card": 0.10,
            "esg": 0.10,
        }

        aggregate_signal = sum(
            signal_strengths[source] * weights[source] for source in signal_strengths.keys()
        )

        return {
            "individual_signals": signal_strengths,
            "aggregate_signal": aggregate_signal,
            "confidence": self._calculate_confidence(signals),
            "detailed_data": signals,
        }

    def _calculate_confidence(self, signals: Dict) -> float:
        """Calculate confidence based on signal agreement."""
        strengths = [data.get("signal_strength", 0.5) for data in signals.values()]

        # High confidence when signals agree
        variance = np.var(strengths)
        confidence = 1 / (1 + variance * 10)  # Lower variance = higher confidence

        return confidence


def main():
    """Demo alternative data integrator."""
    print("🎯 Alternative Data Integrator Demo")
    print("=" * 70)

    integrator = AlternativeDataIntegrator(verbose=True)

    # Get alternative data for a ticker
    ticker = "AAPL"
    alt_data = integrator.aggregate_alternative_signals(ticker)

    print(f"\n📊 Alternative Data Summary for {ticker}:")
    print(f"  Aggregate Signal: {alt_data['aggregate_signal']:.3f}")
    print(f"  Confidence: {alt_data['confidence']:.3f}")

    print(f"\n📋 Individual Signals:")
    for source, strength in alt_data["individual_signals"].items():
        print(f"  {source}: {strength:.3f}")

    print("\n✅ Demo completed!")
    print("\n💡 In production, connect to:")
    print("  - Unusual Whales (options flow)")
    print("  - SEC EDGAR (insider trading)")
    print("  - Twitter/Reddit APIs (social sentiment)")
    print("  - Orbital Insight (satellite data)")
    print("  - SimilarWeb (web traffic)")
    print("  - Affinity Solutions (credit card data)")
    print("  - MSCI/Sustainalytics (ESG scores)")


if __name__ == "__main__":
    main()
