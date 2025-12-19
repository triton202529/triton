#!/usr/bin/env python3
"""
Automated Feature Engineering for Triton

Automatically discovers and engineers predictive features from:
- Price data
- Volume data
- Technical indicators
- Fundamental data
- News sentiment
- Alternative data sources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import product
import warnings

warnings.filterwarnings("ignore")

from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta  # Technical analysis library


class AutomatedFeatureEngineer:
    """
    Automated feature engineering system for Triton.

    Discovers predictive patterns through:
    - Technical indicator engineering
    - Feature interactions
    - Rolling statistics
    - Cross-asset features
    - Regime-aware features
    """

    def __init__(
        self, max_features: int = 50, selection_method: str = "mutual_info", verbose: bool = False
    ):
        self.max_features = max_features
        self.selected_features = []
        self.feature_scores = {}
        self.is_fitted = False
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def engineer_features(self, df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
        """Engineer features from price data."""
        if df is None or df.empty:
            return pd.DataFrame()

        features = df.copy()

        # Ensure numeric columns
        if "close" in features.columns:
            features["close"] = pd.to_numeric(features["close"], errors="coerce")
        if "volume" in features.columns:
            features["volume"] = pd.to_numeric(features["volume"], errors="coerce")
        else:
            features["volume"] = 1000000  # Default

        # Price-based features
        features["returns"] = features["close"].pct_change()
        features["log_returns"] = np.log(features["close"] / features["close"].shift(1))
        features["log_price"] = np.log(features["close"])

        # Rolling statistics
        for window in [5, 10, 20, 50]:
            features[f"ma_{window}"] = features["close"].rolling(window).mean()
            features[f"std_{window}"] = features["returns"].rolling(window).std()
            features[f"max_{window}"] = features["close"].rolling(window).max()
            features[f"min_{window}"] = features["close"].rolling(window).min()

        # Technical indicators (basic implementation)
        self._add_technical_indicators(features)

        # Volume features
        if "volume" in features.columns:
            features["volume_ma_20"] = features["volume"].rolling(20).mean()
            features["volume_ratio"] = features["volume"] / features["volume_ma_20"]
            features["volume_price_trend"] = (
                features["close"] - features["close"].shift(1)
            ) * features["volume"]

        # Volatility features
        features["volatility_5d"] = features["returns"].rolling(5).std()
        features["volatility_20d"] = features["returns"].rolling(20).std()
        features["volatility_ratio"] = features["volatility_5d"] / features["volatility_20d"]

        # Momentum features
        for period in [5, 10, 20]:
            features[f"momentum_{period}"] = features["close"].pct_change(period)
            features[f"roc_{period}"] = (
                features["close"] - features["close"].shift(period)
            ) / features["close"].shift(period)

        # Trend features
        features["ema_12"] = features["close"].ewm(span=12).mean()
        features["ema_26"] = features["close"].ewm(span=26).mean()
        features["macd"] = features["ema_12"] - features["ema_26"]
        features["macd_signal"] = features["macd"].ewm(span=9).mean()

        # Relative strength
        features["rsi"] = self._calculate_rsi(features["close"], 14)

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        features["bb_middle"] = features["close"].rolling(bb_period).mean()
        features["bb_std"] = features["close"].rolling(bb_period).std()
        features["bb_upper"] = features["bb_middle"] + (bb_std * features["bb_std"])
        features["bb_lower"] = features["bb_middle"] - (bb_std * features["bb_std"])
        features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / features["bb_middle"]
        features["bb_position"] = (features["close"] - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"]
        )

        # ATR (Average True Range)
        features["high"] = features["close"].shift(1) * 1.01  # Approximate
        features["low"] = features["close"].shift(1) * 0.99
        features["tr"] = np.maximum(
            features["high"] - features["low"],
            np.maximum(
                np.abs(features["high"] - features["close"].shift(1)),
                np.abs(features["low"] - features["close"].shift(1)),
            ),
        )
        features["atr"] = features["tr"].rolling(14).mean()

        # Drop temporary columns
        features = features.drop(columns=["high", "low", "tr"], errors="ignore")

        return features

    def _add_technical_indicators(self, features: pd.DataFrame):
        """Add technical indicators to features."""
        try:
            # Stochastic
            period = 14
            low_14 = features["close"].rolling(period).min()
            high_14 = features["close"].rolling(period).max()
            features["stoch_k"] = 100 * (features["close"] - low_14) / (high_14 - low_14)
            features["stoch_d"] = features["stoch_k"].rolling(3).mean()
        except Exception:
            pass

        # Simple moving averages
        for period in [9, 12, 26, 50, 200]:
            features[f"sma_{period}"] = features["close"].rolling(period).mean()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info"):
        """Select most important features."""
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Remove infinite and NaN values
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(0)

        y_clean = y.fillna(0)

        if len(X_numeric.columns) == 0 or len(y_clean) == 0:
            self.selected_features = list(X.columns)[: self.max_features]
            return self

        # Select features
        if method == "mutual_info":
            selector = SelectKBest(
                score_func=mutual_info_regression, k=min(self.max_features, len(X_numeric.columns))
            )
        else:
            selector = SelectKBest(
                score_func=f_regression, k=min(self.max_features, len(X_numeric.columns))
            )

        try:
            selector.fit(X_numeric, y_clean)
            selected_idx = selector.get_support(indices=True)
            self.selected_features = [numeric_cols[i] for i in selected_idx]

            # Store scores
            scores = selector.scores_
            for i, col in enumerate(numeric_cols):
                self.feature_scores[col] = scores[i] if i < len(scores) else 0

            self._log(f"✅ Selected {len(self.selected_features)} features")

        except Exception as e:
            self._log(f"⚠️ Feature selection failed: {e}")
            self.selected_features = list(X_numeric.columns)[: self.max_features]

        self.is_fitted = True
        return self

    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_scores

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and selection."""
        # Engineer features
        features = self.engineer_features(df)

        # Select features
        if self.selected_features:
            available_features = [f for f in self.selected_features if f in features.columns]
            features = features[available_features]

        return features


def main():
    """Demo the automated feature engineering."""
    print("🔧 Automated Feature Engineering Demo")
    print("=" * 60)

    # Generate sample data
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        }
    )

    # Engineer features
    engineer = AutomatedFeatureEngineer(verbose=True)
    features = engineer.engineer_features(df)

    print(f"\n✅ Engineered {len(features.columns)} features")
    print(f"📊 Sample features: {list(features.columns[:10])}")

    # Select best features
    y = df["close"].shift(-1)  # Next period price
    engineer.select_features(features, y)

    print(f"\n🎯 Selected {len(engineer.selected_features)} best features")


if __name__ == "__main__":
    main()
