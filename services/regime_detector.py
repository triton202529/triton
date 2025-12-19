import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class RegimeDetector:
    """
    Advanced market regime detection system for Triton.

    Identifies market regimes: Bull, Bear, Sideways, Volatile
    Provides regime-specific risk adjustments and allocation guidance.

    Notes:
      - Expects a DataFrame with at least columns: ['date', 'close']
      - Defensive: will return 'Unknown' when data is insufficient or contains NaNs.
      - Methods are safe to call even if training didn't occur.
    """

    def __init__(self, lookback_days: int = 252, min_samples: int = 50, price_col: str = "close", verbose: bool = False):
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.price_col = price_col
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False
        self.regime_history = []
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime detection features from price data. Defensive handling included."""
        if self.price_col not in df.columns:
            raise ValueError(f"Price column '{self.price_col}' not found in DataFrame")

        features = df.copy().reset_index(drop=True)
        price = features[self.price_col].astype(float)

        # Basic returns
        features['returns'] = price.pct_change(fill_method=None)
        # avoid log(0) and negative issues by only computing when positive and finite
        with np.errstate(divide='ignore', invalid='ignore'):
            features['log_returns'] = np.log(price / price.shift(1))

        # Volatility features (use ddof=0 for population std)
        features['vol_5d'] = features['returns'].rolling(5, min_periods=1).std(ddof=0)
        features['vol_20d'] = features['returns'].rolling(20, min_periods=1).std(ddof=0)
        # avoid division by zero in vol_ratio
        features['vol_ratio'] = features['vol_5d'] / features['vol_20d'].replace({0: np.nan})

        # Trend features
        features['sma_20'] = price.rolling(20, min_periods=1).mean()
        features['sma_50'] = price.rolling(50, min_periods=1).mean()
        # handle divide by zero when sma_50 is zero or NaN
        features['trend_strength'] = (features['sma_20'] - features['sma_50']) / features['sma_50'].replace({0: np.nan})

        # Momentum features
        features['rsi_14'] = self._calculate_rsi(price, 14)
        features['momentum_5d'] = price.pct_change(periods=5)
        features['momentum_20d'] = price.pct_change(periods=20)

        # Volatility clustering
        features['vol_cluster'] = features['vol_5d'].rolling(20, min_periods=1).mean()
        features['vol_regime'] = (features['vol_5d'] > features['vol_cluster'] * 1.5).astype(int)

        # Drawdown features
        features['cummax'] = price.cummax()
        # avoid divide by zero when cummax is 0
        features['drawdown'] = (price - features['cummax']) / features['cummax'].replace({0: np.nan})
        features['max_dd_20d'] = features['drawdown'].rolling(20, min_periods=1).min()

        # It's helpful to drop rows with no price
        features.loc[~np.isfinite(price), :] = np.nan

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator with simple moving average of gains/losses (defensive)."""
        prices = prices.astype(float).copy()
        delta = prices.diff()

        # Gains / losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use rolling mean with min_periods=1 to avoid all-NaN early rows
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()

        # Avoid division by zero; when avg_loss == 0, set RS to large number so RSI -> 100
        rs = avg_gain / avg_loss.replace({0: np.nan})
        rsi = 100 - (100 / (1 + rs))

        # Where avg_loss is zero and avg_gain is zero, RSI is neutral 50
        rsi = rsi.fillna(50.0)

        return rsi

    def _label_regimes(self, features: pd.DataFrame) -> pd.Series:
        """Label market regimes based on price action and volatility.

        This function vectorizes as much logic as reasonable, and uses safe defaults.
        """
        idx = features.index
        regimes = pd.Series(index=idx, dtype='object')

        # required columns - fill missing columns with NaN-safe defaults
        req = ['returns', 'vol_20d', 'trend_strength', 'max_dd_20d']
        for c in req:
            if c not in features.columns:
                features[c] = np.nan

        returns_20d = features['returns'].rolling(20, min_periods=1).mean()
        vol_20d = features['vol_20d']
        trend_strength = features['trend_strength'].fillna(0)
        max_dd = features['max_dd_20d'].fillna(0)

        # Precompute quantiles once
        try:
            high_vol_threshold = vol_20d.quantile(0.8)
            low_vol_threshold = vol_20d.quantile(0.3)
        except Exception:
            high_vol_threshold = np.nan
            low_vol_threshold = np.nan

        # Vectorized decision tree-like rules (in order of priority)
        # Start with default 'Unknown'
        regimes[:] = 'Unknown'

        valid_mask = (~returns_20d.isna()) & (~vol_20d.isna())

        # High volatility
        if np.isfinite(high_vol_threshold):
            mask_volatile = valid_mask & (vol_20d > high_vol_threshold)
            regimes.loc[mask_volatile] = 'Volatile'
        else:
            mask_volatile = pd.Series(False, index=idx)

        # Bear: negative returns & negative trend
        mask_bear = valid_mask & (returns_20d < -0.01) & (trend_strength < -0.05) & (~mask_volatile)
        regimes.loc[mask_bear] = 'Bear'

        # Bull: positive returns & positive trend
        mask_bull = valid_mask & (returns_20d > 0.01) & (trend_strength > 0.05) & (~mask_volatile)
        regimes.loc[mask_bull] = 'Bull'

        # Sideways: low volatility & weak trend
        if np.isfinite(low_vol_threshold):
            mask_sideways = valid_mask & (vol_20d < low_vol_threshold) & (trend_strength.abs() < 0.02) & (~mask_volatile)
            regimes.loc[mask_sideways] = 'Sideways'
        else:
            mask_sideways = pd.Series(False, index=idx)

        # Fallback to trend direction for remaining valid rows
        mask_remaining = valid_mask & (~mask_volatile) & (~mask_bear) & (~mask_bull) & (~mask_sideways)
        regimes.loc[mask_remaining & (trend_strength > 0)] = 'Bull'
        regimes.loc[mask_remaining & (trend_strength <= 0)] = 'Bear'

        # Keep any rows that stayed as 'Unknown' as Unknown
        return regimes

    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """Train the regime detection model.

        Returns self for chaining. If insufficient data, the model will not be fitted.
        """
        self._log("🔍 Training regime detection model...")

        try:
            features = self._calculate_features(df)
        except Exception as e:
            print(f"⚠️ Failed to calculate features for training: {e}")
            return self

        regimes = self._label_regimes(features)

        # Feature columns used for modelling
        feature_cols = [
            'vol_5d', 'vol_20d', 'vol_ratio', 'trend_strength',
            'rsi_14', 'momentum_5d', 'momentum_20d', 'vol_regime',
            'max_dd_20d'
        ]

        # Filter valid rows: all feature columns finite and regime not Unknown
        valid_mask = features[feature_cols].notna().all(axis=1) & (~regimes.isin(['Unknown']))
        X = features.loc[valid_mask, feature_cols]
        y = regimes.loc[valid_mask]

        if len(X) < self.min_samples:
            print(f"⚠️ Insufficient data for training: {len(X)} samples (min required {self.min_samples}). Model not trained.")
            self.is_fitted = False
            return self

        # Scale features (fit)
        try:
            X_scaled = self.scaler.fit_transform(X)
        except Exception as e:
            print(f"⚠️ Failed to scale features: {e}")
            self.is_fitted = False
            return self

        # Train RandomForest
        try:
            self.model.fit(X_scaled, y)
            self.is_fitted = True
        except Exception as e:
            print(f"⚠️ Training failed: {e}")
            self.is_fitted = False
            return self

        self._log(f"✅ Regime model trained on {len(X)} samples")
        try:
            self._log(f"📊 Regime distribution: {y.value_counts().to_dict()}")
        except Exception:
            pass

        return self

    def predict_regime(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Predict current market regime and confidence.

        Returns (regime_label, confidence) where confidence is in [0.0, 1.0].
        If prediction cannot be made, returns ('Unknown', 0.0).
        """
        if not self.is_fitted:
            self._log("Model not fitted — returning Unknown")
            return 'Unknown', 0.0

        try:
            features = self._calculate_features(df)
        except Exception as e:
            self._log(f"Failed to calculate features for prediction: {e}")
            return 'Unknown', 0.0

        feature_cols = [
            'vol_5d', 'vol_20d', 'vol_ratio', 'trend_strength',
            'rsi_14', 'momentum_5d', 'momentum_20d', 'vol_regime',
            'max_dd_20d'
        ]

        latest_features = features[feature_cols].iloc[-1:].copy()

        if latest_features.isna().any().any():
            self._log("Latest features contain NaN — returning Unknown")
            return 'Unknown', 0.0

        try:
            X_scaled = self.scaler.transform(latest_features)
            pred = self.model.predict(X_scaled)
            proba = None
            try:
                proba = self.model.predict_proba(X_scaled)
            except Exception:
                # some classifiers may not support predict_proba
                proba = None

            regime = str(pred[0]) if len(pred) > 0 else 'Unknown'
            confidence = float(proba.max()) if proba is not None else 0.0
            return regime, confidence
        except Exception as e:
            self._log(f"Prediction failed: {e}")
            return 'Unknown', 0.0

    def get_regime_risk_adjustments(self, regime: str) -> Dict[str, float]:
        """Get risk adjustments based on current regime."""
        adjustments = {
            'Bull': {
                'volatility_multiplier': 0.8,
                'position_size_multiplier': 1.2,
                'correlation_adjustment': 0.9,
                'tail_risk_multiplier': 0.7
            },
            'Bear': {
                'volatility_multiplier': 1.5,
                'position_size_multiplier': 0.6,
                'correlation_adjustment': 1.3,
                'tail_risk_multiplier': 2.0
            },
            'Volatile': {
                'volatility_multiplier': 2.0,
                'position_size_multiplier': 0.4,
                'correlation_adjustment': 1.5,
                'tail_risk_multiplier': 2.5
            },
            'Sideways': {
                'volatility_multiplier': 0.6,
                'position_size_multiplier': 1.0,
                'correlation_adjustment': 0.8,
                'tail_risk_multiplier': 0.5
            }
        }

        return adjustments.get(regime, {
            'volatility_multiplier': 1.0,
            'position_size_multiplier': 1.0,
            'correlation_adjustment': 1.0,
            'tail_risk_multiplier': 1.0
        })

    def analyze_regime_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze regime transitions and their impact.

        Returns a DataFrame of transition points with columns:
        ['date', price_col, 'from_regime', 'to_regime', ...]
        """
        try:
            features = self._calculate_features(df)
        except Exception as e:
            print(f"⚠️ Failed to calculate features for transitions: {e}")
            return pd.DataFrame()

        regimes = self._label_regimes(features)
        regime_changes = regimes != regimes.shift(1)
        transition_points = df.loc[regime_changes].copy().reset_index(drop=True)
        transition_points['from_regime'] = regimes.shift(1)[regime_changes].values
        transition_points['to_regime'] = regimes[regime_changes].values

        return transition_points


def main():
    """Test the regime detector (simple CLI test)."""
    import glob

    data_files = glob.glob("data/*.csv")
    if not data_files:
        print("❌ No data files found in data/ — place CSVs and try again.")
        return

    spy_file = next((f for f in data_files if 'SPY' in f.upper()), data_files[0])
    print(f"📊 Using data from: {spy_file}")

    df = pd.read_csv(spy_file)
    # Expect a 'date' column; if present parse it
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.sort_values('date').reset_index(drop=True)

    detector = RegimeDetector(verbose=True)
    detector.fit(df)

    regime, confidence = detector.predict_regime(df)
    print(f"\n🎯 Current Market Regime: {regime} (confidence: {confidence:.2%})")

    adjustments = detector.get_regime_risk_adjustments(regime)
    print(f"📈 Risk Adjustments: {adjustments}")

    transitions = detector.analyze_regime_transitions(df)
    if not transitions.empty:
        print(f"\n🔄 Recent Regime Transitions:")
        to_show = transitions[['date', detector.price_col, 'from_regime', 'to_regime']].tail()
        print(to_show)


if __name__ == "__main__":
    main()


