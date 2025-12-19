import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


class DynamicRiskAllocator:
    """
    Dynamic risk allocation engine for Triton.

    Adjusts portfolio allocation based on:
    - Market regime detection
    - Multi-factor risk model
    - Volatility targeting
    - Correlation breakdown protection
    - Tail risk hedging
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        max_position_size: float = 0.10,
        verbose: bool = False,
    ):
        self.target_volatility = float(target_volatility)
        self.max_position_size = float(max_position_size)
        self.regime_detector = None
        self.risk_model = None
        self.correlation_matrix = pd.DataFrame()
        self.volatility_forecast = {}
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def set_regime_detector(self, regime_detector):
        """Set the regime detector."""
        self.regime_detector = regime_detector

    def set_risk_model(self, risk_model):
        """Set the multi-factor risk model."""
        self.risk_model = risk_model

    def _calculate_correlation_matrix(self, universe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix across tickers using aligned returns.

        Returns a simple pairwise correlation DataFrame (tickers x tickers).
        Falls back to empty DataFrame if insufficient data.
        """
        returns_data = {}

        for ticker, df in universe_data.items():
            try:
                if df is None or len(df) < 20:
                    continue
                if "close" not in df.columns:
                    continue
                series = pd.to_numeric(df["close"], errors="coerce").pct_change().rename(ticker)
                returns_data[ticker] = series
            except Exception as e:
                self._log(f"_calculate_correlation_matrix skip {ticker}: {e}")
                continue

        if not returns_data:
            return pd.DataFrame()

        returns_df = pd.concat(returns_data.values(), axis=1)
        # drop rows with any NaN so correlations are computed on aligned dates
        returns_df = returns_df.dropna(axis=0, how="any")

        if returns_df.shape[0] < 5 or returns_df.shape[1] < 2:
            return pd.DataFrame()

        # Use simple Pearson correlation across the aligned returns (static)
        corr = returns_df.corr()
        return corr

    def _calculate_volatility_forecast(
        self, universe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate volatility forecasts using EWMA-style approach.

        Returns annualized volatility per ticker. Falls back to a small positive number if not computable.
        """
        volatility_forecasts: Dict[str, float] = {}

        for ticker, df in universe_data.items():
            try:
                if df is None or len(df) < 20:
                    continue
                if "close" not in df.columns:
                    continue

                returns = pd.to_numeric(df["close"], errors="coerce").pct_change().dropna()
                if returns.empty:
                    continue

                alpha = 0.06  # RiskMetrics-ish decay; tunable
                # EWMA volatility: use ewm on squared returns then sqrt
                ewma_var = (returns**2).ewm(alpha=alpha).mean().iloc[-1]
                vol_forecast = float(np.sqrt(ewma_var)) if np.isfinite(ewma_var) else np.nan

                # Annualize (if vol_forecast is nan fall back to sample std)
                if not np.isfinite(vol_forecast):
                    vol_forecast = float(returns.std(ddof=0))

                volatility_forecasts[ticker] = max(vol_forecast * np.sqrt(252), 0.0001)
            except Exception as e:
                self._log(f"_calculate_volatility_forecast skip {ticker}: {e}")
                continue

        return volatility_forecasts

    def _calculate_regime_adjustments(self, regime: str) -> Dict[str, float]:
        """Calculate regime-based adjustments (proxy to regime_detector)."""
        if not self.regime_detector:
            return {
                "volatility_multiplier": 1.0,
                "position_size_multiplier": 1.0,
                "tail_risk_multiplier": 1.0,
            }

        try:
            adjustments = self.regime_detector.get_regime_risk_adjustments(regime)
        except Exception as e:
            self._log(f"_calculate_regime_adjustments error: {e}")
            adjustments = {
                "volatility_multiplier": 1.0,
                "position_size_multiplier": 1.0,
                "tail_risk_multiplier": 1.0,
            }
        return adjustments

    def _calculate_correlation_adjustments(
        self, correlation_matrix: pd.DataFrame, portfolio_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate correlation-based position adjustments.

        Higher average correlation reduces allowed position size. Returns mapping ticker -> multiplier.
        """
        adjustments: Dict[str, float] = {}

        if correlation_matrix is None or correlation_matrix.empty:
            return {ticker: 1.0 for ticker in portfolio_weights.keys()}

        # Ensure we have a square DataFrame with tickers as both axes
        for ticker in portfolio_weights.keys():
            if ticker not in correlation_matrix.columns:
                adjustments[ticker] = 1.0
                continue

            other_tickers = [
                t
                for t in portfolio_weights.keys()
                if t != ticker and t in correlation_matrix.columns
            ]
            if not other_tickers:
                adjustments[ticker] = 1.0
                continue

            corrs = []
            for other in other_tickers:
                try:
                    corr = correlation_matrix.at[ticker, other]
                    if pd.notna(corr):
                        corrs.append(abs(float(corr)))
                except Exception:
                    continue

            if corrs:
                avg_corr = float(np.mean(corrs))
                # transform average correlation into an adjustment factor
                # baseline 0.3, scale down linearly; clamp to [0.5, 1.5]
                correlation_adjustment = 1.0 - (avg_corr - 0.3) * 0.5
                correlation_adjustment = float(max(0.5, min(1.5, correlation_adjustment)))
            else:
                correlation_adjustment = 1.0

            adjustments[ticker] = correlation_adjustment

        return adjustments

    def _calculate_volatility_targeting(
        self, signals: Dict[str, float], volatility_forecasts: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate volatility-targeted position sizes (pre-normalization)."""
        volatility_adjusted_weights: Dict[str, float] = {}

        for ticker, signal_strength in signals.items():
            try:
                signal_strength = float(signal_strength)
            except Exception:
                signal_strength = 0.0

            forecast_vol = volatility_forecasts.get(ticker, None)
            if forecast_vol is not None and forecast_vol > 0:
                vol_adjustment = self.target_volatility / max(forecast_vol, 0.01)
                vol_adjustment = min(vol_adjustment, 2.0)
                volatility_adjusted_weights[ticker] = signal_strength * vol_adjustment
            else:
                volatility_adjusted_weights[ticker] = signal_strength

        return volatility_adjusted_weights

    def _calculate_tail_risk_hedging(
        self, portfolio_weights: Dict[str, float], regime: str
    ) -> Dict[str, float]:
        """Calculate tail risk hedging multipliers (reduce positions in high tail-risk regimes)."""
        hedging_adjustments: Dict[str, float] = {}

        regime_adjustments = self._calculate_regime_adjustments(regime)
        tail_risk_multiplier = regime_adjustments.get("tail_risk_multiplier", 1.0)

        if tail_risk_multiplier > 1.5:
            for ticker in portfolio_weights.keys():
                hedging_adjustments[ticker] = float(max(0.0, 1.0 / tail_risk_multiplier))
        else:
            for ticker in portfolio_weights.keys():
                hedging_adjustments[ticker] = 1.0

        return hedging_adjustments

    def allocate_risk_budget(
        self, signals: Dict[str, float], universe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Main risk allocation function. Returns normalized final weights (sum to 1.0)."""
        self._log("🎯 Calculating dynamic risk allocation...")

        # Step 1: Volatility targeting
        volatility_forecasts = self._calculate_volatility_forecast(universe_data)
        vol_adjusted_weights = self._calculate_volatility_targeting(signals, volatility_forecasts)

        # Step 2: Regime-based adjustments
        regime = "Unknown"
        regime_adjustments = {}
        if self.regime_detector:
            try:
                # Prefer SPY as market proxy; otherwise use first available ticker
                market_df = universe_data.get("SPY")
                if market_df is None:
                    market_df = next(iter(universe_data.values()), pd.DataFrame())
                regime, confidence = self.regime_detector.predict_regime(market_df)
            except Exception as e:
                self._log(f"regime detection failed: {e}")
                regime, confidence = "Unknown", 0.0
            regime_adjustments = self._calculate_regime_adjustments(regime)

        # Step 3: Correlation adjustments (static correlation)
        correlation_matrix = self._calculate_correlation_matrix(universe_data)
        correlation_adjustments = self._calculate_correlation_adjustments(
            correlation_matrix, vol_adjusted_weights
        )

        # Step 4: Tail risk hedging
        tail_risk_adjustments = self._calculate_tail_risk_hedging(vol_adjusted_weights, regime)

        # Step 5: Multi-factor risk adjustments (model returns normalized weights)
        factor_adjusted_weights = vol_adjusted_weights.copy()
        if self.risk_model:
            try:
                factor_adjusted_weights = self.risk_model.get_risk_adjusted_weights(
                    vol_adjusted_weights, universe_data, self.target_volatility
                )
                # if model returns empty or invalid, fall back
                if not isinstance(factor_adjusted_weights, dict) or not factor_adjusted_weights:
                    factor_adjusted_weights = vol_adjusted_weights.copy()
            except Exception as e:
                self._log(f"risk_model adjustment failed: {e}")
                factor_adjusted_weights = vol_adjusted_weights.copy()

        # Step 6: Combine adjustments into final pre-normalized weights
        final_weights: Dict[str, float] = {}
        for ticker in signals.keys():
            if ticker not in vol_adjusted_weights:
                continue
            weight = float(vol_adjusted_weights.get(ticker, 0.0))

            # Apply regime multiplier
            regime_multiplier = float(regime_adjustments.get("position_size_multiplier", 1.0))
            weight *= regime_multiplier

            # Correlation adjustment
            weight *= float(correlation_adjustments.get(ticker, 1.0))

            # Tail risk adjustments
            weight *= float(tail_risk_adjustments.get(ticker, 1.0))

            # Replace with factor-adjusted weight if provided
            if ticker in factor_adjusted_weights:
                try:
                    w_fac = float(factor_adjusted_weights[ticker])
                    # Mix factors: prefer model output but keep regime/corr/tail scaling applied
                    # We'll combine by taking model weight * (regime * corr * tail)
                    weight = (
                        w_fac
                        * regime_multiplier
                        * float(correlation_adjustments.get(ticker, 1.0))
                        * float(tail_risk_adjustments.get(ticker, 1.0))
                    )
                except Exception:
                    pass

            # Cap individual position size
            weight = min(weight, self.max_position_size)
            final_weights[ticker] = max(weight, 0.0)

        # Step 7: Normalize final weights (sum to 1)
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        else:
            # If zero, return uniform small weights proportional to signals
            positive_signals = {k: max(0.0, float(v)) for k, v in signals.items()}
            s = sum(positive_signals.values()) or 1.0
            uniform = {k: (positive_signals[k] / s) * 0.001 for k in positive_signals.keys()}
            final_weights = uniform

        # Step 8: Calculate portfolio risk metrics (for printing/logging)
        portfolio_metrics = self._calculate_portfolio_metrics(final_weights, universe_data, regime)

        self._log(f"📊 Regime: {regime}")
        self._log(
            f"🎯 Portfolio Volatility (expected): {portfolio_metrics.get('expected_volatility', 0):.2%}"
        )
        self._log(
            f"📈 Risk-Adjusted Return: {portfolio_metrics.get('risk_adjusted_return', 0):.2%}"
        )

        return final_weights

    def _calculate_portfolio_metrics(
        self, weights: Dict[str, float], universe_data: Dict[str, pd.DataFrame], regime: str
    ) -> Dict[str, float]:
        """Calculate portfolio-level metrics: expected volatility, risk-adjusted return, diversification ratio."""
        metrics: Dict[str, float] = {}
        expected_var = 0.0
        individual_vols: List[float] = []

        for ticker, weight in weights.items():
            try:
                if ticker not in universe_data:
                    continue
                df = universe_data[ticker]
                if df is None or "close" not in df.columns:
                    continue
                returns = pd.to_numeric(df["close"], errors="coerce").pct_change().dropna()
                if len(returns) < 2:
                    continue
                vol = float(returns.std(ddof=0) * np.sqrt(252))
                expected_var += (weight**2) * (vol**2)
                individual_vols.append(weight * vol)
            except Exception as e:
                self._log(f"_calculate_portfolio_metrics skip {ticker}: {e}")
                continue

        expected_vol = float(np.sqrt(expected_var)) if expected_var > 0 else 0.0
        metrics["expected_volatility"] = expected_vol

        # Risk-adjusted return (naive proxy: inverse of volatility scaled by regime)
        if self.regime_detector:
            regime_adjustments = self._calculate_regime_adjustments(regime)
            volatility_multiplier = float(regime_adjustments.get("volatility_multiplier", 1.0))
            risk_adjusted_return = expected_vol / max(volatility_multiplier, 0.0001)
        else:
            risk_adjusted_return = expected_vol

        metrics["risk_adjusted_return"] = risk_adjusted_return

        # Diversification ratio (avg individual vol / portfolio vol)
        if individual_vols and expected_vol > 0:
            avg_individual_vol = float(np.mean(individual_vols))
            diversification_ratio = avg_individual_vol / expected_vol if expected_vol > 0 else 1.0
            metrics["diversification_ratio"] = diversification_ratio
        else:
            metrics["diversification_ratio"] = 1.0

        return metrics

    def get_risk_report(
        self, weights: Dict[str, float], universe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, any]:
        """Generate a comprehensive risk report for the current weights and universe."""
        report: Dict[str, any] = {}

        regime = "Unknown"
        confidence = 0.0
        if self.regime_detector:
            try:
                market_df = universe_data.get("SPY")
                if market_df is None:
                    market_df = next(iter(universe_data.values()), pd.DataFrame())
                regime, confidence = self.regime_detector.predict_regime(market_df)
            except Exception as e:
                self._log(f"get_risk_report regime detection failed: {e}")
                regime, confidence = "Unknown", 0.0

        portfolio_metrics = self._calculate_portfolio_metrics(weights, universe_data, regime)
        report["portfolio_metrics"] = portfolio_metrics

        report["regime"] = {
            "current_regime": regime,
            "confidence": float(confidence),
            "regime_adjustments": self._calculate_regime_adjustments(regime),
        }

        if self.risk_model:
            try:
                risk_decomposition = self.risk_model.get_portfolio_risk_decomposition(
                    weights, universe_data
                )
                report["risk_decomposition"] = risk_decomposition
            except Exception as e:
                self._log(f"get_risk_report decomposition failed: {e}")
                report["risk_decomposition"] = {}

        position_analysis: Dict[str, Dict[str, float]] = {}
        for ticker, weight in weights.items():
            try:
                if ticker not in universe_data:
                    continue
                df = universe_data[ticker]
                if df is None or "close" not in df.columns:
                    continue
                returns = pd.to_numeric(df["close"], errors="coerce").pct_change().dropna()
                if len(returns) < 2:
                    continue
                vol = float(returns.std(ddof=0) * np.sqrt(252))
                position_analysis[ticker] = {
                    "weight": float(weight),
                    "volatility": vol,
                    "risk_contribution": float(weight * vol),
                }
            except Exception as e:
                self._log(f"get_risk_report position skip {ticker}: {e}")
                continue

        report["position_analysis"] = position_analysis

        return report


def main():
    """Local test harness for DynamicRiskAllocator."""
    import glob

    data_files = glob.glob(str(Path("data").joinpath("*.csv")))
    if not data_files:
        print("❌ No data files found in data/")
        return

    universe_data: Dict[str, pd.DataFrame] = {}
    for file in data_files[:10]:
        ticker = Path(file).stem.split("_")[0]
        df = pd.read_csv(file)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        universe_data[ticker] = df

    allocator = DynamicRiskAllocator(target_volatility=0.15, max_position_size=0.10, verbose=True)

    # attach detectors/models if available (no-op if not)
    # allocator.set_regime_detector(my_regime_detector)
    # allocator.set_risk_model(my_risk_model)

    sample_signals = {
        ticker: float(np.random.random()) for ticker in list(universe_data.keys())[:5]
    }

    risk_adjusted_weights = allocator.allocate_risk_budget(sample_signals, universe_data)
    print(f"\n📊 Sample Signals: {sample_signals}")
    print(f"🎯 Risk-Adjusted Weights: {risk_adjusted_weights}")

    risk_report = allocator.get_risk_report(risk_adjusted_weights, universe_data)
    print(f"📈 Risk Report (summary): {risk_report.get('portfolio_metrics', {})}")


if __name__ == "__main__":
    main()
