import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


class MultiFactorRiskModel:
    """
    Multi-factor risk model for Triton.

    Decomposes portfolio risk into systematic factors:
    - Market factor (beta)
    - Size factor (small vs large cap)
    - Value factor (value vs growth)
    - Momentum factor
    - Volatility factor
    - Sector factors (if present / derived externally)
    """

    def __init__(self, min_samples: int = 100, verbose: bool = False):
        self.factor_loadings = (
            pd.DataFrame()
        )  # DataFrame: rows = factor columns, cols = PCA factors
        self.factor_returns = pd.DataFrame()  # Time-indexed factor score matrix
        self.idiosyncratic_risk = pd.Series()  # Var per original factor column
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.min_samples = min_samples
        self.verbose = verbose
        self.fitted_factor_cols: List[str] = []  # store feature names used when fitting

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _safe_pct_change(self, series: pd.Series, periods=1):
        return series.pct_change(periods=periods)

    def _calculate_fundamental_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fundamental risk factors with defensive handling."""
        factors = df.copy().reset_index(drop=True)
        if "close" not in factors.columns:
            raise ValueError("DataFrame must include 'close' column.")

        # ensure numeric
        factors["close"] = pd.to_numeric(factors["close"], errors="coerce")
        volume = pd.to_numeric(
            factors.get("volume", pd.Series(np.repeat(1e6, len(factors)))), errors="coerce"
        ).fillna(1e6)

        # Market factor
        factors["market_return"] = self._safe_pct_change(factors["close"])

        # Size factor (proxy)
        # avoid log of zero/negative
        size_proxy = factors["close"] * volume.replace({0: np.nan})
        factors["size_factor"] = np.log(size_proxy.replace({0: np.nan})).replace(
            [np.inf, -np.inf], np.nan
        )

        # Value proxy (inverse momentum over 1 year)
        try:
            factors["value_factor"] = -factors["close"].pct_change(periods=252)
        except Exception:
            factors["value_factor"] = np.nan

        # Momentum
        factors["momentum_factor"] = self._safe_pct_change(factors["close"], periods=20)

        # Volatility factor
        returns = self._safe_pct_change(factors["close"])
        factors["volatility_factor"] = returns.rolling(window=20, min_periods=1).std(ddof=0)

        # Quality (lower vol = higher quality)
        factors["quality_factor"] = -factors["volatility_factor"]

        return factors

    def _calculate_technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical risk factors defensively."""
        factors = df.copy().reset_index(drop=True)
        if "close" not in factors.columns:
            raise ValueError("DataFrame must include 'close' column.")
        close = pd.to_numeric(factors["close"], errors="coerce")

        def roll_mean(offset):
            return close.rolling(offset, min_periods=1).mean()

        # Trend factors
        factors["trend_5d"] = roll_mean(5) / close - 1
        factors["trend_20d"] = roll_mean(20) / close - 1
        factors["trend_50d"] = roll_mean(50) / close - 1

        # Momentum
        factors["momentum_1d"] = self._safe_pct_change(close, 1)
        factors["momentum_5d"] = self._safe_pct_change(close, 5)
        factors["momentum_20d"] = self._safe_pct_change(close, 20)

        # Volatility factors
        pct = self._safe_pct_change(close)
        factors["vol_5d"] = pct.rolling(5, min_periods=1).std(ddof=0)
        factors["vol_20d"] = pct.rolling(20, min_periods=1).std(ddof=0)
        factors["vol_ratio"] = factors["vol_5d"] / factors["vol_20d"].replace({0: np.nan})

        # Mean reversion / RSI
        factors["rsi"] = self._calculate_rsi(close, 14)
        # mean_reversion: standardized distance from 20-day mean
        ma20 = roll_mean(20)
        ma20_std = close.rolling(20, min_periods=1).std(ddof=0).replace({0: np.nan})
        factors["mean_reversion"] = (close - ma20) / ma20_std

        return factors

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Robust RSI calculation (SMA-based, fills neutral where undefined)."""
        prices = pd.to_numeric(prices, errors="coerce").copy()
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace({0: np.nan})
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)
        return rsi

    def fit(self, universe_data: Dict[str, pd.DataFrame]) -> "MultiFactorRiskModel":
        """Fit the multi-factor risk model over a universe of ticker DataFrames."""
        self._log("🔍 Fitting multi-factor risk model...")
        all_factors = []
        tickers = []

        for ticker, df in universe_data.items():
            try:
                if df is None or df.empty or len(df) < 50:
                    continue

                df2 = df.copy().reset_index(drop=True)
                # Ensure close exists and is numeric
                if "close" not in df2.columns:
                    continue

                fundamental_factors = self._calculate_fundamental_factors(df2)
                technical_factors = self._calculate_technical_factors(df2)

                # select columns defensively
                fund_cols = [
                    "market_return",
                    "size_factor",
                    "value_factor",
                    "momentum_factor",
                    "volatility_factor",
                    "quality_factor",
                ]
                tech_cols = [
                    "trend_5d",
                    "trend_20d",
                    "trend_50d",
                    "momentum_1d",
                    "momentum_5d",
                    "momentum_20d",
                    "vol_5d",
                    "vol_20d",
                    "vol_ratio",
                    "rsi",
                    "mean_reversion",
                ]

                fund_df = fundamental_factors[
                    [c for c in fund_cols if c in fundamental_factors.columns]
                ]
                tech_df = technical_factors[
                    [c for c in tech_cols if c in technical_factors.columns]
                ]

                factors = pd.concat([fund_df, tech_df], axis=1)
                # add ticker and date if present
                if "date" in df2.columns:
                    factors["date"] = pd.to_datetime(df2["date"], errors="coerce").values
                factors["ticker"] = ticker

                all_factors.append(factors)
                tickers.append(ticker)
            except Exception as e:
                self._log(f"Skipping {ticker} due to error: {e}")
                continue

        if not all_factors:
            print("❌ No valid data for factor model")
            return self

        combined_factors = pd.concat(all_factors, ignore_index=True, sort=False)

        # Determine factor columns (exclude metadata)
        factor_cols = [c for c in combined_factors.columns if c not in ("ticker", "date")]
        # Drop rows with NaNs in factor cols
        combined_factors = combined_factors.dropna(subset=factor_cols)

        if len(combined_factors) < self.min_samples:
            print(
                f"⚠️ Insufficient data for factor model: {len(combined_factors)} samples (min {self.min_samples})"
            )
            return self

        X = combined_factors[factor_cols].astype(float)
        # Scale features
        try:
            X_scaled = self.scaler.fit_transform(X)
        except Exception as e:
            print(f"⚠️ Scaling failed: {e}")
            return self

        # Fit PCA with reasonable n_components
        n_components = min(10, X.shape[1], X.shape[0])
        try:
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
        except Exception as e:
            print(f"⚠️ PCA fit failed: {e}")
            return self

        # Store factor loadings (index = original factor cols)
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"Factor_{i+1}" for i in range(pca.n_components_)],
            index=factor_cols,
        )
        self.factor_loadings = loadings

        # Factor returns / scores (time-ordered but repeating tickers stacked)
        try:
            factor_scores = pca.transform(X_scaled)
            factor_returns_df = pd.DataFrame(
                factor_scores,
                columns=[f"Factor_{i+1}" for i in range(pca.n_components_)],
                index=combined_factors.index,
            )
            self.factor_returns = factor_returns_df
        except Exception as e:
            self._log(f"Failed to compute factor scores: {e}")
            self.factor_returns = pd.DataFrame()

        # Idiosyncratic risk: variance of residuals per original feature
        try:
            reconstructed = pca.inverse_transform(pca.transform(X_scaled))
            residuals = X_scaled - reconstructed
            idio_var = pd.Series(np.var(residuals, axis=0), index=factor_cols)
            self.idiosyncratic_risk = idio_var
        except Exception as e:
            self._log(f"Failed to compute idiosyncratic risk: {e}")
            self.idiosyncratic_risk = pd.Series(dtype=float)

        self.is_fitted = True
        self.fitted_factor_cols = factor_cols

        explained = getattr(pca, "explained_variance_ratio_", np.array([0.0]))
        print(f"✅ Multi-factor model fitted on {len(combined_factors)} samples")
        print(f"📊 Explained variance (sum): {np.sum(explained):.2%}")

        return self

    def get_factor_exposures(self, ticker_data: pd.DataFrame) -> Dict[str, float]:
        """Get factor exposures for a specific ticker (latest row)."""
        if not self.is_fitted or self.factor_loadings.empty:
            return {}

        try:
            fund = self._calculate_fundamental_factors(ticker_data)
            tech = self._calculate_technical_factors(ticker_data)
            combined = pd.concat([fund, tech], axis=1)

            # Retain only the features that were used when fitting
            cols = [c for c in self.fitted_factor_cols if c in combined.columns]
            if not cols:
                return {}

            latest = combined[cols].iloc[-1:].astype(float).fillna(0.0)
            # scale using the fitted scaler: scaler expects same number/order of columns used during fit.
            # We will align by creating a zero-filled vector for missing columns if any.
            fitted_cols = self.fitted_factor_cols
            x_vec = np.zeros(len(fitted_cols), dtype=float)
            for i, c in enumerate(fitted_cols):
                if c in latest.columns:
                    x_vec[i] = float(latest[c].iloc[0])
                else:
                    x_vec[i] = 0.0

            # scale vector: need 2D
            try:
                x_scaled = self.scaler.transform(x_vec.reshape(1, -1))[0]
            except Exception:
                # fallback: scale by mean/std of scaler if available
                if hasattr(self.scaler, "mean_") and hasattr(self.scaler, "scale_"):
                    x_scaled = (x_vec - self.scaler.mean_) / np.where(
                        self.scaler.scale_ == 0, 1, self.scaler.scale_
                    )
                else:
                    x_scaled = x_vec

            exposures = {}
            for i, factor_name in enumerate(self.factor_loadings.columns):
                # factor_loadings columns length must match x_scaled length
                loading_vec = (
                    self.factor_loadings[factor_name].reindex(index=fitted_cols).fillna(0).values
                )
                exposure = float(np.dot(x_scaled, loading_vec))
                exposures[factor_name] = exposure

            return exposures
        except Exception as e:
            self._log(f"get_factor_exposures error: {e}")
            return {}

    def get_portfolio_risk_decomposition(
        self, portfolio_weights: Dict[str, float], universe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Decompose portfolio risk into factor contributions + idiosyncratic."""
        if not self.is_fitted or self.factor_loadings.empty:
            return {}

        # portfolio exposures per factor
        portfolio_exposures = {f: 0.0 for f in self.factor_loadings.columns}
        idio_total = 0.0

        for ticker, weight in portfolio_weights.items():
            if ticker not in universe_data:
                continue
            exposures = self.get_factor_exposures(universe_data[ticker])
            for f, val in exposures.items():
                portfolio_exposures[f] = portfolio_exposures.get(f, 0.0) + weight * val

            # approximate idiosyncratic risk contribution using fitted idiosyncratic risk
            # sum variances across fitted columns as proxy (if available)
            if not self.idiosyncratic_risk.empty:
                idio_var_sum = float(self.idiosyncratic_risk.sum())
                idio_total += (weight**2) * idio_var_sum

        # factor variances (from factor_returns if available)
        factor_variances = {}
        if not self.factor_returns.empty:
            factor_variances = self.factor_returns.var().to_dict()
        else:
            # fallback: small constant to avoid zeros
            factor_variances = {f: 1e-6 for f in portfolio_exposures.keys()}

        risk_decomposition = {}
        for f, expo in portfolio_exposures.items():
            var = factor_variances.get(f, 0.0)
            risk_contribution = (expo**2) * var
            risk_decomposition[f] = float(risk_contribution)

        risk_decomposition["Idiosyncratic"] = float(idio_total)
        risk_decomposition["Total_Risk"] = float(sum(risk_decomposition.values()))

        return risk_decomposition

    def get_risk_adjusted_weights(
        self,
        signals: Dict[str, float],
        universe_data: Dict[str, pd.DataFrame],
        target_volatility: float = 0.15,
    ) -> Dict[str, float]:
        """Calculate risk-adjusted portfolio weights from raw signals."""
        if not self.is_fitted or self.factor_loadings.empty:
            # If model not fitted, return normalized signals
            total = sum(signals.values()) or 1.0
            return {k: v / total for k, v in signals.items()}

        exposures_map = {}
        for ticker in signals:
            if ticker in universe_data:
                exposures_map[ticker] = self.get_factor_exposures(universe_data[ticker])

        risk_adjusted = {}
        for ticker, signal_strength in signals.items():
            exposures = exposures_map.get(ticker, {})
            expected_var = 0.0
            for factor_name, exposure in exposures.items():
                if factor_name in self.factor_returns.columns:
                    factor_vol = self.factor_returns[factor_name].std()
                    expected_var += (exposure**2) * (float(factor_vol) ** 2)
            expected_vol = np.sqrt(expected_var) if expected_var > 0 else 0.01
            risk_adj = target_volatility / max(expected_vol, 0.01)
            risk_adjusted[ticker] = float(signal_strength * risk_adj)

        total = sum(risk_adjusted.values())
        if total > 0:
            risk_adjusted = {k: v / total for k, v in risk_adjusted.items()}
        return risk_adjusted


def main():
    """Local test harness."""
    import glob

    data_files = glob.glob(str(Path("data").joinpath("*.csv")))
    if not data_files:
        print("❌ No data files found in data/")
        return

    universe_data = {}
    for file in data_files[:10]:
        ticker = Path(file).stem.split("_")[0]
        df = pd.read_csv(file)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        universe_data[ticker] = df

    model = MultiFactorRiskModel(verbose=True)
    model.fit(universe_data)

    sample_signals = {t: float(np.random.random()) for t in list(universe_data.keys())[:5]}
    risk_adjusted_weights = model.get_risk_adjusted_weights(sample_signals, universe_data)

    print(f"\n📊 Sample Signals: {sample_signals}")
    print(f"🎯 Risk-Adjusted Weights: {risk_adjusted_weights}")

    risk_decomp = model.get_portfolio_risk_decomposition(risk_adjusted_weights, universe_data)
    print(f"📈 Risk Decomposition: {risk_decomp}")


if __name__ == "__main__":
    main()
