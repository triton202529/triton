#!/usr/bin/env python3
"""
VaR/CVaR Risk Engine for Triton

Implements sophisticated risk metrics used by institutional investors:
- Value-at-Risk (VaR): Maximum expected loss at confidence level
- Conditional Value-at-Risk (CVaR/ES): Expected loss beyond VaR
- Multiple methodologies: Parametric, Historical, Monte Carlo
- Stress testing and scenario analysis
- Portfolio risk decomposition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import warnings
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class VaRCVaREngine:
    """
    Value-at-Risk and Conditional Value-at-Risk Engine.

    Provides institutional-grade risk metrics:
    - VaR: Maximum loss at confidence level
    - CVaR (ES): Expected loss in tail
    - Multiple methods: Parametric, Historical, Monte Carlo
    - Marginal/Component VaR
    - Stress testing
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "historical",
        simulation_runs: int = 10000,
        verbose: bool = False,
    ):
        """
        Initialize VaR/CVaR engine.

        Args:
            confidence_level: Confidence level for VaR (e.g., 0.95 = 95%)
            time_horizon: Time horizon in days
            method: 'parametric', 'historical', or 'monte_carlo'
            simulation_runs: Number of Monte Carlo simulations
            verbose: Enable verbose logging
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.method = method
        self.simulation_runs = simulation_runs
        self.verbose = verbose

        # Risk metrics storage
        self.portfolio_var = None
        self.portfolio_cvar = None
        self.component_var = {}
        self.marginal_var = {}

    def _log(self, *args, **kwargs):
        """Logging helper."""
        if self.verbose:
            print(*args, **kwargs)

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from prices."""
        returns = prices.pct_change().dropna()
        return returns

    def parametric_var(
        self, returns: pd.Series, portfolio_value: float = 1000000
    ) -> Tuple[float, float]:
        """
        Calculate VaR using parametric (variance-covariance) method.

        Assumes returns are normally distributed.
        """
        # Calculate mean and std
        mean_return = returns.mean()
        std_return = returns.std()

        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)

        # VaR calculation
        var_pct = -(mean_return + z_score * std_return) * np.sqrt(self.time_horizon)
        var_dollar = var_pct * portfolio_value

        # CVaR (Expected Shortfall)
        # For normal distribution: CVaR = μ - σ * φ(z) / (1-α)
        phi_z = stats.norm.pdf(z_score)
        cvar_pct = -(mean_return - std_return * phi_z / (1 - self.confidence_level)) * np.sqrt(
            self.time_horizon
        )
        cvar_dollar = cvar_pct * portfolio_value

        return var_dollar, cvar_dollar

    def historical_var(
        self, returns: pd.Series, portfolio_value: float = 1000000
    ) -> Tuple[float, float]:
        """
        Calculate VaR using historical method.

        Uses actual historical distribution (no assumptions).
        """
        # Scale returns for time horizon
        scaled_returns = returns * np.sqrt(self.time_horizon)

        # Sort returns
        sorted_returns = sorted_returns = np.sort(scaled_returns)

        # VaR: percentile of losses
        var_index = int(len(sorted_returns) * (1 - self.confidence_level))
        var_pct = -sorted_returns[var_index]
        var_dollar = var_pct * portfolio_value

        # CVaR: average of losses beyond VaR
        tail_losses = sorted_returns[:var_index]
        cvar_pct = -np.mean(tail_losses) if len(tail_losses) > 0 else var_pct
        cvar_dollar = cvar_pct * portfolio_value

        return var_dollar, cvar_dollar

    def monte_carlo_var(
        self, returns: pd.Series, portfolio_value: float = 1000000
    ) -> Tuple[float, float]:
        """
        Calculate VaR using Monte Carlo simulation.

        Simulates future returns based on historical statistics.
        """
        # Fit distribution parameters
        mean_return = returns.mean()
        std_return = returns.std()

        # Simulate returns
        np.random.seed(42)
        simulated_returns = np.random.normal(
            mean_return * self.time_horizon,
            std_return * np.sqrt(self.time_horizon),
            self.simulation_runs,
        )

        # Sort simulated returns
        sorted_returns = np.sort(simulated_returns)

        # VaR
        var_index = int(len(sorted_returns) * (1 - self.confidence_level))
        var_pct = -sorted_returns[var_index]
        var_dollar = var_pct * portfolio_value

        # CVaR
        tail_losses = sorted_returns[:var_index]
        cvar_pct = -np.mean(tail_losses) if len(tail_losses) > 0 else var_pct
        cvar_dollar = cvar_pct * portfolio_value

        return var_dollar, cvar_dollar

    def calculate_portfolio_var(
        self, returns_df: pd.DataFrame, weights: Dict[str, float], portfolio_value: float = 1000000
    ) -> Dict[str, float]:
        """
        Calculate portfolio VaR and CVaR.

        Args:
            returns_df: DataFrame with returns for each asset
            weights: Portfolio weights {ticker: weight}
            portfolio_value: Total portfolio value

        Returns:
            Dictionary with VaR and CVaR metrics
        """
        self._log(f"📊 Calculating portfolio VaR/CVaR ({self.method} method)...")

        # Filter returns for tickers in portfolio
        portfolio_tickers = [t for t in weights.keys() if t in returns_df.columns]
        portfolio_returns = returns_df[portfolio_tickers]

        # Calculate portfolio returns
        weights_array = np.array([weights.get(t, 0) for t in portfolio_tickers])
        portfolio_return_series = (portfolio_returns * weights_array).sum(axis=1)

        # Calculate VaR/CVaR based on method
        if self.method == "parametric":
            var, cvar = self.parametric_var(portfolio_return_series, portfolio_value)
        elif self.method == "historical":
            var, cvar = self.historical_var(portfolio_return_series, portfolio_value)
        elif self.method == "monte_carlo":
            var, cvar = self.monte_carlo_var(portfolio_return_series, portfolio_value)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Store results
        self.portfolio_var = var
        self.portfolio_cvar = cvar

        # Calculate component VaR
        self.component_var = self._calculate_component_var(
            returns_df, weights, portfolio_tickers, portfolio_value
        )

        results = {
            "var": var,
            "cvar": cvar,
            "var_pct": var / portfolio_value,
            "cvar_pct": cvar / portfolio_value,
            "confidence_level": self.confidence_level,
            "time_horizon": self.time_horizon,
            "method": self.method,
            "portfolio_value": portfolio_value,
            "component_var": self.component_var,
        }

        self._log(f"✅ VaR: ${var:,.2f} ({var/portfolio_value:.2%})")
        self._log(f"✅ CVaR: ${cvar:,.2f} ({cvar/portfolio_value:.2%})")

        return results

    def _calculate_component_var(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float],
        portfolio_tickers: List[str],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """Calculate component VaR (contribution of each position to total VaR)."""
        component_var = {}

        for ticker in portfolio_tickers:
            # Calculate marginal VaR
            weight = weights.get(ticker, 0)

            # Simple approximation: position size * correlation with portfolio
            position_returns = returns_df[ticker]
            weights_array = np.array([weights.get(t, 0) for t in portfolio_tickers])
            portfolio_returns = (returns_df[portfolio_tickers] * weights_array).sum(axis=1)

            correlation = position_returns.corr(portfolio_returns)
            position_var = (
                weight * portfolio_value * self.portfolio_var / portfolio_value
                if self.portfolio_var
                else 0
            )
            component_var[ticker] = position_var * correlation

        return component_var

    def stress_test(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]],
        portfolio_value: float = 1000000,
    ) -> Dict[str, Dict]:
        """
        Run stress tests under different scenarios.

        Args:
            returns_df: Historical returns
            weights: Portfolio weights
            scenarios: Dict of scenarios {name: {ticker: shock}}
            portfolio_value: Portfolio value

        Returns:
            Stress test results
        """
        self._log("🧪 Running stress tests...")

        results = {}

        for scenario_name, shocks in scenarios.items():
            self._log(f"  Testing scenario: {scenario_name}")

            # Apply shocks
            shocked_returns = returns_df.copy()
            for ticker, shock in shocks.items():
                if ticker in shocked_returns.columns:
                    shocked_returns[ticker] = shocked_returns[ticker] + shock

            # Calculate portfolio loss
            portfolio_tickers = [t for t in weights.keys() if t in shocked_returns.columns]
            weights_array = np.array([weights.get(t, 0) for t in portfolio_tickers])
            shocked_portfolio_returns = (shocked_returns[portfolio_tickers] * weights_array).sum(
                axis=1
            )

            # Calculate loss
            expected_loss = shocked_portfolio_returns.mean() * portfolio_value
            worst_case_loss = shocked_portfolio_returns.min() * portfolio_value

            results[scenario_name] = {
                "expected_loss": expected_loss,
                "worst_case_loss": worst_case_loss,
                "expected_loss_pct": expected_loss / portfolio_value,
                "worst_case_loss_pct": worst_case_loss / portfolio_value,
            }

        return results

    def calculate_risk_metrics(
        self, returns_df: pd.DataFrame, weights: Dict[str, float], portfolio_value: float = 1000000
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.

        Returns all risk metrics in one call.
        """
        self._log("📊 Calculating comprehensive risk metrics...")

        # VaR/CVaR
        var_cvar = self.calculate_portfolio_var(returns_df, weights, portfolio_value)

        # Portfolio statistics
        portfolio_tickers = [t for t in weights.keys() if t in returns_df.columns]
        weights_array = np.array([weights.get(t, 0) for t in portfolio_tickers])
        portfolio_returns = (returns_df[portfolio_tickers] * weights_array).sum(axis=1)

        # Additional metrics
        metrics = {
            **var_cvar,
            "expected_return": portfolio_returns.mean() * 252,  # Annualized
            "volatility": portfolio_returns.std() * np.sqrt(252),  # Annualized
            "sharpe_ratio": (
                (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
                if portfolio_returns.std() > 0
                else 0
            ),
            "skewness": portfolio_returns.skew(),
            "kurtosis": portfolio_returns.kurtosis(),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "downside_deviation": self._calculate_downside_deviation(portfolio_returns),
        }

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0) -> float:
        """Calculate downside deviation (semi-deviation)."""
        downside_returns = returns[returns < target]
        if len(downside_returns) > 0:
            return downside_returns.std() * np.sqrt(252)
        return 0

    def optimize_risk_budget(
        self,
        returns_df: pd.DataFrame,
        target_var: float,
        portfolio_value: float = 1000000,
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Optimize portfolio to target VaR level.

        Find optimal weights that achieve target VaR.
        """
        self._log(f"🎯 Optimizing portfolio for target VaR: ${target_var:,.2f}")

        tickers = list(returns_df.columns)
        n_assets = len(tickers)

        # Objective: minimize distance from target VaR
        def objective(weights):
            weights_dict = {ticker: w for ticker, w in zip(tickers, weights)}
            metrics = self.calculate_portfolio_var(returns_df, weights_dict, portfolio_value)
            return abs(metrics["var"] - target_var)

        # Constraints
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # Weights sum to 1

        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess: equal weights
        x0 = np.array([1 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints_list
        )

        if result.success:
            optimal_weights = {ticker: w for ticker, w in zip(tickers, result.x)}
            self._log("✅ Optimization successful")
            return optimal_weights
        else:
            self._log("⚠️ Optimization failed")
            return {ticker: 1 / n_assets for ticker in tickers}

    def generate_risk_report(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float],
        portfolio_value: float = 1000000,
        scenarios: Optional[Dict] = None,
    ) -> Dict:
        """Generate comprehensive risk report."""
        self._log("📋 Generating comprehensive risk report...")

        # Calculate all metrics
        metrics = self.calculate_risk_metrics(returns_df, weights, portfolio_value)

        # Stress test if scenarios provided
        stress_results = {}
        if scenarios:
            stress_results = self.stress_test(returns_df, weights, scenarios, portfolio_value)

        # Build report
        report = {
            "summary": {
                "portfolio_value": portfolio_value,
                "var": metrics["var"],
                "cvar": metrics["cvar"],
                "expected_return": metrics["expected_return"],
                "volatility": metrics["volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
            },
            "detailed_metrics": metrics,
            "stress_tests": stress_results,
            "component_var": metrics.get("component_var", {}),
            "risk_limits": self._check_risk_limits(metrics, portfolio_value),
        }

        return report

    def _check_risk_limits(self, metrics: Dict, portfolio_value: float) -> Dict:
        """Check if risk metrics exceed limits."""
        limits = {
            "var_limit": 0.05,  # 5% of portfolio
            "cvar_limit": 0.10,  # 10% of portfolio
            "volatility_limit": 0.25,  # 25% annualized
            "max_drawdown_limit": -0.20,  # -20%
        }

        breaches = {}

        if metrics["var"] / portfolio_value > limits["var_limit"]:
            breaches["var"] = (
                f"VaR {metrics['var']/portfolio_value:.2%} exceeds limit {limits['var_limit']:.2%}"
            )

        if metrics["cvar"] / portfolio_value > limits["cvar_limit"]:
            breaches["cvar"] = (
                f"CVaR {metrics['cvar']/portfolio_value:.2%} exceeds limit {limits['cvar_limit']:.2%}"
            )

        if metrics["volatility"] > limits["volatility_limit"]:
            breaches["volatility"] = (
                f"Volatility {metrics['volatility']:.2%} exceeds limit {limits['volatility_limit']:.2%}"
            )

        if metrics["max_drawdown"] < limits["max_drawdown_limit"]:
            breaches["max_drawdown"] = (
                f"Max drawdown {metrics['max_drawdown']:.2%} exceeds limit {limits['max_drawdown_limit']:.2%}"
            )

        return {"limits": limits, "breaches": breaches, "status": "BREACHED" if breaches else "OK"}

    def save_report(self, report: Dict, path: str):
        """Save risk report to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self._log(f"💾 Saved risk report to {path}")


def main():
    """Demo the VaR/CVaR engine."""
    print("📊 VaR/CVaR Risk Engine Demo")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")

    # Create returns for multiple assets
    returns_data = {}
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        returns_data[ticker] = np.random.normal(0.0005, 0.02, len(dates))

    returns_df = pd.DataFrame(returns_data, index=dates)

    # Portfolio weights
    weights = {"AAPL": 0.3, "MSFT": 0.25, "GOOGL": 0.2, "AMZN": 0.15, "TSLA": 0.1}
    portfolio_value = 1000000

    # Initialize engine
    engine = VaRCVaREngine(confidence_level=0.95, method="historical", verbose=True)

    # Calculate VaR/CVaR
    metrics = engine.calculate_risk_metrics(returns_df, weights, portfolio_value)

    print(f"\n📊 Risk Metrics:")
    print(f"  VaR (95%): ${metrics['var']:,.2f} ({metrics['var_pct']:.2%})")
    print(f"  CVaR (95%): ${metrics['cvar']:,.2f} ({metrics['cvar_pct']:.2%})")
    print(f"  Expected Return: {metrics['expected_return']:.2%}")
    print(f"  Volatility: {metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")

    # Stress test scenarios
    scenarios = {
        "Market Crash": {
            "AAPL": -0.20,
            "MSFT": -0.18,
            "GOOGL": -0.22,
            "AMZN": -0.25,
            "TSLA": -0.30,
        },
        "Tech Selloff": {
            "AAPL": -0.15,
            "MSFT": -0.15,
            "GOOGL": -0.18,
            "AMZN": -0.12,
            "TSLA": -0.20,
        },
        "Mild Correction": {
            "AAPL": -0.05,
            "MSFT": -0.05,
            "GOOGL": -0.06,
            "AMZN": -0.05,
            "TSLA": -0.08,
        },
    }

    stress_results = engine.stress_test(returns_df, weights, scenarios, portfolio_value)

    print(f"\n🧪 Stress Test Results:")
    for scenario, result in stress_results.items():
        print(f"\n  {scenario}:")
        print(
            f"    Expected Loss: ${result['expected_loss']:,.2f} ({result['expected_loss_pct']:.2%})"
        )
        print(
            f"    Worst Case: ${result['worst_case_loss']:,.2f} ({result['worst_case_loss_pct']:.2%})"
        )

    # Generate full report
    report = engine.generate_risk_report(returns_df, weights, portfolio_value, scenarios)
    engine.save_report(report, "data/results/risk_report_var_cvar.json")

    print("\n✅ Demo completed!")
    print("💾 Risk report saved to data/results/risk_report_var_cvar.json")


if __name__ == "__main__":
    main()
