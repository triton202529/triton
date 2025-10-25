# services/generate_risk_report.py

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")


def calculate_performance_metrics(portfolio_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    if len(portfolio_data) < 2:
        return {}

    # Basic metrics
    initial_value = portfolio_data["total_value"].iloc[0]
    final_value = portfolio_data["total_value"].iloc[-1]
    total_return = (final_value - initial_value) / initial_value

    # Daily returns
    portfolio_data["daily_return"] = portfolio_data["total_value"].pct_change()

    # Annualized metrics
    days = (portfolio_data["date"].iloc[-1] - portfolio_data["date"].iloc[0]).days
    annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

    # Volatility
    daily_vol = portfolio_data["daily_return"].std()
    annualized_vol = daily_vol * np.sqrt(252)

    # Sharpe ratio
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    cumulative = portfolio_data["total_value"].cummax()
    drawdown = (portfolio_data["total_value"] - cumulative) / cumulative
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    positive_days = (portfolio_data["daily_return"] > 0).sum()
    total_days = len(portfolio_data["daily_return"].dropna())
    win_rate = positive_days / total_days if total_days > 0 else 0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "total_days": days,
    }


def calculate_regime_metrics(portfolio_data: pd.DataFrame) -> Dict[str, any]:
    """Calculate regime-specific metrics."""
    regime_metrics = {}

    # Regime distribution
    regime_counts = portfolio_data["regime"].value_counts()
    regime_metrics["regime_distribution"] = regime_counts.to_dict()

    # Regime performance
    regime_performance = {}
    for regime in portfolio_data["regime"].unique():
        regime_data = portfolio_data[portfolio_data["regime"] == regime]
        if len(regime_data) > 1:
            regime_return = (
                regime_data["total_value"].iloc[-1] - regime_data["total_value"].iloc[0]
            ) / regime_data["total_value"].iloc[0]
            regime_vol = regime_data["total_value"].pct_change().std() * np.sqrt(252)
            regime_performance[regime] = {
                "return": regime_return,
                "volatility": regime_vol,
                "days": len(regime_data),
            }

    regime_metrics["regime_performance"] = regime_performance

    # Regime transitions
    regime_changes = portfolio_data["regime"] != portfolio_data["regime"].shift(1)
    transition_count = regime_changes.sum()
    regime_metrics["transition_count"] = transition_count
    regime_metrics["transition_frequency"] = (
        transition_count / len(portfolio_data) if len(portfolio_data) > 0 else 0
    )

    return regime_metrics


def calculate_risk_metrics(portfolio_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate risk-specific metrics."""
    if len(portfolio_data) < 20:
        return {}

    # Rolling volatility
    portfolio_data["rolling_vol_20d"] = portfolio_data["total_value"].pct_change().rolling(
        20
    ).std() * np.sqrt(252)

    # VaR calculations
    returns = portfolio_data["total_value"].pct_change().dropna()
    var_95 = returns.quantile(0.05)
    var_99 = returns.quantile(0.01)

    # Expected Shortfall (CVaR)
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()

    # Skewness and Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Tail ratio
    tail_ratio = (
        abs(returns.quantile(0.05)) / abs(returns.quantile(0.95))
        if returns.quantile(0.95) != 0
        else 0
    )

    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "tail_ratio": tail_ratio,
        "current_volatility": (
            portfolio_data["rolling_vol_20d"].iloc[-1]
            if not pd.isna(portfolio_data["rolling_vol_20d"].iloc[-1])
            else 0
        ),
    }


def generate_risk_report():
    """Generate comprehensive risk report."""
    print("📊 Generating comprehensive risk report...")

    # Load portfolio data
    portfolio_file = "data/results/enhanced_portfolio_history.csv"
    if not Path(portfolio_file).exists():
        print(f"❌ Portfolio file not found: {portfolio_file}")
        return

    portfolio_data = pd.read_csv(portfolio_file)
    portfolio_data["date"] = pd.to_datetime(portfolio_data["date"])

    # Load existing risk report if available
    risk_report_file = "data/results/risk_report.json"
    existing_report = {}
    if Path(risk_report_file).exists():
        try:
            with open(risk_report_file, "r") as f:
                existing_report = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading existing risk report: {e}")

    # Calculate metrics
    performance_metrics = calculate_performance_metrics(portfolio_data)
    regime_metrics = calculate_regime_metrics(portfolio_data)
    risk_metrics = calculate_risk_metrics(portfolio_data)

    # Create comprehensive report
    risk_report = {
        "report_date": pd.Timestamp.now().isoformat(),
        "portfolio_summary": {
            "initial_value": portfolio_data["total_value"].iloc[0],
            "final_value": portfolio_data["total_value"].iloc[-1],
            "total_positions": portfolio_data["num_positions"].iloc[-1],
            "current_regime": portfolio_data["regime"].iloc[-1],
            "days_traded": len(portfolio_data),
        },
        "performance_metrics": performance_metrics,
        "regime_metrics": regime_metrics,
        "risk_metrics": risk_metrics,
        "portfolio_history": portfolio_data.to_dict("records"),
    }

    # Merge with existing report data
    if existing_report:
        risk_report.update(existing_report)

    # Save report
    Path("data/results").mkdir(parents=True, exist_ok=True)
    with open(risk_report_file, "w") as f:
        json.dump(risk_report, f, indent=2, default=str)

    # Generate summary
    print("\n📈 Risk Report Summary:")
    print("=" * 50)

    if performance_metrics:
        print(f"Total Return: {performance_metrics['total_return']:.2%}")
        print(f"Annualized Return: {performance_metrics['annualized_return']:.2%}")
        print(f"Annualized Volatility: {performance_metrics['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {performance_metrics['calmar_ratio']:.2f}")
        print(f"Win Rate: {performance_metrics['win_rate']:.2%}")

    if regime_metrics:
        print(f"\nRegime Distribution: {regime_metrics['regime_distribution']}")
        print(f"Regime Transitions: {regime_metrics['transition_count']}")

    if risk_metrics:
        print(f"\nVaR (95%): {risk_metrics['var_95']:.2%}")
        print(f"VaR (99%): {risk_metrics['var_99']:.2%}")
        print(f"Skewness: {risk_metrics['skewness']:.2f}")
        print(f"Kurtosis: {risk_metrics['kurtosis']:.2f}")

    print(f"\n✅ Risk report saved to: {risk_report_file}")


def main():
    """Main function."""
    generate_risk_report()


if __name__ == "__main__":
    main()
