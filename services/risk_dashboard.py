import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Try to import AdaptiveRiskEngine, but degrade gracefully if not available.
try:
    from .adaptive_risk_engine import AdaptiveRiskEngine
except Exception as e:
    AdaptiveRiskEngine = None
    _adaptive_import_error = str(e)


class RiskDashboard:
    """
    Interactive Risk Dashboard for Triton's Adaptive Risk Engine.

    Provides real-time visualization of:
    - Market regime detection
    - Portfolio risk metrics
    - Factor exposures
    - Performance attribution
    - Risk decomposition
    """

    def __init__(self):
        if AdaptiveRiskEngine is not None:
            try:
                self.adaptive_risk_engine = AdaptiveRiskEngine()
            except Exception:
                # If engine construction fails, keep as None and continue.
                self.adaptive_risk_engine = None
        else:
            self.adaptive_risk_engine = None

        self.portfolio_data: Optional[pd.DataFrame] = None
        self.risk_report: Optional[Dict] = None

    def load_data(self):
        """Load portfolio and risk data."""
        # Load portfolio history
        portfolio_file = Path("data/results/enhanced_portfolio_history.csv")
        if portfolio_file.exists():
            try:
                df = pd.read_csv(portfolio_file)
                # defensive: ensure date column exists
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                else:
                    st.warning("`date` column missing from portfolio file — attempting to proceed.")
                self.portfolio_data = df.sort_values("date").reset_index(drop=True)
            except Exception as e:
                st.error(f"Failed to read portfolio file: {e}")
                self.portfolio_data = None
        else:
            self.portfolio_data = None

        # Load risk report
        risk_report_file = Path("data/results/risk_report.json")
        if risk_report_file.exists():
            try:
                with open(risk_report_file, "r") as f:
                    self.risk_report = json.load(f)
            except Exception as e:
                st.error(f"Failed to read risk report file: {e}")
                self.risk_report = None
        else:
            self.risk_report = None

    def render_dashboard(self):
        """Render the main dashboard."""
        st.set_page_config(page_title="Triton Risk Dashboard", page_icon="📊", layout="wide")

        st.title("🎯 Triton Adaptive Risk Dashboard")
        st.markdown("---")

        # If adaptive engine import failed, show a note (non-blocking)
        if AdaptiveRiskEngine is None:
            st.info(
                "AdaptiveRiskEngine not available in the environment. Dashboard will still render with saved data."
            )
            if "_adaptive_import_error" in globals():
                st.caption(f"Adaptive engine import error: {_adaptive_import_error}")

        # Load data
        self.load_data()

        if self.portfolio_data is None or self.portfolio_data.empty:
            st.error(
                "❌ No portfolio data found. Run enhanced portfolio manager first or place `data/results/enhanced_portfolio_history.csv`."
            )
            return

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📈 Portfolio Overview",
                "🎯 Risk Metrics",
                "🔄 Regime Analysis",
                "📊 Factor Exposure",
                "⚙️ Risk Controls",
            ]
        )

        with tab1:
            self._render_portfolio_overview()

        with tab2:
            self._render_risk_metrics()

        with tab3:
            self._render_regime_analysis()

        with tab4:
            self._render_factor_exposure()

        with tab5:
            self._render_risk_controls()

    def _render_portfolio_overview(self):
        """Render portfolio overview tab."""
        st.header("📈 Portfolio Overview")

        df = self.portfolio_data

        # Ensure columns exist. Provide sensible defaults if they're missing.
        required_cols = ["total_value", "market_value", "cash", "num_positions", "regime"]
        for c in required_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            try:
                current_value = float(df["total_value"].dropna().iloc[-1])
                st.metric("Portfolio Value", f"${current_value:,.2f}")
            except Exception:
                current_value = (
                    float(df["total_value"].dropna().iloc[-1])
                    if not df["total_value"].dropna().empty
                    else 0.0
                )
                st.metric("Portfolio Value", f"${current_value:,.2f}")

        with col2:
            # compute total return compared to first non-null total_value
            try:
                first_value = float(df["total_value"].dropna().iloc[0])
                if first_value != 0:
                    total_return = (current_value / first_value - 1) * 100
                else:
                    total_return = 0.0
            except Exception:
                total_return = 0.0
            st.metric("Total Return", f"{total_return:.2f}%")

        with col3:
            try:
                current_positions = int(df["num_positions"].dropna().iloc[-1])
            except Exception:
                current_positions = (
                    int(df["num_positions"].dropna().iloc[-1])
                    if not df["num_positions"].dropna().empty
                    else 0
                )
            st.metric("Active Positions", current_positions)

        with col4:
            try:
                current_regime = df["regime"].dropna().iloc[-1]
            except Exception:
                current_regime = "Unknown"
            st.metric("Market Regime", current_regime)

        # Portfolio value chart
        fig = go.Figure()

        # Plot only if numeric data exists, otherwise skip that trace
        if df["total_value"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["total_value"],
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(width=2),
                )
            )

        if "market_value" in df.columns and df["market_value"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["market_value"],
                    mode="lines",
                    name="Market Value",
                    line=dict(width=2),
                )
            )

        if "cash" in df.columns and df["cash"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["date"], y=df["cash"], mode="lines", name="Cash", line=dict(width=2)
                )
            )

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

        # Regime timeline (scatter colored by regime). If regime missing, show a simple line.
        if "regime" in df.columns and df["regime"].notna().any():
            fig_regime = px.scatter(
                df,
                x="date",
                y="total_value",
                color="regime",
                title="Portfolio Value by Market Regime",
                labels={"total_value": "Portfolio Value ($)", "date": "Date"},
            )
            st.plotly_chart(fig_regime, width="stretch")
        else:
            st.info("No regime labels present — showing portfolio value line only.")
            fig_line = px.line(df, x="date", y="total_value", title="Portfolio Value")
            st.plotly_chart(fig_line, width="stretch")

    def _render_risk_metrics(self):
        """Render risk metrics tab."""
        st.header("🎯 Risk Metrics")

        if self.risk_report is None:
            st.warning("⚠️ No risk report data available")
            return

        # Risk metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            expected_vol = self.risk_report.get("portfolio_metrics", {}).get(
                "expected_volatility", None
            )
            if expected_vol is None:
                st.metric("Expected Volatility", "N/A")
            else:
                # expected_vol assumed to be decimal (e.g., 0.12)
                try:
                    st.metric("Expected Volatility", f"{expected_vol:.2%}")
                except Exception:
                    st.metric("Expected Volatility", str(expected_vol))

        with col2:
            diversification = self.risk_report.get("portfolio_metrics", {}).get(
                "diversification_ratio", None
            )
            if diversification is None:
                st.metric("Diversification Ratio", "N/A")
            else:
                st.metric("Diversification Ratio", f"{diversification:.2f}")

        with col3:
            risk_adjusted_return = self.risk_report.get("portfolio_metrics", {}).get(
                "risk_adjusted_return", None
            )
            if risk_adjusted_return is None:
                st.metric("Risk-Adjusted Return", "N/A")
            else:
                try:
                    st.metric("Risk-Adjusted Return", f"{risk_adjusted_return:.2%}")
                except Exception:
                    st.metric("Risk-Adjusted Return", str(risk_adjusted_return))

        # Risk decomposition
        risk_decomp = self.risk_report.get("risk_decomposition", {})
        if risk_decomp:
            st.subheader("📊 Risk Decomposition")

            # Create pie chart
            labels = list(risk_decomp.keys())
            values = [float(v) if v is not None else 0.0 for v in risk_decomp.values()]

            # handle case where all zeros
            if sum(values) <= 0:
                st.info("Risk decomposition contains no positive contributions.")
            else:
                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                fig.update_layout(title="Portfolio Risk Decomposition")
                st.plotly_chart(fig, width="stretch")

            # Risk table
            risk_df = pd.DataFrame(
                [
                    {"Factor": factor, "Risk Contribution": risk}
                    for factor, risk in risk_decomp.items()
                ]
            )
            st.dataframe(risk_df, width="stretch")

    def _render_regime_analysis(self):
        """Render regime analysis tab."""
        st.header("🔄 Market Regime Analysis")

        df = self.portfolio_data

        if "regime" not in df.columns or df["regime"].isna().all():
            st.info("No regime data available for analysis.")
            return

        # Regime distribution
        regime_counts = df["regime"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=regime_counts.values, names=regime_counts.index, title="Regime Distribution"
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            # Regime performance
            regime_performance = (
                df.groupby("regime")["total_value"]
                .agg([("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")])
                .round(2)
            )

            st.subheader("Regime Performance")
            st.dataframe(regime_performance, width="stretch")

        # Regime transitions
        st.subheader("🔄 Regime Transitions")

        regime_changes = df["regime"] != df["regime"].shift(1)
        transition_points = df[regime_changes].copy()

        if not transition_points.empty:
            fig = px.scatter(
                transition_points,
                x="date",
                y="total_value",
                color="regime",
                title="Regime Transition Points",
                labels={"total_value": "Portfolio Value ($)", "date": "Date"},
            )
            st.plotly_chart(fig, width="stretch")

            st.dataframe(transition_points[["date", "regime", "total_value"]], width="stretch")
        else:
            st.info("No regime transitions detected")

    def _render_factor_exposure(self):
        """Render factor exposure tab."""
        st.header("📊 Factor Exposure Analysis")

        if self.risk_report is None:
            st.warning("⚠️ No factor exposure data available")
            return

        # Position analysis
        position_analysis = self.risk_report.get("position_analysis", {})

        if position_analysis:
            st.subheader("Position Analysis")

            # Create position table with numeric values for plotting
            position_data = []
            for ticker, data in position_analysis.items():
                # Defensive extraction with defaults
                weight = data.get("weight", None)
                volatility = data.get("volatility", None)
                risk_contribution = data.get("risk_contribution", None)

                # Try to coerce to numeric if possible
                try:
                    weight_num = float(weight) if weight is not None else np.nan
                except Exception:
                    weight_num = np.nan
                try:
                    vol_num = float(volatility) if volatility is not None else np.nan
                except Exception:
                    vol_num = np.nan
                try:
                    rc_num = float(risk_contribution) if risk_contribution is not None else np.nan
                except Exception:
                    rc_num = np.nan

                position_data.append(
                    {
                        "Ticker": ticker,
                        "Weight": weight_num,
                        "Volatility": vol_num,
                        "Risk Contribution": rc_num,
                    }
                )

            position_df = pd.DataFrame(position_data)

            # Format a display-friendly table while keeping numeric columns for charts
            display_df = position_df.copy()
            display_df["Weight"] = display_df["Weight"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
            display_df["Volatility"] = display_df["Volatility"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
            display_df["Risk Contribution"] = display_df["Risk Contribution"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )

            st.dataframe(display_df, width="stretch")

            # Risk contribution chart (use numeric 'Risk Contribution' column)
            if position_df["Risk Contribution"].notna().any():
                fig = px.bar(
                    position_df.sort_values("Risk Contribution", ascending=False),
                    x="Ticker",
                    y="Risk Contribution",
                    title="Risk Contribution by Position",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No numeric risk contribution data available for plotting.")

        # Factor weights
        factor_weights = self.risk_report.get("factor_weights", {})
        if factor_weights:
            st.subheader("Factor Weights")
            keys = list(factor_weights.keys())
            vals = []
            for v in factor_weights.values():
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
            factor_df = pd.DataFrame({"Factor": keys, "Weight": vals})
            st.dataframe(factor_df, width="stretch")

            if factor_df["Weight"].notna().any():
                fig = px.bar(
                    factor_df.sort_values("Weight", ascending=False),
                    x="Factor",
                    y="Weight",
                    title="Factor Weights",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No numeric factor weights available for plotting.")

    def _render_risk_controls(self):
        """Render risk controls tab."""
        st.header("⚙️ Risk Controls")

        if self.risk_report is None:
            st.warning("⚠️ No risk controls data available")
            return

        # Risk limits
        risk_limits = self.risk_report.get("risk_limits", {})
        if risk_limits:
            st.subheader("Risk Limits")

            limit_data = []
            for limit, value in risk_limits.items():
                display_value = f"{value:.2%}" if isinstance(value, float) else str(value)
                limit_data.append(
                    {"Limit": limit.replace("_", " ").title(), "Value": display_value}
                )

            limit_df = pd.DataFrame(limit_data)
            st.dataframe(limit_df, width="stretch")

        # Regime adjustments
        regime_adjustments = self.risk_report.get("regime_adjustments", {})
        if regime_adjustments:
            st.subheader("Regime Adjustments")

            adjustment_data = []
            for regime, adjustments in regime_adjustments.items():
                for adjustment, value in adjustments.items():
                    adjustment_data.append(
                        {
                            "Regime": regime,
                            "Adjustment": adjustment.replace("_", " ").title(),
                            "Value": f"{value:.2f}" if isinstance(value, float) else str(value),
                        }
                    )

            adjustment_df = pd.DataFrame(adjustment_data)
            st.dataframe(adjustment_df, width="stretch")

        # Performance attribution
        performance_attribution = self.risk_report.get("performance_attribution", {})
        if performance_attribution:
            st.subheader("Performance Attribution")

            for metric, value in performance_attribution.items():
                if isinstance(value, dict):
                    st.write(f"**{metric.replace('_', ' ').title()}:**")
                    for sub_metric, sub_value in value.items():
                        st.write(f"- {sub_metric}: {sub_value}")
                else:
                    st.write(f"**{metric.replace('_', ' ').title()}:** {value}")


def main():
    """Run the risk dashboard."""
    dashboard = RiskDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
