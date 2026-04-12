# services/risk_dashboard.py
"""
Triton Risk Dashboard (Phase 1.5)

- Reads saved artifacts (portfolio history + risk_report.json + adaptive_risk_state.json)
- Degrades gracefully if AdaptiveRiskEngine isn't available
- Defensive schema handling (contracts stay green)
- Live vs Backtest selectable (prevents trusting the wrong equity curve)
- Integrity Panel flags outlier moves / broken data BEFORE real-money use
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# AdaptiveRiskEngine import (robust for Streamlit script execution)
# ─────────────────────────────────────────────────────────────
_adaptive_import_error = ""
AdaptiveRiskEngine = None

try:
    # Preferred when streamlit runs from repo root: streamlit run services/risk_dashboard.py
    from services.adaptive_risk_engine import AdaptiveRiskEngine as _ARE  # type: ignore

    AdaptiveRiskEngine = _ARE
except Exception as e1:
    try:
        # Fallback for package execution contexts
        from .adaptive_risk_engine import AdaptiveRiskEngine as _ARE  # type: ignore

        AdaptiveRiskEngine = _ARE
    except Exception as e2:
        AdaptiveRiskEngine = None
        _adaptive_import_error = f"{e1} | {e2}"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON defensively. Returns None if missing/empty/unreadable."""
    if not (path.exists() and path.stat().st_size > 0):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to read JSON: {path} ({e})")
            return None


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _candidate_portfolio_files() -> List[Tuple[str, Path]]:
    """
    Present portfolio sources clearly.
    Priority order in UI is deliberate: Live first, then Backtest.
    """
    return [
        (
            "LIVE (preferred) — enhanced_portfolio_history.csv",
            Path("data/results/enhanced_portfolio_history.csv"),
        ),
        ("LIVE (fallback) — portfolio_history.csv", Path("data/results/portfolio_history.csv")),
        (
            "BACKTEST — backtest_portfolio_history.csv",
            Path("data/results/backtest_portfolio_history.csv"),
        ),
    ]


def _pick_existing(files: List[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for label, p in files:
        if p.exists() and p.stat().st_size > 0:
            out.append((label, p))
    return out


def _is_truthy_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _compute_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns integrity diagnostics to prevent trusting inflated/broken curves.
    """
    diag: Dict[str, Any] = {}

    if df is None or df.empty or "date" not in df.columns:
        diag["ok"] = False
        diag["reason"] = "Missing or empty dataframe / no date column."
        return diag

    x = df.copy()

    for c in ("total_value", "cash", "market_value", "num_positions"):
        if c in x.columns:
            x[c] = _safe_to_numeric(x[c])

    # Daily returns on total_value
    if "total_value" in x.columns and x["total_value"].notna().any():
        x = x.sort_values("date").reset_index(drop=True)
        x["tv_ret"] = x["total_value"].pct_change()

        out = x.loc[x["tv_ret"].abs() > 0.20, ["date", "total_value", "tv_ret"]].copy()
        diag["max_abs_daily_move"] = (
            float(x["tv_ret"].abs().max()) if x["tv_ret"].notna().any() else None
        )
        diag["outliers_20pct"] = out.tail(25)

        top = x.loc[x["tv_ret"].notna(), ["date", "total_value", "tv_ret"]].copy()
        top = top.reindex(top["tv_ret"].abs().sort_values(ascending=False).index).head(10)
        diag["top_moves"] = top
    else:
        diag["max_abs_daily_move"] = None
        diag["outliers_20pct"] = pd.DataFrame()
        diag["top_moves"] = pd.DataFrame()

    # Negatives
    negs: List[str] = []
    for c in ("cash", "market_value", "total_value"):
        if c in x.columns and x[c].notna().any():
            if (x[c] < 0).any():
                negs.append(c)
    diag["negative_columns"] = negs

    # NaN summary
    nan_summary: Dict[str, float] = {}
    for c in ("cash", "market_value", "total_value", "num_positions", "regime"):
        if c in x.columns:
            nan_summary[c] = float(x[c].isna().mean())
    diag["nan_fraction"] = nan_summary

    diag["ok"] = True
    return diag


# ─────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────


class RiskDashboard:
    """
    Interactive Risk Dashboard for Triton's Adaptive Risk Engine.

    Provides visualization of:
    - Portfolio overview
    - Integrity panel
    - Risk metrics
    - Factor exposures
    - Risk controls (prefers adaptive_risk_state.json)
    """

    def __init__(self) -> None:
        self.adaptive_risk_engine = None
        if AdaptiveRiskEngine is not None:
            try:
                self.adaptive_risk_engine = AdaptiveRiskEngine()
            except Exception:
                self.adaptive_risk_engine = None

        self.portfolio_data: Optional[pd.DataFrame] = None
        self.portfolio_source_label: str = ""
        self.portfolio_source_path: Optional[Path] = None

        self.risk_report: Optional[Dict[str, Any]] = None
        self.adaptive_state: Optional[Dict[str, Any]] = None

    # ─────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────

    def load_data(self, portfolio_path: Optional[Path]) -> None:
        """Load portfolio and risk data (saved artifacts)."""
        self.portfolio_data = self._load_portfolio_history(portfolio_path)
        self.risk_report = self._load_risk_report()
        self.adaptive_state = self._load_adaptive_state()

    def _load_portfolio_history(self, portfolio_path: Optional[Path]) -> Optional[pd.DataFrame]:
        """
        Load portfolio history from chosen path.
        Defensive contract handling:
          required: date
          core: cash/market_value/total_value (filled with NaN if missing)
          optional: num_positions, regime
        """
        if portfolio_path is None:
            return None

        try:
            df = pd.read_csv(portfolio_path)

            if "date" not in df.columns:
                st.error(f"Portfolio file missing required column 'date': {portfolio_path}")
                return None

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).copy()

            # Core columns (ensure present)
            for c in ("cash", "market_value", "total_value"):
                if c not in df.columns:
                    df[c] = np.nan

            # Optional columns
            if "num_positions" not in df.columns:
                df["num_positions"] = np.nan
            if "regime" not in df.columns:
                df["regime"] = np.nan

            # Coerce numerics
            for c in ("cash", "market_value", "total_value", "num_positions"):
                if c in df.columns:
                    df[c] = _safe_to_numeric(df[c])

            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            st.error(f"Failed to read portfolio file: {portfolio_path} ({e})")
            return None

    def _load_risk_report(self) -> Optional[Dict[str, Any]]:
        return _read_json(Path("data/results/risk_report.json"))

    def _load_adaptive_state(self) -> Optional[Dict[str, Any]]:
        # Preferred state file per your UI
        return _read_json(Path("data/results/adaptive_risk_state.json"))

    # ─────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────

    def render_dashboard(self) -> None:
        st.set_page_config(page_title="Triton Risk Dashboard", page_icon="📊", layout="wide")

        st.title("🎯 Triton Adaptive Risk Dashboard")
        st.markdown("---")

        # Non-blocking engine note
        if AdaptiveRiskEngine is None:
            st.info(
                "AdaptiveRiskEngine not available. Dashboard will render using saved artifacts."
            )
            if _adaptive_import_error:
                st.caption(f"Adaptive engine import error: {_adaptive_import_error}")

        # Sidebar: portfolio source selection
        existing = _pick_existing(_candidate_portfolio_files())
        if not existing:
            st.error(
                "❌ No portfolio data found.\n\nExpected one of:\n"
                "- data/results/enhanced_portfolio_history.csv\n"
                "- data/results/portfolio_history.csv\n"
                "- data/results/backtest_portfolio_history.csv"
            )
            return

        with st.sidebar:
            st.subheader("Data Source")
            labels = [lbl for lbl, _ in existing]
            default_idx = 0  # LIVE preferred if present
            choice = st.radio("Portfolio history", labels, index=default_idx)
            chosen_path = dict(existing)[choice]
            st.caption(f"Using: {chosen_path.as_posix()}")

        self.portfolio_source_label = choice
        self.portfolio_source_path = chosen_path

        self.load_data(chosen_path)

        if self.portfolio_data is None or self.portfolio_data.empty:
            st.error("❌ Portfolio data loaded but empty.")
            return

        # Quick header metrics
        df = self.portfolio_data
        last_tv = (
            float(df["total_value"].dropna().iloc[-1]) if df["total_value"].notna().any() else 0.0
        )
        first_tv = (
            float(df["total_value"].dropna().iloc[0]) if df["total_value"].notna().any() else 0.0
        )
        total_return = ((last_tv / first_tv) - 1.0) * 100.0 if first_tv else 0.0
        peak_tv = (
            float(df["total_value"].dropna().max()) if df["total_value"].notna().any() else 0.0
        )
        drawdown = ((last_tv / peak_tv) - 1.0) * 100.0 if peak_tv else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Total Value", f"${last_tv:,.2f}")
        c2.metric("Total Return", f"{total_return:.2f}%")
        c3.metric("Peak Total Value", f"${peak_tv:,.2f}")
        c4.metric("Drawdown vs Peak", f"{drawdown:.2f}%")

        st.caption(f"Source: {self.portfolio_source_label}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📈 Portfolio Overview",
                "🧪 Integrity Panel",
                "🎯 Risk Metrics",
                "📊 Factor Exposure",
                "⚙️ Risk Controls",
            ]
        )

        with tab1:
            self._render_portfolio_overview()

        with tab2:
            self._render_integrity_panel()

        with tab3:
            self._render_risk_metrics()

        with tab4:
            self._render_factor_exposure()

        with tab5:
            self._render_risk_controls()

    # ─────────────────────────────────────────────────────────────
    # Tab 1: Portfolio Overview
    # ─────────────────────────────────────────────────────────────

    def _render_portfolio_overview(self) -> None:
        st.header("📈 Portfolio Overview")
        df = self.portfolio_data.copy()

        fig = go.Figure()

        if df["total_value"].notna().any():
            fig.add_trace(
                go.Scatter(x=df["date"], y=df["total_value"], mode="lines", name="Total Value")
            )

        if "market_value" in df.columns and df["market_value"].notna().any():
            fig.add_trace(
                go.Scatter(x=df["date"], y=df["market_value"], mode="lines", name="Market Value")
            )

        if "cash" in df.columns and df["cash"].notna().any():
            fig.add_trace(go.Scatter(x=df["date"], y=df["cash"], mode="lines", name="Cash"))

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Regime overlay if available
        if (
            "regime" in df.columns
            and df["regime"].notna().any()
            and df["total_value"].notna().any()
        ):
            fig_regime = px.scatter(
                df,
                x="date",
                y="total_value",
                color="regime",
                title="Portfolio Value by Market Regime",
                labels={"total_value": "Portfolio Value ($)", "date": "Date"},
            )
            st.plotly_chart(fig_regime, use_container_width=True)
        else:
            st.info("No regime labels present — portfolio value shown without regime overlay.")

        st.subheader("Raw portfolio file (tail)")
        cols = [
            c
            for c in ["date", "cash", "market_value", "total_value", "regime", "num_positions"]
            if c in df.columns
        ]
        st.dataframe(df[cols].tail(25), use_container_width=True)

    # ─────────────────────────────────────────────────────────────
    # Tab 2: Integrity Panel
    # ─────────────────────────────────────────────────────────────

    def _render_integrity_panel(self) -> None:
        st.header("🧪 Integrity Panel")

        df = self.portfolio_data.copy()
        diag = _compute_integrity(df)

        if not diag.get("ok", False):
            st.error(f"Integrity diagnostics failed: {diag.get('reason','unknown')}")
            return

        max_move = diag.get("max_abs_daily_move")
        neg_cols: List[str] = diag.get("negative_columns", [])
        nan_frac: Dict[str, float] = diag.get("nan_fraction", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Max |Daily Move|", f"{(max_move or 0.0) * 100:.2f}%")
        c2.metric("Negative Columns", ", ".join(neg_cols) if neg_cols else "None")
        c3.metric("NaN (total_value)", f"{nan_frac.get('total_value', 0.0) * 100:.1f}%")

        # Warn if the curve looks “too good to be live”
        if max_move is not None and max_move > 0.20:
            st.warning(
                "Large daily moves detected (>20%). "
                "This is common in BACKTEST curves (compounding / sizing artifacts). "
                "Before real-money, confirm you are viewing LIVE portfolio history."
            )

        st.subheader("Top 10 Daily Moves (by absolute %)")
        top = diag.get("top_moves", pd.DataFrame())
        if isinstance(top, pd.DataFrame) and not top.empty:
            show = top.copy()
            show["tv_ret"] = show["tv_ret"].apply(lambda x: f"{x * 100:.2f}%")
            st.dataframe(show, use_container_width=True)
        else:
            st.info("Not enough data to compute daily moves.")

        st.subheader("Outliers > 20% Daily Move (tail)")
        out = diag.get("outliers_20pct", pd.DataFrame())
        if isinstance(out, pd.DataFrame) and not out.empty:
            show = out.copy()
            show["tv_ret"] = show["tv_ret"].apply(lambda x: f"{x * 100:.2f}%")
            st.dataframe(show, use_container_width=True)
        else:
            st.info("No >20% daily outliers detected.")

        st.subheader("NaN Fraction by Column")
        st.dataframe(pd.DataFrame([nan_frac]), use_container_width=True)

    # ─────────────────────────────────────────────────────────────
    # Tab 3: Risk Metrics
    # ─────────────────────────────────────────────────────────────

    def _render_risk_metrics(self) -> None:
        st.header("🎯 Risk Metrics")

        if self.risk_report is None:
            st.info("No risk report data available (data/results/risk_report.json).")
            return

        pm = (
            self.risk_report.get("portfolio_metrics", {})
            if isinstance(self.risk_report, dict)
            else {}
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            v = pm.get("expected_volatility")
            if v is None:
                st.metric("Expected Volatility", "N/A")
            else:
                try:
                    st.metric("Expected Volatility", f"{float(v):.2%}")
                except Exception:
                    st.metric("Expected Volatility", str(v))

        with col2:
            v = pm.get("diversification_ratio")
            if v is None:
                st.metric("Diversification Ratio", "N/A")
            else:
                try:
                    st.metric("Diversification Ratio", f"{float(v):.2f}")
                except Exception:
                    st.metric("Diversification Ratio", str(v))

        with col3:
            v = pm.get("risk_adjusted_return")
            if v is None:
                st.metric("Risk-Adjusted Return", "N/A")
            else:
                try:
                    st.metric("Risk-Adjusted Return", f"{float(v):.2%}")
                except Exception:
                    st.metric("Risk-Adjusted Return", str(v))

        risk_decomp = self.risk_report.get("risk_decomposition", {})
        if isinstance(risk_decomp, dict) and risk_decomp:
            st.subheader("📊 Risk Decomposition")

            labels = list(risk_decomp.keys())
            values: List[float] = []
            for v in risk_decomp.values():
                try:
                    values.append(float(v))
                except Exception:
                    values.append(0.0)

            if sum(values) <= 0:
                st.info("Risk decomposition contains no positive contributions.")
            else:
                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                fig.update_layout(title="Portfolio Risk Decomposition")
                st.plotly_chart(fig, use_container_width=True)

            risk_df = pd.DataFrame(
                [{"Factor": k, "Risk Contribution": v} for k, v in risk_decomp.items()]
            )
            st.dataframe(risk_df, use_container_width=True)
        else:
            st.info("No risk decomposition data present.")

    # ─────────────────────────────────────────────────────────────
    # Tab 4: Factor Exposure
    # ─────────────────────────────────────────────────────────────

    def _render_factor_exposure(self) -> None:
        st.header("📊 Factor Exposure Analysis")

        if self.risk_report is None:
            st.info("No factor exposure data available (risk_report.json missing).")
            return

        position_analysis = self.risk_report.get("position_analysis", {})
        if isinstance(position_analysis, dict) and position_analysis:
            st.subheader("Position Analysis")

            rows: List[Dict[str, Any]] = []
            for ticker, data in position_analysis.items():
                if not isinstance(data, dict):
                    continue

                def _f(x: Any) -> float:
                    try:
                        return float(x)
                    except Exception:
                        return float("nan")

                rows.append(
                    {
                        "Ticker": str(ticker),
                        "Weight": _f(data.get("weight")),
                        "Volatility": _f(data.get("volatility")),
                        "Risk Contribution": _f(data.get("risk_contribution")),
                    }
                )

            position_df = pd.DataFrame(rows)

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
            st.dataframe(display_df, use_container_width=True)

            if position_df["Risk Contribution"].notna().any():
                fig = px.bar(
                    position_df.sort_values("Risk Contribution", ascending=False),
                    x="Ticker",
                    y="Risk Contribution",
                    title="Risk Contribution by Position",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No position_analysis found in risk_report.json.")

        factor_weights = self.risk_report.get("factor_weights", {})
        if isinstance(factor_weights, dict) and factor_weights:
            st.subheader("Factor Weights")

            factor_df = pd.DataFrame(
                {
                    "Factor": list(factor_weights.keys()),
                    "Weight": [pd.to_numeric(v, errors="coerce") for v in factor_weights.values()],
                }
            )

            st.dataframe(factor_df, use_container_width=True)

            if factor_df["Weight"].notna().any():
                fig = px.bar(
                    factor_df.sort_values("Weight", ascending=False),
                    x="Factor",
                    y="Weight",
                    title="Factor Weights",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No factor_weights found in risk_report.json.")

    # ─────────────────────────────────────────────────────────────
    # Tab 5: Risk Controls (prefers adaptive_state)
    # ─────────────────────────────────────────────────────────────

    def _render_risk_controls(self) -> None:
        st.header("⚙️ Risk Controls")

        # Preferred state: adaptive_risk_state.json
        if self.adaptive_state is not None and isinstance(self.adaptive_state, dict):
            st.subheader("Adaptive Risk State (preferred)")
            st.json(self.adaptive_state)

            # Quick “headline” fields if present
            headline_keys = [
                "mode",
                "regime",
                "risk_level",
                "exposure_multiplier",
                "halt_trading",
                "reason",
            ]
            headline = {
                k: self.adaptive_state.get(k) for k in headline_keys if k in self.adaptive_state
            }

            if headline:
                st.subheader("State Summary")
                st.dataframe(pd.DataFrame([headline]), use_container_width=True)
        else:
            st.info("adaptive_risk_state.json not available.")

        # Secondary: risk_report.json
        if self.risk_report is None:
            st.info("No risk_report.json available.")
            return

        risk_limits = self.risk_report.get("risk_limits", {})
        if isinstance(risk_limits, dict) and risk_limits:
            st.subheader("Risk Limits (from risk_report.json)")

            limit_rows: List[Dict[str, str]] = []
            for k, v in risk_limits.items():
                if isinstance(v, (float, int)):
                    display_value = (
                        f"{float(v):.2%}" if abs(float(v)) <= 1.0 else f"{float(v):,.4f}"
                    )
                else:
                    display_value = str(v)
                limit_rows.append(
                    {"Limit": str(k).replace("_", " ").title(), "Value": display_value}
                )

            st.dataframe(pd.DataFrame(limit_rows), use_container_width=True)

        regime_adjustments = self.risk_report.get("regime_adjustments", {})
        if isinstance(regime_adjustments, dict) and regime_adjustments:
            st.subheader("Regime Adjustments (from risk_report.json)")

            rows: List[Dict[str, str]] = []
            for regime, adj in regime_adjustments.items():
                if not isinstance(adj, dict):
                    continue
                for k, v in adj.items():
                    rows.append(
                        {
                            "Regime": str(regime),
                            "Adjustment": str(k).replace("_", " ").title(),
                            "Value": f"{v:.4f}" if isinstance(v, (float, int)) else str(v),
                        }
                    )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        perf_attr = self.risk_report.get("performance_attribution", {})
        if isinstance(perf_attr, dict) and perf_attr:
            st.subheader("Performance Attribution (from risk_report.json)")
            st.json(perf_attr)


def main() -> None:
    dashboard = RiskDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
