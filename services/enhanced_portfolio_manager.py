#!/usr/bin/env python3
"""
Enhanced Portfolio Manager for Triton
Provides an enhanced simulation harness that integrates with the AdaptiveRiskEngine.
This is an overwrite-ready implementation that includes a defensive
_generate_risk_report method to avoid TypeErrors when position values are non-numeric.
"""

import os
import sys
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Allow imports from project root if executed as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EnhancedPortfolioManager:
    def __init__(self, universe_data: Dict[str, pd.DataFrame] = None, risk_allocator=None):
        """
        Minimal manager initialization. In your real code this may accept more arguments.
        """
        self.universe_data = universe_data or {}
        self.risk_allocator = risk_allocator
        self.current_positions = {}  # ticker -> position dict or numeric value
        self.portfolio_value = 0.0
        self.last_regime = "Unknown"
        self.last_allocation = {}

    def run_enhanced_simulation(self):
        """
        Entry point for the pipeline step. This stub demonstrates loading positions,
        running allocation logic if present, and generating a risk report snapshot.
        Replace or extend with your real simulation logic as needed.
        """
        print("🚀 Starting Enhanced Portfolio Simulation with Adaptive Risk Management...")
        # Load or compute universe_data / positions here if not already set

        # If no universe was provided, attempt to load from data/*.csv
        if not self.universe_data:
            data_files = sorted(Path("data").glob("*.csv"))
            for f in data_files:
                try:
                    df = pd.read_csv(f)
                    if "close" in df.columns:
                        ticker = f.stem.split("_")[0].upper()
                        self.universe_data[ticker] = df
                except Exception:
                    continue

        # If no positions exist, create a simple equal-weight sample from universe
        if not self.current_positions:
            tickers = list(self.universe_data.keys())[:5]
            if tickers:
                pv = 100000.0
                self.portfolio_value = pv
                weight = 1.0 / max(len(tickers), 1)
                for t in tickers:
                    self.current_positions[t] = {"market_value": pv * weight}
                print(f"✅ Loaded data for {len(tickers)} tickers")
            else:
                print("⚠️ No universe data available to build sample positions")

        # Generate a risk report snapshot
        current_date = pd.Timestamp.now().date()
        report = self._generate_risk_report(current_date, getattr(self, "portfolio_value", 0.0))

        # Persist the manager's risk report (pipeline compatibility)
        rp_path = RESULTS_DIR.joinpath("risk_report_from_manager.json")
        try:
            with open(rp_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✔ Saved manager risk report to: {rp_path}")
        except Exception as e:
            print(f"⚠️ Could not save manager risk report: {e}")

        # Update last_allocation for downstream usage
        try:
            self.last_allocation = report.get("current_weights", {})
        except Exception:
            pass

        return True

    def _generate_risk_report(self, current_date, portfolio_value):
        """
        Generate a risk report snapshot.

        Defensive behaviour:
        - Coerce portfolio_value and position values to numeric.
        - Skip positions with invalid numeric values and emit a warning for each.
        - Avoid divide-by-zero; normalize weights safely.
        - Returns a dict with portfolio metrics, current_weights, and basic meta.
        """
        # Ensure portfolio_value is numeric
        try:
            portfolio_value = float(portfolio_value)
        except Exception:
            try:
                portfolio_value = float(pd.to_numeric(portfolio_value, errors="coerce"))
            except Exception:
                portfolio_value = float("nan")

        if portfolio_value == 0 or math.isnan(portfolio_value):
            print(f"⚠️ Invalid portfolio_value for risk report: {portfolio_value}. Setting to 1.0 to avoid divide-by-zero.")
            portfolio_value = 1.0

        # Gather current weights from positions (defensive casting)
        current_weights = {}
        positions = getattr(self, "current_positions", {}) or {}

        for ticker, pos in positions.items():
            # pos might be a dict or a scalar number. Try common dict keys first.
            position_value = None
            if isinstance(pos, dict):
                # try multiple common keys
                for key in ("market_value", "value", "position_value", "size", "notional", "mv", "amount"):
                    if key in pos and pos[key] is not None:
                        position_value = pos[key]
                        break
                # fallback to an explicit 'quantity' * 'price' if present
                if position_value is None and ("quantity" in pos and "price" in pos):
                    try:
                        position_value = float(pos.get("quantity", 0)) * float(pos.get("price", 0))
                    except Exception:
                        position_value = None
            else:
                # pos may be already numeric or a string representing a number
                position_value = pos

            # Coerce to numeric (NaN on failure)
            pv_numeric = pd.to_numeric(position_value, errors="coerce")
            if pd.isna(pv_numeric):
                print(f"⚠️ Skipping position '{ticker}' — non-numeric position value: {repr(position_value)}")
                continue

            # Compute weight safely
            try:
                weight = float(pv_numeric) / float(portfolio_value)
            except Exception as e:
                print(f"⚠️ Could not compute weight for {ticker}: {e}")
                continue

            current_weights[ticker] = float(weight)

        # Normalize weights (preserve sign if present)
        total_w = sum([abs(w) for w in current_weights.values()]) or 1.0
        if total_w > 0:
            current_weights = {t: (w / total_w) for t, w in current_weights.items()}

        # Build the risk report skeleton
        report = {
            "date": str(current_date),
            "portfolio_value": float(portfolio_value),
            "current_weights": current_weights,
            "num_positions": len(current_weights),
        }

        # Add portfolio metrics if available via risk allocator
        try:
            metrics = {}
            if getattr(self, "risk_allocator", None) is not None:
                metrics = self.risk_allocator._calculate_portfolio_metrics(
                    current_weights,
                    getattr(self, "universe_data", {}),
                    getattr(self, "last_regime", "Unknown"),
                )
            report["portfolio_metrics"] = metrics
        except Exception as e:
            print(f"⚠️ Failed to calculate portfolio metrics in risk report: {e}")
            report["portfolio_metrics"] = {}

        # Attach regime/last allocation metadata if present
        report["regime"] = getattr(self, "last_regime", "Unknown")
        report["last_allocation"] = getattr(self, "last_allocation", {})

        return report


def main():
    manager = EnhancedPortfolioManager()
    ok = manager.run_enhanced_simulation()
    if ok:
        print("✅ Enhanced portfolio simulation completed (demo run).")
    else:
        print("❌ Enhanced portfolio simulation failed (demo run).")


if __name__ == "__main__":
    main()


