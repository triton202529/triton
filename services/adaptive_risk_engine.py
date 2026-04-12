# services/adaptive_risk_engine.py
"""
TRITON — Adaptive Risk Engine (Phase 1.5)

Goals:
- Streamlit-safe imports (no "attempted relative import with no known parent package")
- Works when executed as a module:  python -m services.adaptive_risk_engine
- Degrades gracefully if subcomponents fail (keeps pipeline alive)
- Defensive schema/typing + clean logging

This engine integrates:
- Market regime detection
- Multi-factor risk modeling
- Dynamic risk allocation
- Optional tail-risk hedging
- Lightweight performance attribution
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Imports (Streamlit / module-safe)
#   Prefer absolute imports (services.*). Fallback to relative.
# ─────────────────────────────────────────────────────────────
try:
    from services.regime_detector import RegimeDetector
    from services.multi_factor_risk_model import MultiFactorRiskModel
    from services.dynamic_risk_allocator import DynamicRiskAllocator
except Exception:
    from .regime_detector import RegimeDetector
    from .multi_factor_risk_model import MultiFactorRiskModel
    from .dynamic_risk_allocator import DynamicRiskAllocator


class AdaptiveRiskEngine:
    """
    Master Adaptive Risk Budgeting Engine for Triton.

    Integrates:
    - Market regime detection
    - Multi-factor risk modeling
    - Dynamic risk allocation
    - Tail risk hedging (optional)
    - Performance attribution (lightweight)
    """

    def __init__(self, config_path: str = "config/adaptive_risk.json", verbose: bool = False):
        self.config_path = config_path
        self.verbose = bool(verbose)
        self.config: Dict[str, Any] = self._load_config()

        # Components
        self.regime_detector = RegimeDetector(
            lookback_days=int(self.config.get("lookback_days", 252)),
            min_samples=int(self.config.get("min_samples", 50)),
        )

        self.risk_model = MultiFactorRiskModel()

        self.risk_allocator = DynamicRiskAllocator(
            target_volatility=float(self.config.get("target_volatility", 0.15)),
            max_position_size=float(self.config.get("max_position_size", 0.10)),
            verbose=self.verbose,
        )

        # Connect components
        try:
            self.risk_allocator.set_regime_detector(self.regime_detector)
        except Exception:
            pass
        try:
            self.risk_allocator.set_risk_model(self.risk_model)
        except Exception:
            pass

        # State
        self.is_initialized: bool = False
        self.last_regime: str = "Unknown"
        self.last_allocation: Dict[str, float] = {}
        self.performance_history: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────

    def _log(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with sensible defaults."""
        default_config: Dict[str, Any] = {
            "target_volatility": 0.15,
            "max_position_size": 0.10,
            "lookback_days": 252,
            "min_samples": 50,
            "regime_threshold": 0.7,
            "correlation_threshold": 0.8,
            "tail_risk_threshold": 0.05,
            "rebalance_frequency": "daily",
            "enable_hedging": True,
            "enable_factor_timing": True,
            "enable_volatility_targeting": True,
        }

        try:
            cfg_path = Path(self.config_path)
            if cfg_path.exists() and cfg_path.stat().st_size > 0:
                user_config = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(user_config, dict):
                    default_config.update(user_config)
        except Exception as e:
            # Keep going with defaults (never block pipeline)
            print(f"⚠️ Error loading config ({self.config_path}): {e}. Using defaults.")

        return default_config

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    def initialize(self, universe_data: Dict[str, pd.DataFrame]) -> bool:
        """Initialize the engine (fit underlying models best-effort)."""
        self._log("🚀 Initializing Adaptive Risk Engine...")

        if not isinstance(universe_data, dict) or not universe_data:
            print("❌ initialize() called with empty or invalid universe_data")
            return False

        # Determine market proxy for regime detector
        market_df = universe_data.get("SPY")
        if market_df is None:
            try:
                market_df = next(iter(universe_data.values()))
            except StopIteration:
                print("❌ No valid market data found to initialize regime detector")
                return False

        # Fit components (best-effort)
        try:
            self.regime_detector.fit(market_df)
        except Exception as e:
            self._log(f"Regime detector training failed: {e}")

        try:
            self.risk_model.fit(universe_data)
        except Exception as e:
            self._log(f"Risk model training failed: {e}")

        self.is_initialized = True
        self._log("✅ Adaptive Risk Engine initialized successfully")
        return True

    # ─────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────

    def process_signals(
        self,
        signals: Dict[str, float],
        universe_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Process incoming signals and return final normalized allocations."""
        if not isinstance(signals, dict) or not signals:
            return {}

        if not self.is_initialized:
            self._log("⚠️ Engine not initialized — using simple allocation fallback")
            return self._simple_allocation(signals)

        self._log("🎯 Processing signals through Adaptive Risk Engine...")

        # Detect regime (safe)
        regime, confidence = self._detect_regime_safe(universe_data)

        # Regime transition log
        if regime != self.last_regime:
            self._log(f"🔄 Regime change detected: {self.last_regime} → {regime}")
            self.last_regime = regime

        # Allocate risk budget (fallback if allocator fails)
        try:
            adaptive_weights = self.risk_allocator.allocate_risk_budget(signals, universe_data)
            if not isinstance(adaptive_weights, dict):
                raise ValueError("risk_allocator returned non-dict weights")
        except Exception as e:
            self._log(f"risk_allocator failed: {e}")
            adaptive_weights = self._simple_allocation(signals)

        # Regime adjustments
        try:
            regime_adjusted = self._apply_regime_adjustments(
                adaptive_weights, regime, universe_data
            )
        except Exception as e:
            self._log(f"_apply_regime_adjustments failed: {e}")
            regime_adjusted = adaptive_weights

        # Tail-risk hedging (optional)
        hedged = regime_adjusted
        if bool(self.config.get("enable_hedging", True)):
            try:
                hedged = self._apply_tail_risk_hedging(regime_adjusted, regime, universe_data)
            except Exception as e:
                self._log(f"_apply_tail_risk_hedging failed: {e}")
                hedged = regime_adjusted

        # Risk limits
        try:
            final_weights = self._apply_risk_limits(hedged, universe_data)
        except Exception as e:
            self._log(f"_apply_risk_limits failed: {e}")
            final_weights = hedged

        # Final cleanup
        final_weights = self._clip_and_normalize(final_weights)

        # Risk report (best-effort)
        try:
            risk_report = self.risk_allocator.get_risk_report(final_weights, universe_data)
            if not isinstance(risk_report, dict):
                risk_report = {"portfolio_metrics": {}}
        except Exception as e:
            self._log(f"get_risk_report failed: {e}")
            risk_report = {"portfolio_metrics": {}}

        # Update state / history
        self.last_allocation = final_weights
        try:
            self._update_performance_history(risk_report)
        except Exception as e:
            self._log(f"_update_performance_history failed: {e}")

        expected_vol = risk_report.get("portfolio_metrics", {}).get("expected_volatility", 0.0)

        self._log("✅ Adaptive allocation complete")
        try:
            self._log(f"📊 Regime: {regime} (confidence: {float(confidence):.2%})")
        except Exception:
            self._log(f"📊 Regime: {regime} (confidence: {confidence})")
        try:
            self._log(f"🎯 Portfolio volatility: {float(expected_vol):.2%}")
        except Exception:
            self._log(f"🎯 Portfolio volatility: {expected_vol}")

        return final_weights

    # ─────────────────────────────────────────────────────────────
    # Regime / adjustments
    # ─────────────────────────────────────────────────────────────

    def _detect_regime_safe(self, universe_data: Dict[str, pd.DataFrame]) -> Tuple[str, float]:
        """Safe wrapper around regime detection; returns (regime, confidence)."""
        try:
            if not isinstance(universe_data, dict) or not universe_data:
                return "Unknown", 0.0

            market_df = universe_data.get("SPY")
            if market_df is None:
                market_df = next(iter(universe_data.values()), pd.DataFrame())

            out = self.regime_detector.predict_regime(market_df)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return str(out[0]), float(out[1])
            return str(out), 0.0
        except Exception as e:
            self._log(f"_detect_regime_safe failed: {e}")
            return "Unknown", 0.0

    def _apply_regime_adjustments(
        self,
        weights: Dict[str, float],
        regime: str,
        universe_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Apply regime-specific multipliers to weights, then renormalize."""
        if not weights:
            return {}

        try:
            adj = self.regime_detector.get_regime_risk_adjustments(regime)
            regime_adjustments = adj if isinstance(adj, dict) else {}
        except Exception:
            regime_adjustments = {}

        position_multiplier = float(regime_adjustments.get("position_size_multiplier", 1.0))
        volatility_multiplier = float(regime_adjustments.get("volatility_multiplier", 1.0))

        adjusted: Dict[str, float] = {}
        for t, w in weights.items():
            try:
                wv = float(w)
            except Exception:
                wv = 0.0

            new_w = wv * position_multiplier
            # Defensive: if vol multiplier > 1, shrink weights further
            if volatility_multiplier > 1.0:
                new_w = new_w / volatility_multiplier

            adjusted[str(t)] = max(new_w, 0.0)

        return self._clip_and_normalize(adjusted)

    def _apply_tail_risk_hedging(
        self,
        weights: Dict[str, float],
        regime: str,
        universe_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Tail risk hedging:
        - In high tail-risk regimes: shrink weights
        - Optionally add small hedge placeholder (kept negative)
        NOTE: _clip_and_normalize() will drop negatives, so we keep hedge logic minimal
              unless you later support long/short normalization explicitly.
        """
        if not weights:
            return {}

        try:
            adj = self.regime_detector.get_regime_risk_adjustments(regime)
            regime_adjustments = adj if isinstance(adj, dict) else {}
        except Exception:
            regime_adjustments = {}

        tail_risk_multiplier = float(regime_adjustments.get("tail_risk_multiplier", 1.0))

        if tail_risk_multiplier <= 1.5:
            return self._clip_and_normalize(weights)

        # shrink positions
        hedged: Dict[str, float] = {}
        for t, w in weights.items():
            try:
                hedged[str(t)] = max(float(w) / max(tail_risk_multiplier, 1e-9), 0.0)
            except Exception:
                hedged[str(t)] = 0.0

        # IMPORTANT:
        # Your current system normalizes long-only weights; negative hedges would be clipped away.
        # We keep this stub for future long/short support without breaking today.
        # Example placeholder (not applied to final):
        # if "SPY" in universe_data:
        #     hedged["SPY_HEDGE"] = -abs(min(0.05, tail_risk_multiplier * 0.02))

        return self._clip_and_normalize(hedged)

    # ─────────────────────────────────────────────────────────────
    # Risk limits
    # ─────────────────────────────────────────────────────────────

    def _apply_risk_limits(
        self,
        weights: Dict[str, float],
        universe_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Apply max position caps and correlation shrinkage; then renormalize."""
        if not weights:
            return {}

        max_pos = float(self.config.get("max_position_size", 0.10))

        limited: Dict[str, float] = {}
        for t, w in weights.items():
            try:
                limited[str(t)] = min(float(w), max_pos)
            except Exception:
                limited[str(t)] = 0.0

        # Correlation shrinkage (best-effort)
        try:
            corr_threshold = float(self.config.get("correlation_threshold", 0.8))
            corr = self.risk_allocator._calculate_correlation_matrix(
                universe_data
            )  # existing internal method
            if corr is not None and hasattr(corr, "empty") and not corr.empty:
                for t in list(limited.keys()):
                    if t in corr.columns:
                        # If this asset is highly correlated to others, shrink a bit
                        high_corr = corr[t] > corr_threshold
                        if bool(high_corr.any()):
                            limited[t] *= 0.8
        except Exception as e:
            self._log(f"_apply_risk_limits correlation adjustment failed: {e}")

        return self._clip_and_normalize(limited)

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────

    def _clip_and_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Long-only normalize: clip negatives to 0 and normalize to sum=1."""
        if not weights:
            return {}

        clipped: Dict[str, float] = {}
        for k, v in weights.items():
            try:
                clipped[str(k)] = max(0.0, float(v))
            except Exception:
                clipped[str(k)] = 0.0

        s = float(sum(clipped.values()))
        if s <= 0.0:
            return {}

        return {k: v / s for k, v in clipped.items()}

    def _simple_allocation(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Fallback: allocate proportional to positive signals."""
        if not signals:
            return {}

        positive: Dict[str, float] = {}
        for k, v in signals.items():
            try:
                positive[str(k)] = max(0.0, float(v))
            except Exception:
                positive[str(k)] = 0.0

        s = float(sum(positive.values()))
        if s <= 0.0:
            return {}

        return {k: v / s for k, v in positive.items()}

    def _update_performance_history(self, risk_report: Dict[str, Any]) -> None:
        """Append summary metrics to performance history (bounded list)."""
        timestamp = pd.Timestamp.now()

        regime = "Unknown"
        try:
            regime = str(risk_report.get("regime", {}).get("current_regime", "Unknown"))
        except Exception:
            pass

        pm = risk_report.get("portfolio_metrics", {}) if isinstance(risk_report, dict) else {}
        volatility = pm.get("expected_volatility", 0.0)
        diversification = pm.get("diversification_ratio", 1.0)

        def _f(x: Any, default: float = 0.0) -> float:
            try:
                if x is None:
                    return default
                return float(x)
            except Exception:
                return default

        self.performance_history.append(
            {
                "timestamp": str(timestamp),
                "regime": regime,
                "volatility": _f(volatility, 0.0),
                "diversification": _f(diversification, 1.0),
            }
        )

        # keep bounded
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    # ─────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────

    def get_performance_attribution(self) -> Dict[str, Any]:
        """Return lightweight performance attribution from stored history."""
        if not self.performance_history:
            return {}

        df = pd.DataFrame(self.performance_history)
        try:
            return {
                "regime_performance": df.groupby("regime")["volatility"].mean().to_dict(),
                "volatility_trend": (
                    float(df["volatility"].tail(20).mean()) if len(df) >= 1 else 0.0
                ),
                "diversification_trend": (
                    float(df["diversification"].tail(20).mean()) if len(df) >= 1 else 0.0
                ),
                "regime_stability": (
                    float(len(df["regime"].unique()) / len(df)) if len(df) > 0 else 0.0
                ),
            }
        except Exception as e:
            self._log(f"get_performance_attribution failed: {e}")
            return {}

    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Return payload suitable for the Risk Dashboard."""
        return {
            "current_regime": self.last_regime,
            "last_allocation": self.last_allocation,
            "performance_attribution": self.get_performance_attribution(),
            "config": self.config,
            "is_initialized": self.is_initialized,
        }

    def save_state(self, filepath: str = "data/results/adaptive_risk_state.json") -> None:
        """Persist engine state to disk (JSON)."""
        try:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "last_regime": self.last_regime,
                "last_allocation": self.last_allocation,
                "performance_history": self.performance_history,
                "config": self.config,
                "is_initialized": self.is_initialized,
            }
            p.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
            self._log(f"Saved engine state to {filepath}")
        except Exception as e:
            print(f"⚠️ save_state failed: {e}")

    def load_state(self, filepath: str = "data/results/adaptive_risk_state.json") -> None:
        """Load engine state from disk if present."""
        try:
            p = Path(filepath)
            if not p.exists() or p.stat().st_size == 0:
                self._log(f"No state file at {filepath}")
                return

            state = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return

            self.last_regime = str(state.get("last_regime", "Unknown"))
            self.last_allocation = state.get("last_allocation", {}) or {}
            self.performance_history = state.get("performance_history", []) or []
            self.is_initialized = bool(state.get("is_initialized", False))
            self._log("✅ Adaptive Risk Engine state loaded")
        except Exception as e:
            print(f"⚠️ load_state failed: {e}")


# ─────────────────────────────────────────────────────────────
# CLI / local test harness
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """Quick local test harness to exercise initialization and allocation."""
    import glob

    data_files = glob.glob(str(Path("data").joinpath("*.csv")))
    if not data_files:
        print("❌ No data files found in data/ (expect CSVs with 'date' and 'close' columns)")
        return

    universe_data: Dict[str, pd.DataFrame] = {}
    for file in data_files[:10]:
        ticker = Path(file).stem.split("_")[0]
        try:
            df = pd.read_csv(file)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
            universe_data[ticker] = df
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    engine = AdaptiveRiskEngine(verbose=True)
    if not engine.initialize(universe_data):
        print("❌ Engine initialization failed — exiting")
        return

    sample_signals = {t: float(np.random.random()) for t in list(universe_data.keys())[:5]}
    alloc = engine.process_signals(sample_signals, universe_data)

    print(f"\n📊 Sample Signals: {sample_signals}")
    print(f"🎯 Allocation: {alloc}")

    dashboard = engine.get_risk_dashboard_data()
    print(f"\n📈 Dashboard data summary: {dashboard}")

    engine.save_state()


if __name__ == "__main__":
    main()
