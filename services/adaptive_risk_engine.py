import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
    - Tail risk hedging
    - Performance attribution
    """

    def __init__(self, config_path: str = "config/adaptive_risk.json", verbose: bool = False):
        self.config_path = config_path
        self.config = self._load_config()
        self.verbose = verbose

        # Initialize components with config values where applicable
        self.regime_detector = RegimeDetector(
            lookback_days=int(self.config.get('lookback_days', 252)),
            min_samples=int(self.config.get('min_samples', 50))
        )

        self.risk_model = MultiFactorRiskModel()

        self.risk_allocator = DynamicRiskAllocator(
            target_volatility=float(self.config.get('target_volatility', 0.15)),
            max_position_size=float(self.config.get('max_position_size', 0.10)),
            verbose=self.verbose
        )

        # Connect components
        self.risk_allocator.set_regime_detector(self.regime_detector)
        self.risk_allocator.set_risk_model(self.risk_model)

        # State tracking
        self.is_initialized = False
        self.last_regime = 'Unknown'
        self.last_allocation: Dict[str, float] = {}
        self.performance_history: List[Dict[str, Any]] = []

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _load_config(self) -> Dict:
        """Load configuration from file with defaults."""
        default_config = {
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
            "enable_volatility_targeting": True
        }

        try:
            cfg_path = Path(self.config_path)
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            print(f"⚠️ Error loading config ({self.config_path}): {e}. Using defaults.")

        return default_config

    def initialize(self, universe_data: Dict[str, pd.DataFrame]) -> bool:
        """Initialize the adaptive risk engine (train models)."""
        self._log("🚀 Initializing Adaptive Risk Engine...")

        try:
            if not universe_data:
                print("❌ initialize() called with empty universe_data")
                return False

            # Determine market proxy for regime detector: prefer SPY, else first ticker
            market_df = universe_data.get('SPY')
            if market_df is None:
                # safe fallback to first available dataframe
                try:
                    market_df = next(iter(universe_data.values()))
                except StopIteration:
                    print("❌ No valid market data found to initialize regime detector")
                    return False

            # Train regime detector -- handle exceptions inside fit to avoid crashing
            try:
                self.regime_detector.fit(market_df)
            except Exception as e:
                self._log(f"Regime detector training failed: {e}")

            # Train multi-factor model (may be heavy)
            try:
                self.risk_model.fit(universe_data)
            except Exception as e:
                self._log(f"Risk model training failed: {e}")

            self.is_initialized = True
            self._log("✅ Adaptive Risk Engine initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Error initializing Adaptive Risk Engine: {e}")
            return False

    def process_signals(self, signals: Dict[str, float], universe_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Process incoming signals and return a final allocation dict (normalized)."""
        if not isinstance(signals, dict) or not signals:
            return {}

        if not self.is_initialized:
            self._log("⚠️ Engine not initialized — using simple allocation fallback")
            return self._simple_allocation(signals)

        self._log("🎯 Processing signals through Adaptive Risk Engine...")

        # Detect regime (safe)
        regime, confidence = self._detect_regime_safe(universe_data)

        # If regime changed, log it
        if regime != self.last_regime:
            self._log(f"🔄 Regime change detected: {self.last_regime} → {regime}")
            self.last_regime = regime

        # Get adaptive allocation from allocator (handles many checks internally)
        adaptive_weights = {}
        try:
            adaptive_weights = self.risk_allocator.allocate_risk_budget(signals, universe_data)
        except Exception as e:
            self._log(f"risk_allocator failed: {e}")
            adaptive_weights = self._simple_allocation(signals)

        # Apply regime-specific adjustments
        try:
            regime_adjusted_weights = self._apply_regime_adjustments(adaptive_weights, regime, universe_data)
        except Exception as e:
            self._log(f"_apply_regime_adjustments failed: {e}")
            regime_adjusted_weights = adaptive_weights

        # Optionally apply hedging
        hedged_weights = regime_adjusted_weights
        if self.config.get('enable_hedging', True):
            try:
                hedged_weights = self._apply_tail_risk_hedging(regime_adjusted_weights, regime, universe_data)
            except Exception as e:
                self._log(f"_apply_tail_risk_hedging failed: {e}")
                hedged_weights = regime_adjusted_weights

        # Final risk limits and checks
        try:
            final_weights = self._apply_risk_limits(hedged_weights, universe_data)
        except Exception as e:
            self._log(f"_apply_risk_limits failed: {e}")
            final_weights = hedged_weights

        # Ensure non-negative and normalized
        final_weights = self._clip_and_normalize(final_weights)

        # Generate risk report (best-effort)
        try:
            risk_report = self.risk_allocator.get_risk_report(final_weights, universe_data)
        except Exception as e:
            self._log(f"get_risk_report failed: {e}")
            risk_report = {'portfolio_metrics': {}}

        # Update state & performance history
        self.last_allocation = final_weights
        try:
            self._update_performance_history(risk_report)
        except Exception as e:
            self._log(f"_update_performance_history failed: {e}")

        # Logging summary (safe access)
        expected_vol = risk_report.get('portfolio_metrics', {}).get('expected_volatility', 0.0)
        self._log("✅ Adaptive allocation complete")
        try:
            self._log(f"📊 Regime: {regime} (confidence: {confidence:.2%})")
        except Exception:
            self._log(f"📊 Regime: {regime} (confidence: {confidence})")
        try:
            self._log(f"🎯 Portfolio volatility: {expected_vol:.2%}")
        except Exception:
            self._log(f"🎯 Portfolio volatility: {expected_vol}")

        return final_weights

    def _detect_regime_safe(self, universe_data: Dict[str, pd.DataFrame]) -> Tuple[str, float]:
        """Safe wrapper around regime detection; returns (regime, confidence)."""
        try:
            if not universe_data:
                return 'Unknown', 0.0
            market_df = universe_data.get('SPY')
            if market_df is None:
                market_df = next(iter(universe_data.values()), pd.DataFrame())
            return self.regime_detector.predict_regime(market_df)
        except Exception as e:
            self._log(f"_detect_regime_safe failed: {e}")
            return 'Unknown', 0.0

    def _apply_regime_adjustments(self, weights: Dict[str, float],
                                  regime: str, universe_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Apply regime-specific multipliers to weights, then renormalize."""
        if not weights:
            return {}

        try:
            regime_adjustments = self.regime_detector.get_regime_risk_adjustments(regime)
        except Exception:
            regime_adjustments = {}

        position_multiplier = float(regime_adjustments.get('position_size_multiplier', 1.0))
        volatility_multiplier = float(regime_adjustments.get('volatility_multiplier', 1.0))

        adjusted: Dict[str, float] = {}
        for t, w in weights.items():
            try:
                w = float(w)
            except Exception:
                w = 0.0
            adj = w * position_multiplier
            # If volatility multiplier >1, shrink weights further (defensive)
            if volatility_multiplier > 1.0:
                adj = adj / volatility_multiplier
            adjusted[t] = max(adj, 0.0)

        return self._clip_and_normalize(adjusted)

    def _apply_tail_risk_hedging(self, weights: Dict[str, float],
                                 regime: str, universe_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Apply tail risk hedging: shrink positions and optionally add a hedge instrument."""
        if not weights:
            return {}

        try:
            regime_adjustments = self.regime_detector.get_regime_risk_adjustments(regime)
        except Exception:
            regime_adjustments = {}

        tail_risk_multiplier = float(regime_adjustments.get('tail_risk_multiplier', 1.0))
        hedged = {}

        if tail_risk_multiplier > 1.5:
            # shrink existing positions
            for t, w in weights.items():
                try:
                    hedged[t] = max(float(w) / tail_risk_multiplier, 0.0)
                except Exception:
                    hedged[t] = 0.0

            # optionally add a small market hedge (short)
            if 'SPY' in universe_data:
                hedge_weight = min(0.05, tail_risk_multiplier * 0.02)
                # negative weight indicates short hedge
                hedged['SPY_HEDGE'] = -abs(float(hedge_weight))

            # if sum is zero or all negative (rare), fallback to original weights
            total = sum(hedged.values())
            if total == 0 or np.isclose(total, 0.0):
                return self._clip_and_normalize(weights)
            # Renormalize while preserving sign distribution (normalize positives separately)
            pos_sum = sum([v for v in hedged.values() if v > 0])
            if pos_sum > 0:
                for k in list(hedged.keys()):
                    if hedged[k] > 0:
                        hedged[k] = hedged[k] / pos_sum
            else:
                # no positive weights, fallback
                return self._clip_and_normalize(weights)

            return hedged

        # Not a high tail-risk regime: return original weights
        return self._clip_and_normalize(weights)

    def _apply_risk_limits(self, weights: Dict[str, float],
                           universe_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Apply max position limits and correlation-based shrinkage; then renormalize."""
        if not weights:
            return {}

        max_pos = float(self.config.get('max_position_size', 0.10))
        limited: Dict[str, float] = {}
        for t, w in weights.items():
            try:
                limited[t] = min(float(w), max_pos)
            except Exception:
                limited[t] = 0.0

        # Correlation shrinkage (best-effort)
        try:
            corr_threshold = float(self.config.get('correlation_threshold', 0.8))
            corr = self.risk_allocator._calculate_correlation_matrix(universe_data)
            if corr is not None and not corr.empty:
                for t in list(limited.keys()):
                    if t in corr.columns:
                        high_corr = (corr[t] > corr_threshold)
                        if high_corr.any():
                            limited[t] *= 0.8
        except Exception as e:
            self._log(f"_apply_risk_limits correlation adjustment failed: {e}")

        return self._clip_and_normalize(limited)

    def _clip_and_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights are non-negative and sum to 1 (unless empty)."""
        if not weights:
            return {}

        clipped = {k: float(max(0.0, v)) for k, v in weights.items()}
        s = sum(clipped.values())
        if s <= 0:
            return {}
        return {k: v / s for k, v in clipped.items()}

    def _simple_allocation(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Simple normalized allocation proportional to signals (fallback)."""
        if not signals:
            return {}
        positive = {k: max(0.0, float(v)) for k, v in signals.items()}
        s = sum(positive.values())
        if s <= 0:
            return {}
        return {k: v / s for k, v in positive.items()}

    def _update_performance_history(self, risk_report: Dict):
        """Append summary metrics to performance history (bounded list)."""
        timestamp = pd.Timestamp.now()
        regime = risk_report.get('regime', {}).get('current_regime', 'Unknown')
        pm = risk_report.get('portfolio_metrics', {})
        volatility = pm.get('expected_volatility', 0.0)
        diversification = pm.get('diversification_ratio', 1.0)

        self.performance_history.append({
            'timestamp': str(timestamp),
            'regime': regime,
            'volatility': float(volatility),
            'diversification': float(diversification)
        })

        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_performance_attribution(self) -> Dict[str, Any]:
        """Return lightweight performance attribution from stored history."""
        if not self.performance_history:
            return {}

        df = pd.DataFrame(self.performance_history)
        try:
            attribution = {
                'regime_performance': df.groupby('regime')['volatility'].mean().to_dict(),
                'volatility_trend': float(df['volatility'].tail(20).mean()) if len(df) >= 1 else 0.0,
                'diversification_trend': float(df['diversification'].tail(20).mean()) if len(df) >= 1 else 0.0,
                'regime_stability': float(len(df['regime'].unique()) / len(df)) if len(df) > 0 else 0.0
            }
        except Exception as e:
            self._log(f"get_performance_attribution failed: {e}")
            attribution = {}
        return attribution

    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Return a payload suitable for the Risk Dashboard."""
        return {
            'current_regime': self.last_regime,
            'last_allocation': self.last_allocation,
            'performance_attribution': self.get_performance_attribution(),
            'config': self.config,
            'is_initialized': self.is_initialized
        }

    def save_state(self, filepath: str = "data/results/adaptive_risk_state.json"):
        """Persist engine state to disk (JSON)."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            state = {
                'last_regime': self.last_regime,
                'last_allocation': self.last_allocation,
                'performance_history': self.performance_history,
                'config': self.config,
                'is_initialized': self.is_initialized
            }
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            self._log(f"Saved engine state to {filepath}")
        except Exception as e:
            print(f"⚠️ save_state failed: {e}")

    def load_state(self, filepath: str = "data/results/adaptive_risk_state.json"):
        """Load engine state from disk if present."""
        try:
            p = Path(filepath)
            if not p.exists():
                self._log(f"No state file at {filepath}")
                return
            with open(p, 'r') as f:
                state = json.load(f)
            self.last_regime = state.get('last_regime', 'Unknown')
            self.last_allocation = state.get('last_allocation', {})
            self.performance_history = state.get('performance_history', [])
            self.is_initialized = state.get('is_initialized', False)
            self._log("✅ Adaptive Risk Engine state loaded")
        except Exception as e:
            print(f"⚠️ load_state failed: {e}")

# -- CLI / local test harness -------------------------------------------------
def main():
    """Quick local test harness to exercise initialization and allocation."""
    import glob

    data_files = glob.glob(str(Path("data").joinpath("*.csv")))
    if not data_files:
        print("❌ No data files found in data/ (expect CSVs with 'date' and 'close' columns)")
        return

    universe_data: Dict[str, pd.DataFrame] = {}
    for file in data_files[:10]:
        ticker = Path(file).stem.split('_')[0]
        try:
            df = pd.read_csv(file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
            universe_data[ticker] = df
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    engine = AdaptiveRiskEngine(verbose=True)
    if not engine.initialize(universe_data):
        print("❌ Engine initialization failed — exiting")
        return

    # Random example signals for first 5 tickers
    sample_signals = {t: float(np.random.random()) for t in list(universe_data.keys())[:5]}

    alloc = engine.process_signals(sample_signals, universe_data)
    print(f"\n📊 Sample Signals: {sample_signals}")
    print(f"🎯 Allocation: {alloc}")

    dashboard = engine.get_risk_dashboard_data()
    print(f"\n📈 Dashboard data summary: {dashboard}")

    engine.save_state()


if __name__ == "__main__":
    main()


