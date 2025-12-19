#!/usr/bin/env python3
"""
Enhanced Signal Generator (robust fusion import + graceful baseline)

Generates per-ticker signals with optional fusion and optional adaptive risk
adjustments. If the fusion engine cannot be imported or PyTorch isn’t available,
it falls back to a simple momentum baseline.

Exports:
- EnhancedSignalGenerator
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------- robust fusion import (tolerant) --------------------------
FUSION_AVAILABLE = False
try:
    from services.deep_learning_fusion_engine import DeepLearningFusionEngine
    FUSION_AVAILABLE = True
except Exception as _e:
    DeepLearningFusionEngine = None  # type: ignore
    print(f"[EnhancedSignalGenerator] Fusion engine unavailable: {_e}")
# ------------------------------------------------------------------------------


@dataclass
class ESGConfig:
    use_fusion: bool = True
    use_adaptive_risk: bool = True
    verbose: bool = False
    lookback: int = 20     # for simple momentum baseline
    fusion_kwargs: dict = None


class EnhancedSignalGenerator:
    def __init__(self, use_fusion: bool = True, use_adaptive_risk: bool = True,
                 verbose: bool = False, **kwargs):
        self.cfg = ESGConfig(
            use_fusion=use_fusion,
            use_adaptive_risk=use_adaptive_risk,
            verbose=verbose,
            fusion_kwargs=kwargs.get("fusion_kwargs") or {}
        )
        self.use_fusion = bool(self.cfg.use_fusion) and FUSION_AVAILABLE
        self.fusion = None
        if self.use_fusion:
            try:
                self.fusion = DeepLearningFusionEngine(**self.cfg.fusion_kwargs)  # type: ignore
            except Exception as e:
                print(f"[EnhancedSignalGenerator] Failed to init fusion engine: {e}")
                self.use_fusion, self.fusion = False, None

    # ------------------------ main API ----------------------------------------
    def generate_signals(
        self,
        universe_data: Dict[str, pd.DataFrame],
        model_predictions: Optional[Dict[str, float]] = None,
        sentiment_data: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        universe_data: {ticker: DataFrame[date, close, (volume?)]}
        model_predictions: {ticker: expected_return (decimal)}
        sentiment_data: {ticker: sentiment score [-1..1] or 0..1 mapped}
        """
        model_predictions = model_predictions or {}
        sentiment_data = sentiment_data or {}

        if self.use_fusion and self.fusion is not None:
            try:
                return self._generate_with_fusion(universe_data, model_predictions, sentiment_data)
            except Exception as e:
                print(f"[EnhancedSignalGenerator] Fusion path failed: {e}")
                # Fall through to baseline
        return self._generate_baseline(universe_data)

    def apply_adaptive_risk(self, signals_df: pd.DataFrame, universe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Optional post-processing to tilt or cap signals.
        This is a light placeholder that scales confidence by realized volatility.
        """
        if not self.cfg.use_adaptive_risk:
            return signals_df

        rows = []
        for _, r in signals_df.iterrows():
            t = r["ticker"]
            conf = float(r.get("confidence", 0.5))
            df = universe_data.get(t)
            if df is None or len(df) < 21:
                rows.append(conf)
                continue
            vol = float(df["close"].pct_change().rolling(20).std().iloc[-1] or 0.0)
            # Downscale confidence when recent vol is high
            adj = conf / (1.0 + 5.0 * max(0.0, vol))
            rows.append(max(0.0, min(1.0, adj)))
        out = signals_df.copy()
        out["confidence"] = rows
        return out

    def save_config(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "use_fusion": self.use_fusion,
            "use_adaptive_risk": self.cfg.use_adaptive_risk,
            "fusion_kwargs": self.cfg.fusion_kwargs or {},
            "lookback": self.cfg.lookback,
        }
        import json
        p.write_text(json.dumps(cfg, indent=2))

    # ------------------------ internals ---------------------------------------
    def _generate_baseline(self, universe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        lookback = int(max(2, self.cfg.lookback))
        rows: List[dict] = []
        for tkr, df in universe_data.items():
            if df is None or len(df) <= lookback:
                continue
            c = df["close"].values
            ret = (float(c[-1]) / float(c[-lookback]) - 1.0) if float(c[-lookback]) != 0 else 0.0
            sig = "BUY" if ret > 0 else "HOLD"
            rows.append({"ticker": tkr, "signal": sig, "confidence": float(abs(ret))})
        return pd.DataFrame(rows)

    def _stack_features(
        self,
        universe_data: Dict[str, pd.DataFrame],
        model_predictions: Dict[str, float],
        sentiment_data: Dict[str, float],
        T: int = 20
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Build a simple [N, T, F] tensor using returns, rolling vol, and extras.
        y is a pseudo-label derived from forward return sign (demo purpose).
        """
        tickers, X_list, y_list = [], [], []
        for tkr, df in universe_data.items():
            if len(df) < T + 2:
                continue
            close = df["close"].astype(float).values
            ret = np.diff(np.log(close))  # [L-1]
            if len(ret) < T + 1:
                continue
            # recent window
            r_win = ret[-T:]                     # [T]
            vol_win = pd.Series(ret).rolling(10).std().fillna(0.0).values[-T:]
            pred = float(model_predictions.get(tkr, 0.0))
            sent = float(sentiment_data.get(tkr, 0.0))
            # features: [ret, vol, pred, sent] -> [T, 4]
            X = np.stack([r_win, vol_win,
                          np.full_like(r_win, pred, dtype=float),
                          np.full_like(r_win, sent, dtype=float)], axis=1)
            # pseudo target: forward 5-day return sign
            fwd = (close[-1] / close[-6] - 1.0) if len(close) >= 6 and close[-6] != 0 else 0.0
            y = 1.0 if fwd > 0 else 0.0

            tickers.append(tkr)
            X_list.append(X)
            y_list.append(y)

        if not X_list:
            return [], np.zeros((0, T, 4), dtype=float), np.zeros((0,), dtype=float)
        return tickers, np.stack(X_list, axis=0), np.array(y_list, dtype=float)

    def _generate_with_fusion(
        self,
        universe_data: Dict[str, pd.DataFrame],
        model_predictions: Dict[str, float],
        sentiment_data: Dict[str, float]
    ) -> pd.DataFrame:
        # Build features
        tickers, X, y = self._stack_features(universe_data, model_predictions, sentiment_data, T=20)
        if len(tickers) == 0:
            return self._generate_baseline(universe_data)
        # Light fit (demo-level) then predict own window
        try:
            self.fusion.fit(X, y)   # type: ignore
            scores = self.fusion.predict(X)  # type: ignore
        except Exception as e:
            print(f"[EnhancedSignalGenerator] Fusion training/prediction failed: {e}")
            return self._generate_baseline(universe_data)

        rows = []
        for tkr, s in zip(tickers, scores):
            sig = "BUY" if float(s) >= 0.5 else "HOLD"
            rows.append({"ticker": tkr, "signal": sig, "confidence": float(np.clip(s, 0.0, 1.0))})
        return pd.DataFrame(rows)

