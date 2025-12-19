#!/usr/bin/env python3
"""
Deep Learning Fusion Engine (robust import + safe no-torch fallback)

- Exports: DeepLearningFusionEngine
- Works with or without PyTorch installed.
- If PyTorch is unavailable, the engine becomes a no-op that returns neutral scores.

Typical usage:
    from services.deep_learning_fusion_engine import DeepLearningFusionEngine
    eng = DeepLearningFusionEngine(model="lstm", hidden_dim=64, verbose=True)
    eng.fit(train_X, train_y)
    scores = eng.predict(test_X)   # list[float] in [0,1]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# --- Torch guard + safe stubs -------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

    class _NNStub:  # minimal stub to avoid NameError
        Module = object
    nn = _NNStub()  # type: ignore
# ------------------------------------------------------------------------------


# --------------------------- Torch models (optional) ---------------------------
if TORCH_AVAILABLE:
    class _LSTMBlock(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
                nn.ReLU(),
                nn.Linear(max(16, hidden_dim // 2), 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: [B, T, F]
            out, _ = self.lstm(x)
            last = out[:, -1, :]  # [B, H]
            return self.head(last).squeeze(-1)


    class _TransformerBlock(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, n_heads: int, n_layers: int, dropout: float):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=max(1, n_heads), dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: [B, T, F]
            enc = self.encoder(x)           # [B, T, F]
            pooled = enc.mean(dim=1)        # mean pool over time -> [B, F]
            return self.head(pooled).squeeze(-1)
# ------------------------------------------------------------------------------


@dataclass
class FusionConfig:
    model: str = "lstm"          # "lstm" | "transformer"
    hidden_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4             # transformer only
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 5
    device: str = "cpu"
    verbose: bool = False


class DeepLearningFusionEngine:
    """
    High-level wrapper that trains a small sequence model on multivariate time-series
    features and returns BUY/HOLD confidence in [0,1]. If torch is not available,
    falls back to a neutral scorer that returns 0.5 for all inputs.
    """

    def __init__(self, **kwargs):
        # accept only known config keys
        self.cfg = FusionConfig(**{k: v for k, v in kwargs.items() if k in FusionConfig.__annotations__})
        self.model = None
        self._fitted = False

        if TORCH_AVAILABLE:
            self.device = torch.device(self.cfg.device if torch.cuda.is_available() or self.cfg.device == "cpu" else "cpu")
        else:
            self.device = "cpu"

        if self.cfg.verbose:
            print(f"[Fusion] TORCH_AVAILABLE={TORCH_AVAILABLE} | model={self.cfg.model}")

    # ---------------------------- public API ----------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepLearningFusionEngine":
        """
        X: [N, T, F]  float
        y: [N]        labels in {0,1} or floats in [0,1]
        """
        if not TORCH_AVAILABLE:
            # No-op fit
            self._fitted = True
            return self

        X = self._to_float3d(X)
        y = self._to_float1d(y)
        N, T, F = X.shape

        if self.cfg.model.lower() == "transformer":
            net = _TransformerBlock(in_dim=F, hidden_dim=self.cfg.hidden_dim, n_heads=self.cfg.n_heads,
                                    n_layers=self.cfg.n_layers, dropout=self.cfg.dropout)
        else:
            net = _LSTMBlock(in_dim=F, hidden_dim=self.cfg.hidden_dim, n_layers=self.cfg.n_layers, dropout=self.cfg.dropout)

        net = net.to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.cfg.lr)
        loss_fn = nn.BCELoss()

        X_t = torch.from_numpy(X).float().to(self.device)        # [N, T, F]
        y_t = torch.from_numpy(y).float().to(self.device)        # [N]

        net.train()
        for ep in range(max(1, int(self.cfg.epochs))):
            opt.zero_grad()
            pred = net(X_t)
            # Ensure shapes match for BCE
            if getattr(pred, "dim", lambda: 1)() == 2 and pred.size(1) == 1:
                pred = pred.squeeze(1)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            if self.cfg.verbose:
                print(f"[Fusion] epoch {ep+1}/{self.cfg.epochs} | loss={float(loss):.4f}")

        self.model = net
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> List[float]:
        """
        X: [N, T, F] -> returns list of floats in [0,1]
        """
        # allow predict-before-fit in demo contexts by returning neutral 0.5
        if not self._fitted:
            try:
                n = len(X)
            except Exception:
                n = 0
            return [0.5] * n

        if not TORCH_AVAILABLE or self.model is None:
            try:
                n = len(X)
            except Exception:
                n = 0
            return [0.5] * n

        X = self._to_float3d(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().to(self.device)
            out = self.model(X_t)
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            scores = np.clip(np.asarray(out).reshape(-1).tolist(), 0.0, 1.0)
        return scores

    # --------------------------- utils ----------------------------------------
    @staticmethod
    def _to_float3d(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[None, :, :]  # [1, T, F]
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape [N,T,F], got {X.shape}")
        return X

    @staticmethod
    def _to_float1d(y) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        # normalize labels from {-1,1} or {0,1} to [0,1]
        if (y.min() < 0.0) or (y.max() > 1.0):
            y = (y > 0).astype(float)
        return y


# --- Ensure expected export name exists for downstream imports -----------------
try:
    DeepLearningFusionEngine  # noqa: F401
except NameError:  # pragma: no cover
    # Fallback aliasing if code above is modified
    class DeepLearningFusionEngine:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def fit(self, *a, **k): return self
        def predict(self, *a, **k): return []
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny smoke test
    np.random.seed(0)
    N, T, F = 32, 20, 8
    X = np.random.normal(size=(N, T, F))
    y = (np.random.rand(N) > 0.5).astype(float)
    eng = DeepLearningFusionEngine(verbose=True)
    eng.fit(X, y)
    print("Scores:", np.array(eng.predict(X[:5])))

