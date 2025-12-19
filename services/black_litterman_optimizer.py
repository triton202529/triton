#!/usr/bin/env python3
"""
Black-Litterman Optimizer (sanity-checked scaling and reporting)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class BLConfig:
    risk_aversion: float = 3.0  # λ
    tau: float = 0.05  # prior uncertainty scaling
    long_only: bool = True  # clip negative weights
    periods_per_year: int = 252  # to annualize


class BlackLittermanOptimizer:
    def __init__(self, cfg: BLConfig | None = None, verbose: bool = False):
        self.cfg = cfg or BLConfig()
        self.verbose = verbose

    # ---- helpers ----
    @staticmethod
    def _as_decimal_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
        df = returns_df.copy().dropna(how="any")
        try:
            med = df.abs().median(numeric_only=True).mean()
        except Exception:
            med = 0.0
        if med > 1.0:  # looks like percentage returns
            df = df / 100.0
        return df

    def _annualize(self, mu: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ppy = float(self.cfg.periods_per_year)
        return mu * ppy, cov * ppy

    # ---- views ----
    def _build_views(
        self, tickers: list[str], views: Dict[str, Tuple[str, float, float]], cov_ann: np.ndarray
    ):
        """
        views: { 'AAPL': ('absolute', 0.12, 0.7), ... }  # 12% annual expected return with 70% confidence
        Returns P (k x n), Q (k,), Omega (k x k)
        """
        k = len(views)
        n = len(tickers)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))

        idx = {t: i for i, t in enumerate(tickers)}
        for r, (tkr, (vtype, val, conf)) in enumerate(views.items()):
            i = idx.get(tkr)
            if i is None:
                continue
            P[r, i] = 1.0
            # Normalize 12 -> 0.12 if needed
            q = float(val) / 100.0 if abs(val) > 1.5 else float(val)
            Q[r] = q
            base = self.cfg.tau * (P[r : r + 1] @ cov_ann @ P[r : r + 1].T)[0, 0]
            conf = max(1e-6, min(1.0, float(conf)))
            Omega[r, r] = base / conf
        return P, Q, Omega

    # ---- main ----
    def run_black_litterman(
        self,
        returns_df: pd.DataFrame,
        market_caps: Dict[str, float],
        views: Dict[str, Tuple[str, float, float]] | None = None,
    ) -> Dict:
        """
        returns_df: daily returns (decimal) indexed by date, columns=tickers
        market_caps: map ticker -> cap (any positive scale)
        views: {ticker: ('absolute', annual_return_decimal, confidence_in_0_to_1)}
        """
        df = self._as_decimal_returns(returns_df)
        tickers = list(df.columns)
        n = len(tickers)
        if n == 0:
            raise ValueError("No tickers in returns_df")

        mu = df.mean().values  # daily
        cov = df.cov().values  # daily
        mu_ann, cov_ann = self._annualize(mu, cov)

        # market equilibrium weights from market caps
        mktw = np.array([max(0.0, float(market_caps.get(t, 1.0))) for t in tickers], dtype=float)
        if mktw.sum() == 0:
            mktw = np.ones(n, dtype=float)
        mktw /= mktw.sum()

        # equilibrium returns (pi = λ Σ w_mkt)
        lam = float(self.cfg.risk_aversion)
        pi = lam * cov_ann @ mktw

        # incorporate views (absolute only here)
        if not views:
            P = np.zeros((0, n))
            Q = np.zeros(0)
            Omega = np.zeros((0, 0))
        else:
            P, Q, Omega = self._build_views(tickers, views, cov_ann)

        tau = float(self.cfg.tau)
        inv_cov = np.linalg.inv(cov_ann)
        middle = np.linalg.inv(P @ (tau * cov_ann) @ P.T + Omega) if P.shape[0] > 0 else None

        if P.shape[0] > 0:
            mu_bl = pi + (tau * cov_ann) @ P.T @ middle @ (Q - P @ pi)
            cov_bl = cov_ann  # conservative & stable
        else:
            mu_bl, cov_bl = pi, cov_ann

        # mean-variance optimal weights: w* = (1/λ) Σ^{-1} μ
        inv_cov_bl = np.linalg.inv(cov_bl)
        w = (1.0 / lam) * inv_cov_bl @ mu_bl

        # long-only normalization if requested
        if self.cfg.long_only:
            w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()

        # portfolio metrics (annualized)
        exp_ret = float(w @ mu_ann)
        vol = float(np.sqrt(w @ cov_ann @ w))
        sharpe = exp_ret / (vol + 1e-12)

        weights = {t: float(w[i]) for i, t in enumerate(tickers)}
        if self.verbose:
            print("🎯 Optimization complete.")
        return {
            "optimal_weights": weights,
            "expected_return": exp_ret,  # decimal annual
            "volatility": vol,  # decimal annual
            "sharpe_ratio": sharpe,
            "posterior_mean": {t: float(mu_bl[i]) for i, t in enumerate(tickers)},
        }


if __name__ == "__main__":
    # Simple self-test with random returns
    np.random.seed(7)
    cols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    r = pd.DataFrame(np.random.normal(0.0005, 0.02, size=(600, len(cols))), columns=cols)
    caps = {t: np.random.uniform(1000, 3000) for t in cols}
    views = {"AAPL": ("absolute", 0.12, 0.7)}  # 12% annual with 70% confidence

    opt = BlackLittermanOptimizer(verbose=True)
    res = opt.run_black_litterman(r, caps, views)
    print("Volatility:", f"{res['volatility']:.2%}")
    print("Sharpe:", f"{res['sharpe_ratio']:.2f}")
    print(
        "Top weights:", sorted(res["optimal_weights"].items(), key=lambda x: x[1], reverse=True)[:3]
    )
