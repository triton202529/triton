# services/buffett_allocator.py
from __future__ import annotations

import os
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# Default excludes (override from CLI if desired)
DEFAULT_TICKERS_EXCLUDE = ["UNG", "WFC", "GE"]

# ---- Vendor column coalescers ------------------------------------------------

COLMAP = {
    "pe": ["pe", "trailing_pe", "pe_ratio"],
    "pb": ["pb", "price_to_book", "pb_ratio"],
    "roe": ["roe", "return_on_equity", "roe_ttm"],
    "roa": ["roa", "return_on_assets", "roa_ttm"],
    "margin": ["net_margin", "profit_margin", "net_profit_margin"],
    "de_ratio": ["de", "de_ratio", "debt_to_equity"],
    "fcf_yield": ["fcf_yield", "free_cash_flow_yield"],
    "op_margin": ["operating_margin", "op_margin"],
    "rev_growth": ["revenue_growth", "rev_growth_yoy", "revenue_growth_ttm"],
    "eps_growth": ["eps_growth", "eps_growth_yoy", "eps_growth_ttm"],
}

PERCENT_LIKE = {
    "roe",
    "roa",
    "margin",
    "op_margin",
    "rev_growth",
    "eps_growth",
    "fcf_yield",
}


def _coalesce(df: pd.DataFrame, name: str) -> Optional[str]:
    """Return the first column in df that matches the semantic 'name'."""
    for c in COLMAP.get(name, []):
        if c in df.columns:
            return c
    return None


def _to_float(x):
    """Lenient str→float with %, commas, and placeholder handling."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    if s in {"-", "", "na", "n/a", "none", "null"}:
        return np.nan
    s = s.replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _autoscale_percent(col: pd.Series) -> pd.Series:
    """
    If the series is mostly in [−1,1], treat as fraction and scale by 100 → percent space.
    Returns float series (NaNs preserved).
    """
    if col is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(col, errors="coerce")
    sample = s.dropna()
    if len(sample) == 0:
        return s
    frac_ratio = (sample.abs() <= 1.0).mean()
    if frac_ratio > 0.8:
        return s * 100.0
    return s


def _zscore_safe(series: pd.Series) -> pd.Series:
    """NaN-safe z-score with tiny epsilon to avoid div/0."""
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return s * 0  # all zeros; avoids NaNs
    return (s - mu) / (sd + 1e-9)


class BuffettAllocator:
    def __init__(
        self,
        max_weight: float = 0.15,
        max_sell_fraction: float = 0.33,
        rebalance_days: int = 1,
        guardrails: str = "loose",  # "strict" | "loose"
        excludes: Optional[List[str]] = None,
        debug: bool = False,
    ):
        self.max_weight = float(max_weight)
        self.max_sell_fraction = float(max_sell_fraction)
        self.rebalance_days = int(rebalance_days)
        self.guardrails = guardrails
        self.excludes = set(e.upper() for e in (excludes or []))
        self.debug = debug

    # ---- Quality / Value / Growth components --------------------------------

    def _quality_mask(self, df: pd.DataFrame) -> pd.Series:
        """Basic quality filters used when guardrails=='strict'."""
        m = pd.Series(True, index=df.index)

        roe_col = _coalesce(df, "roe")
        if roe_col:
            roe = _autoscale_percent(df[roe_col].map(_to_float))
            m &= roe > 5  # >5% ROE

        margin_col = _coalesce(df, "margin")
        if margin_col:
            nm = _autoscale_percent(df[margin_col].map(_to_float))
            m &= nm > 0  # positive net margins

        de_col = _coalesce(df, "de_ratio")
        if de_col:
            de = pd.to_numeric(df[de_col].map(_to_float), errors="coerce")
            m &= (de < 3.0) | (de.isna())  # allow unknowns but skip extreme leverage

        return m

    def _value_boost(self, df: pd.DataFrame) -> pd.Series:
        """Reward cheaper names by PE/PB inverses (z-scored)."""
        pe_col = _coalesce(df, "pe")
        pb_col = _coalesce(df, "pb")

        pe = (
            pd.to_numeric(df[pe_col].map(_to_float), errors="coerce")
            if pe_col
            else pd.Series(np.nan, index=df.index)
        )
        pb = (
            pd.to_numeric(df[pb_col].map(_to_float), errors="coerce")
            if pb_col
            else pd.Series(np.nan, index=df.index)
        )

        inv_pe = 1.0 / pe.replace(0, np.nan)
        inv_pb = 1.0 / pb.replace(0, np.nan)

        z = pd.concat([_zscore_safe(inv_pe), _zscore_safe(inv_pb)], axis=1)
        return z.mean(axis=1).fillna(0.0)

    def _growth_boost(self, df: pd.DataFrame) -> pd.Series:
        """
        Your cleaned function: uses revenue & EPS growth if available,
        auto-scales fraction→percent, z-scores, averages, NaN→0.
        """
        rg = _coalesce(df, "rev_growth")
        eg = _coalesce(df, "eps_growth")

        r = _autoscale_percent(df[rg].map(_to_float)) if rg else pd.Series(np.nan, index=df.index)
        e = _autoscale_percent(df[eg].map(_to_float)) if eg else pd.Series(np.nan, index=df.index)

        z = pd.concat([_zscore_safe(r), _zscore_safe(e)], axis=1)
        return z.mean(axis=1).fillna(0.0)

    # ---- Score composition ---------------------------------------------------

    def _normalize_base_score(self, col: pd.Series) -> pd.Series:
        """
        Normalize base score to ~0..1 if needed.
        Heuristics:
          - If median <= 1.5 → assume already 0..1.
          - Else if max <= 100 → treat as 0..100 and divide by 100.
          - Else z-score transform as last resort then rescale to ~0..1 via CDF-ish clamp.
        """
        s = pd.to_numeric(col, errors="coerce")
        med = s.median()
        mx = s.max()

        if pd.isna(med) or pd.isna(mx):
            return s.fillna(0.0)

        if med <= 1.5:
            return s.clip(lower=0.0)  # already fractional
        if mx <= 100.0:
            return (s / 100.0).clip(lower=0.0, upper=1.0)

        # Fallback: z-score then squash to (0,1) with a simple logistic-ish clamp
        z = _zscore_safe(s).clip(-3, 3)
        return (z + 3) / 6.0  # maps [-3,3] → [0,1]

    def _compose_score(self, df: pd.DataFrame, base_score_col: str) -> pd.Series:
        base = self._normalize_base_score(df[base_score_col])
        value = self._value_boost(df)
        growth = self._growth_boost(df)
        return (0.6 * base) + (0.25 * value) + (0.15 * growth)

    # ---- Main planning API ---------------------------------------------------

    def plan(
        self,
        scores_df: pd.DataFrame,
        score_col: str,
        fundamentals_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge scores with fundamentals (optional), apply guardrails, compose final
        score, and produce target weights capped by max_weight.
        Returns (preview_df, weights_df).
        """
        if "ticker" not in scores_df.columns:
            raise KeyError("Scores CSV must contain 'ticker'")

        df = scores_df.copy()
        df["ticker"] = df["ticker"].astype(str).str.upper()
        if self.excludes:
            df = df[~df["ticker"].isin(self.excludes)]

        # Optional fundamentals join
        fpath = (
            fundamentals_path if (fundamentals_path and os.path.exists(fundamentals_path)) else None
        )
        if fpath:
            fdf = pd.read_csv(fpath)
            if "ticker" in fdf.columns:
                fdf["ticker"] = fdf["ticker"].astype(str).str.upper()
                df = df.merge(fdf, on="ticker", how="left")

        # Guardrails
        mask = (
            self._quality_mask(df)
            if self.guardrails == "strict"
            else pd.Series(True, index=df.index)
        )
        df = df[mask].copy()

        # Compose final score: if fundamentals are present, add value/growth boosts
        final_score = (
            self._compose_score(df, score_col)
            if fpath
            else self._normalize_base_score(df[score_col])
        )
        df["buffett_score"] = pd.to_numeric(final_score, errors="coerce").fillna(0.0)

        # Rank and cap
        df = df.sort_values("buffett_score", ascending=False).reset_index(drop=True)
        if df.empty:
            return df, pd.DataFrame(columns=["ticker", "target_weight"])

        z = _zscore_safe(df["buffett_score"])
        pos = np.maximum(z, 0)
        if float(pos.sum()) == 0.0:
            pos = np.ones(len(df), dtype=float)

        w = pos / pos.sum()
        w = np.minimum(w, self.max_weight)
        w = w / w.sum()  # re-normalize after cap

        weights = pd.DataFrame(
            {
                "ticker": df["ticker"].values,
                "target_weight": w,
            }
        )

        preview_cols = [c for c in ["ticker", score_col, "buffett_score"] if c in df.columns]
        preview = df[preview_cols].copy()
        return preview, weights
