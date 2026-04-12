# services/regime_detector.py
"""
TRITON — Regime Detector (Phase 1.5)

Outputs:
- data/results/regimes.csv
- Adds/updates 'regime' column inside:
    - data/results/portfolio_history.csv
    - data/results/enhanced_portfolio_history.csv (if present)

This script is defensive:
- Accepts common column variants like Date/Close/Adj Close, etc.
- Will auto-detect best date + price columns if not exactly 'date'/'close'.
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Path bootstrap (Windows script runs)
# ─────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_col(c) for c in df.columns]
    return df


def _pick_first_existing(cols: List[str], df_cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df_cols:
            return c
    return None


def _detect_market_columns(df: pd.DataFrame, price_col_hint: str = "close") -> Tuple[str, str]:
    """
    Return (date_col, price_col) from a market price df.

    Accepts common variants:
      - date: date, datetime, timestamp, time
      - price: close, adj_close, adjusted_close, last, price
    """
    df_cols = list(df.columns)

    date_candidates = ["date", "datetime", "timestamp", "time"]
    price_candidates = [
        price_col_hint,
        _norm_col(price_col_hint),
        "close",
        "adj_close",
        "adjusted_close",
        "adjusted",
        "last",
        "price",
        "settle",
    ]

    date_col = _pick_first_existing(date_candidates, df_cols)
    price_col = _pick_first_existing(price_candidates, df_cols)

    if date_col is None:
        raise ValueError(f"Could not detect date column. Found columns: {df_cols}")
    if price_col is None:
        raise ValueError(f"Could not detect price column. Found columns: {df_cols}")

    return date_col, price_col


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    return df


def _normalize_date_only(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.floor("D")


# ─────────────────────────────────────────────────────────────
# RegimeDetector
# ─────────────────────────────────────────────────────────────


class RegimeDetector:
    def __init__(
        self,
        lookback_days: int = 252,
        min_samples: int = 50,
        price_col: str = "close",
        verbose: bool = False,
    ):
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.price_col = price_col
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        )
        self.is_fitted = False

    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        prices = prices.astype(float).copy()
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace({0: np.nan})
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df.columns:
            raise ValueError(f"Price column '{self.price_col}' not found in DataFrame")

        features = df.copy().reset_index(drop=True)
        if "date" in features.columns:
            features["date"] = pd.to_datetime(features["date"], errors="coerce")

        price = pd.to_numeric(features[self.price_col], errors="coerce").astype(float)

        features["returns"] = price.pct_change(fill_method=None)
        with np.errstate(divide="ignore", invalid="ignore"):
            features["log_returns"] = np.log(price / price.shift(1))

        features["vol_5d"] = features["returns"].rolling(5, min_periods=1).std(ddof=0)
        features["vol_20d"] = features["returns"].rolling(20, min_periods=1).std(ddof=0)
        features["vol_ratio"] = features["vol_5d"] / features["vol_20d"].replace({0: np.nan})

        features["sma_20"] = price.rolling(20, min_periods=1).mean()
        features["sma_50"] = price.rolling(50, min_periods=1).mean()
        features["trend_strength"] = (features["sma_20"] - features["sma_50"]) / features[
            "sma_50"
        ].replace({0: np.nan})

        features["rsi_14"] = self._calculate_rsi(price, 14)
        features["momentum_5d"] = price.pct_change(periods=5)
        features["momentum_20d"] = price.pct_change(periods=20)

        features["vol_cluster"] = features["vol_5d"].rolling(20, min_periods=1).mean()
        features["vol_regime"] = (features["vol_5d"] > features["vol_cluster"] * 1.5).astype(int)

        features["cummax"] = price.cummax()
        features["drawdown"] = (price - features["cummax"]) / features["cummax"].replace(
            {0: np.nan}
        )
        features["max_dd_20d"] = features["drawdown"].rolling(20, min_periods=1).min()

        features.loc[~np.isfinite(price), :] = np.nan
        return features

    def _label_regimes(self, features: pd.DataFrame) -> pd.Series:
        idx = features.index
        regimes = pd.Series(index=idx, dtype="object")
        regimes[:] = "Unknown"

        req = ["returns", "vol_20d", "trend_strength"]
        for c in req:
            if c not in features.columns:
                features[c] = np.nan

        returns_20d = features["returns"].rolling(20, min_periods=1).mean()
        vol_20d = features["vol_20d"]
        trend_strength = features["trend_strength"].fillna(0)

        valid_mask = (~returns_20d.isna()) & (~vol_20d.isna())

        try:
            high_vol_threshold = vol_20d.quantile(0.80)
            low_vol_threshold = vol_20d.quantile(0.30)
        except Exception:
            high_vol_threshold = np.nan
            low_vol_threshold = np.nan

        if np.isfinite(high_vol_threshold):
            mask_volatile = valid_mask & (vol_20d > high_vol_threshold)
            regimes.loc[mask_volatile] = "Volatile"
        else:
            mask_volatile = pd.Series(False, index=idx)

        mask_bear = valid_mask & (returns_20d < -0.01) & (trend_strength < -0.05) & (~mask_volatile)
        regimes.loc[mask_bear] = "Bear"

        mask_bull = valid_mask & (returns_20d > 0.01) & (trend_strength > 0.05) & (~mask_volatile)
        regimes.loc[mask_bull] = "Bull"

        if np.isfinite(low_vol_threshold):
            mask_sideways = (
                valid_mask
                & (vol_20d < low_vol_threshold)
                & (trend_strength.abs() < 0.02)
                & (~mask_volatile)
            )
            regimes.loc[mask_sideways] = "Sideways"
        else:
            mask_sideways = pd.Series(False, index=idx)

        mask_remaining = (
            valid_mask & (~mask_volatile) & (~mask_bear) & (~mask_bull) & (~mask_sideways)
        )
        regimes.loc[mask_remaining & (trend_strength > 0)] = "Bull"
        regimes.loc[mask_remaining & (trend_strength <= 0)] = "Bear"

        return regimes

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        try:
            features = self._calculate_features(df)
        except Exception as e:
            print(f"⚠️ Failed to calculate features for training: {e}")
            return self

        regimes = self._label_regimes(features)

        feature_cols = [
            "vol_5d",
            "vol_20d",
            "vol_ratio",
            "trend_strength",
            "rsi_14",
            "momentum_5d",
            "momentum_20d",
            "vol_regime",
            "max_dd_20d",
        ]

        valid_mask = features[feature_cols].notna().all(axis=1) & (~regimes.isin(["Unknown"]))
        X = features.loc[valid_mask, feature_cols]
        y = regimes.loc[valid_mask]

        if len(X) < self.min_samples:
            self.is_fitted = False
            return self

        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

        return self

    def label_series(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._calculate_features(df)
        regimes = self._label_regimes(features)

        out = pd.DataFrame({"date": df["date"].copy()})
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.floor("D")
        out["regime"] = regimes.values

        out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        out = out.sort_values("date").reset_index(drop=True)
        return out


# ─────────────────────────────────────────────────────────────
# Phase 1.5: Artifact writer
# ─────────────────────────────────────────────────────────────


def write_regime_artifacts(
    market_price_csv: Path,
    results_dir: Path = Path("data/results"),
    portfolio_files: Tuple[Path, ...] = (
        Path("data/results/portfolio_history.csv"),
        Path("data/results/enhanced_portfolio_history.csv"),
    ),
    price_col_hint: str = "close",
    verbose: bool = False,
) -> int:
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_csv_safe(market_price_csv)
    if raw is None:
        print(f"❌ Market price file missing/empty: {market_price_csv}")
        return 2

    raw = _normalize_headers(raw)

    try:
        date_col, price_col = _detect_market_columns(raw, price_col_hint=price_col_hint)
    except Exception as e:
        print(f"❌ {e}")
        return 2

    mkt = raw[[date_col, price_col]].copy()
    mkt = mkt.rename(columns={date_col: "date", price_col: "close"})

    mkt = _ensure_datetime(mkt, "date")
    mkt = mkt.sort_values("date").reset_index(drop=True)

    det = RegimeDetector(price_col="close", verbose=verbose)

    # Fit is optional; labels are rule-based. Fit helps future expansion.
    try:
        det.fit(mkt)
    except Exception:
        pass

    regimes_df = det.label_series(mkt)
    regimes_path = results_dir / "regimes.csv"
    regimes_df.to_csv(regimes_path, index=False)
    print(f"✅ Wrote {regimes_path} ({len(regimes_df)} rows)")

    # Merge into portfolio files
    for pf in portfolio_files:
        dfp = _read_csv_safe(pf)
        if dfp is None:
            continue

        dfp = _normalize_headers(dfp)
        if "date" not in dfp.columns:
            print(f"⚠️ Skipping merge (no date col): {pf}")
            continue

        dfp = _ensure_datetime(dfp, "date")
        dfp["date"] = _normalize_date_only(dfp["date"])

        merged = dfp.merge(regimes_df, on="date", how="left")

        # If portfolio already had a regime col, keep it unless missing/Unknown
        if "regime_x" in merged.columns and "regime_y" in merged.columns:
            merged["regime"] = merged["regime_x"]
            m = merged["regime"].isna() | (merged["regime"].astype(str).str.upper() == "UNKNOWN")
            merged.loc[m, "regime"] = merged.loc[m, "regime_y"]
            merged = merged.drop(columns=["regime_x", "regime_y"], errors="ignore")

        merged = merged.sort_values("date").reset_index(drop=True)

        # Write back with original filename
        merged.to_csv(pf, index=False)
        print(f"✅ Updated {pf} (added/filled regime)")

    return 0


def _auto_find_market_proxy() -> Optional[Path]:
    # Prefer your already-found SPY file if present
    preferred = Path("data/SPY_2020-07-07_to_2025-07-06.csv")
    if preferred.exists() and preferred.stat().st_size > 0:
        return preferred

    candidates = [
        Path("data/processed/SPY.csv"),
        Path("data/SPY.csv"),
        Path("data/results/SPY.csv"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c

    for base in [Path("data/processed"), Path("data"), Path("data/results")]:
        if base.exists():
            for p in base.glob("*.csv"):
                if "SPY" in p.name.upper() and p.stat().st_size > 0:
                    return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--market-csv",
        type=str,
        default="",
        help="CSV containing Date+Close (any common header form)",
    )
    ap.add_argument(
        "--price-col", type=str, default="close", help="hint: close / adj_close / price"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    market_csv = Path(args.market_csv) if args.market_csv else _auto_find_market_proxy()
    if market_csv is None:
        print("❌ Could not auto-find a market CSV (SPY). Provide --market-csv path/to.csv")
        return 2

    print(f"📌 Using market proxy: {market_csv}")
    return write_regime_artifacts(
        market_price_csv=market_csv,
        results_dir=Path("data/results"),
        price_col_hint=args.price_col,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
