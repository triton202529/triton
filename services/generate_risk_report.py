# services/generate_risk_report.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

PORTFOLIO_CANDIDATES = [
    RESULTS_DIR / "enhanced_portfolio_history.csv",
    RESULTS_DIR / "portfolio_history.csv",
]

RISK_REPORT_FILE = RESULTS_DIR / "risk_report.json"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _read_portfolio_file() -> Path:
    for p in PORTFOLIO_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            return p
    tried = ", ".join(str(p) for p in PORTFOLIO_CANDIDATES)
    raise FileNotFoundError(f"Portfolio file not found or empty. Tried: {tried}")


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["date", "cash", "market_value", "total_value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Portfolio history missing required columns: {missing}. Found: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cash"] = pd.to_numeric(df["cash"], errors="coerce")
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")

    # Optional fields (do NOT require)
    if "num_positions" in df.columns:
        df["num_positions"] = pd.to_numeric(df["num_positions"], errors="coerce")
    else:
        df["num_positions"] = np.nan

    if "regime" not in df.columns:
        df["regime"] = np.nan

    df = df.dropna(subset=["date", "total_value"]).sort_values("date").reset_index(drop=True)
    return df


def _returns(df: pd.DataFrame) -> pd.Series:
    r = df["total_value"].pct_change()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def _max_drawdown(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(v)
    dd = (v / peak) - 1.0
    return float(dd.min())  # negative


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────


def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or df["total_value"].dropna().shape[0] < 2:
        return {}

    tv = df["total_value"].dropna()
    initial_value = float(tv.iloc[0])
    final_value = float(tv.iloc[-1])
    total_return = (final_value - initial_value) / initial_value if initial_value != 0 else 0.0

    rets = _returns(df)

    days = int((df["date"].iloc[-1] - df["date"].iloc[0]).days) if len(df) >= 2 else 0
    annualized_return = (1.0 + total_return) ** (365.0 / days) - 1.0 if days > 0 else 0.0

    daily_vol = float(rets.std(ddof=0)) if not rets.empty else 0.0
    annualized_vol = daily_vol * float(np.sqrt(252.0)) if daily_vol > 0 else 0.0

    sharpe_ratio = (annualized_return / annualized_vol) if annualized_vol > 0 else 0.0

    max_dd = _max_drawdown(tv.values)
    calmar_ratio = (
        (annualized_return / abs(max_dd)) if max_dd and np.isfinite(max_dd) and max_dd != 0 else 0.0
    )

    positive_days = int((rets > 0).sum()) if not rets.empty else 0
    total_days = int(rets.shape[0]) if not rets.empty else 0
    win_rate = (positive_days / total_days) if total_days > 0 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else None,
        "calmar_ratio": float(calmar_ratio),
        "win_rate": float(win_rate),
        "total_days": int(days),
    }


def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    # Needs enough rows to be meaningful, but don't crash if short.
    if df is None or df.empty or df.shape[0] < 20:
        return {}

    rets = _returns(df)
    if rets.empty:
        return {}

    var_95 = float(rets.quantile(0.05))
    var_99 = float(rets.quantile(0.01))

    cvar_95 = float(rets[rets <= var_95].mean()) if (rets <= var_95).any() else None
    cvar_99 = float(rets[rets <= var_99].mean()) if (rets <= var_99).any() else None

    skewness = float(rets.skew())
    kurtosis = float(rets.kurtosis())

    q05 = float(rets.quantile(0.05))
    q95 = float(rets.quantile(0.95))
    tail_ratio = (abs(q05) / abs(q95)) if q95 != 0 else 0.0

    rolling_vol_20d = rets.rolling(20).std(ddof=0) * np.sqrt(252.0)
    current_volatility = (
        float(rolling_vol_20d.iloc[-1])
        if not rolling_vol_20d.empty and np.isfinite(rolling_vol_20d.iloc[-1])
        else None
    )

    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "tail_ratio": float(tail_ratio),
        "current_volatility": current_volatility,
    }


def calculate_regime_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    # Graceful: if no regime column or it's empty, return a clear status instead of KeyError.
    if "regime" not in df.columns:
        return {"status": "unavailable", "reason": "regime column missing"}

    series = df["regime"].dropna()
    if series.empty:
        return {"status": "unavailable", "reason": "regime column empty"}

    regime_counts = series.value_counts().to_dict()

    # transitions
    regime_changes = df["regime"] != df["regime"].shift(1)
    transition_count = int(regime_changes.sum())
    transition_frequency = float(transition_count / len(df)) if len(df) > 0 else 0.0

    return {
        "status": "ok",
        "regime_distribution": regime_counts,
        "transition_count": transition_count,
        "transition_frequency": transition_frequency,
    }


# ─────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────


def generate_risk_report() -> Path:
    print("📊 Generating comprehensive risk report...")

    portfolio_path = _read_portfolio_file()
    df_raw = pd.read_csv(portfolio_path)
    df = _coerce_schema(df_raw)

    perf = calculate_performance_metrics(df)
    risk = calculate_risk_metrics(df)
    regime = calculate_regime_metrics(df)

    # Dashboard-friendly "portfolio_metrics" (RiskDashboard expects this key)
    # expected_volatility: use annualized_volatility if present
    expected_vol = perf.get("annualized_volatility", None)
    risk_adj = perf.get("sharpe_ratio", None)

    # Portfolio summary: safe defaults if optional cols missing
    tv = df["total_value"].dropna()
    initial_value = float(tv.iloc[0]) if not tv.empty else None
    final_value = float(tv.iloc[-1]) if not tv.empty else None

    last_num_pos = (
        df["num_positions"].dropna().iloc[-1] if df["num_positions"].notna().any() else None
    )
    last_regime = df["regime"].dropna().iloc[-1] if df["regime"].notna().any() else None

    report: Dict[str, Any] = {
        "report_date": pd.Timestamp.now().isoformat(),
        "meta": {
            "portfolio_source": str(portfolio_path),
            "rows": int(len(df)),
            "min_date": str(df["date"].min().date()) if df["date"].notna().any() else None,
            "max_date": str(df["date"].max().date()) if df["date"].notna().any() else None,
        },
        "portfolio_summary": {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_positions": (
                int(last_num_pos)
                if last_num_pos is not None and np.isfinite(last_num_pos)
                else None
            ),
            "current_regime": str(last_regime) if last_regime is not None else None,
            "days_traded": int(len(df)),
        },
        # ✅ What RiskDashboard reads
        "portfolio_metrics": {
            "expected_volatility": float(expected_vol) if expected_vol is not None else None,
            "diversification_ratio": None,  # needs positions/covariance; placeholder
            "risk_adjusted_return": float(risk_adj) if risk_adj is not None else None,
            "max_drawdown": perf.get("max_drawdown", None),
            "return_total_pct": (
                (float(perf["total_return"]) * 100.0) if "total_return" in perf else None
            ),
            "return_annualized_pct": (
                (float(perf["annualized_return"]) * 100.0) if "annualized_return" in perf else None
            ),
        },
        # Keep your existing outputs too
        "performance_metrics": perf,
        "regime_metrics": regime,
        "risk_metrics": risk,
        # Placeholders (stabilizes UI expectations)
        "risk_decomposition": {},
        "position_analysis": {},
        "factor_weights": {},
        "risk_limits": {},
        "regime_adjustments": {},
        "performance_attribution": {},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RISK_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ Risk report saved to: {RISK_REPORT_FILE}")
    return RISK_REPORT_FILE


def main() -> None:
    try:
        generate_risk_report()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"❌ Failed generating risk report: {e}")
        raise


if __name__ == "__main__":
    main()
