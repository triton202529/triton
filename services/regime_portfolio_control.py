"""Market regime (risk-on / neutral / risk-off) for reallocation exposure scaling — not execution guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


def _read_processed_parquet(symbol: str, processed_dir: Path) -> Optional[pd.DataFrame]:
    sym = symbol.strip().upper().replace("^", "")
    for v in (sym, sym.replace(".", "-")):
        p = processed_dir / f"{v}.parquet"
        if p.is_file():
            try:
                return pd.read_parquet(p)
            except Exception:
                return None
    return None


def _spy_drawdown_pct(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "close" not in df.columns:
        return None
    c = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(c) < 20:
        return None
    peak = float(c.cummax().iloc[-1])
    last = float(c.iloc[-1])
    if peak <= 0:
        return None
    return (last / peak) - 1.0


def _last_vix_level(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty or "close" not in df.columns:
        return None
    c = pd.to_numeric(df["close"], errors="coerce").dropna()
    if c.empty:
        return None
    return float(c.iloc[-1])


def _atr_pct_last(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    if df is None or len(df) < n + 2:
        return None
    need = ("high", "low", "close")
    if not all(x in df.columns for x in need):
        return None
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(n).mean().iloc[-1])
    last = float(c.iloc[-1])
    if last <= 0:
        return None
    return atr / last


def _classify_regime(
    vix: Optional[float],
    dd_pct: Optional[float],
    atr_pct: Optional[float],
    cfg: Dict[str, Any],
) -> str:
    v_off = float(cfg.get("regime_vix_risk_off", 25.0))
    v_on = float(cfg.get("regime_vix_risk_on", 16.0))
    dd_off = float(cfg.get("regime_spy_dd_risk_off", -0.08))
    dd_on = float(cfg.get("regime_spy_dd_risk_on", -0.02))
    atr_off = float(cfg.get("regime_atr_risk_off_pct", 0.022))
    atr_on = float(cfg.get("regime_atr_risk_on_pct", 0.012))

    if vix is not None and vix >= v_off:
        return "RISK_OFF"
    if dd_pct is not None and dd_pct <= dd_off:
        return "RISK_OFF"

    if vix is not None and vix <= v_on:
        if dd_pct is None or dd_pct >= dd_on:
            return "RISK_ON"

    if vix is None and atr_pct is not None:
        if atr_pct >= atr_off:
            return "RISK_OFF"
        if atr_pct <= atr_on and (dd_pct is None or dd_pct >= dd_on):
            return "RISK_ON"

    return "NEUTRAL"


def detect_market_regime(
    cfg: Dict[str, Any], processed_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Classify regime using VIX (if available) + SPY drawdown from peak, with ATR% fallback when VIX missing.
    """
    pdir = processed_dir or PROCESSED_DIR
    out: Dict[str, Any] = {
        "regime_label": "NEUTRAL",
        "regime_exposure_multiplier": 1.0,
        "vix_last": None,
        "spy_drawdown_pct": None,
        "spy_atr_pct": None,
        "source": "none",
    }

    if not cfg.get("regime_portfolio_control_enabled", True):
        out["source"] = "disabled"
        return out

    spy = _read_processed_parquet("SPY", pdir)
    vix_df = _read_processed_parquet("VIX", pdir)
    if vix_df is None:
        vix_df = _read_processed_parquet("^VIX", pdir)

    dd = _spy_drawdown_pct(spy) if spy is not None else None
    vix = _last_vix_level(vix_df)
    atr = _atr_pct_last(spy) if spy is not None else None

    out["spy_drawdown_pct"] = round(dd, 6) if dd is not None else None
    out["vix_last"] = round(vix, 4) if vix is not None else None
    out["spy_atr_pct"] = round(atr, 6) if atr is not None else None

    if spy is None and vix_df is None:
        out["source"] = "no_data"
        return out

    label = _classify_regime(vix, dd, atr, cfg)
    out["regime_label"] = label
    out["source"] = "vix+spy" if vix is not None else "spy+atr"

    m_off = float(cfg.get("regime_risk_off_exposure", 0.6))
    m_neu = float(cfg.get("regime_neutral_exposure", 0.85))
    m_on = float(cfg.get("regime_risk_on_exposure", 1.0))

    if label == "RISK_OFF":
        out["regime_exposure_multiplier"] = min(1.0, max(0.05, m_off))
    elif label == "RISK_ON":
        out["regime_exposure_multiplier"] = min(1.0, max(0.05, m_on))
    else:
        out["regime_exposure_multiplier"] = min(1.0, max(0.05, m_neu))

    return out


def apply_regime_max_weight_scale(cfg: Dict[str, Any], regime_label: str) -> Dict[str, Any]:
    """Optional: tighten per-name cap in RISK_OFF (does not change execution_guard)."""
    out = dict(cfg)
    if not cfg.get("regime_reduce_max_weight_in_risk_off", True):
        return out
    if regime_label != "RISK_OFF":
        return out
    base = float(cfg.get("max_position_weight_pct", 0.35))
    scale = float(cfg.get("regime_risk_off_max_weight_scale", 0.85))
    out["max_position_weight_pct"] = max(0.05, min(1.0, base * scale))
    return out
