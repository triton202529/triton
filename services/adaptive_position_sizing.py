# services/adaptive_position_sizing.py
"""Confidence + volatility-adjusted size factors (planning only; guards unchanged)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXEC_GUARD_CONFIG = ROOT / "config" / "execution_guard.json"

VOL_COLUMNS: Tuple[str, ...] = ("atr_pct", "realized_vol", "volatility_proxy", "iv_rank")


def _guard_max_notional() -> Optional[float]:
    try:
        if EXEC_GUARD_CONFIG.is_file():
            u = json.loads(EXEC_GUARD_CONFIG.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict) and u.get("max_notional_usd") is not None:
                return float(u["max_notional_usd"])
    except Exception:
        pass
    return None


def max_order_notional_usd(cfg: Optional[Dict[str, Any]] = None) -> float:
    """Per-trade notional cap: config override, else execution_guard, else large default."""
    if cfg:
        v = cfg.get("max_order_notional_usd")
        if v is not None:
            try:
                return max(1.0, float(v))
            except Exception:
                pass
    g = _guard_max_notional()
    if g is not None and g > 0:
        return float(g)
    return 1500.0


def pick_volatility_from_row(row: Any) -> Tuple[Optional[float], str]:
    """First available volatility column (ordered); value + column name."""
    try:
        if hasattr(row, "index"):
            for c in VOL_COLUMNS:
                if c in row.index:
                    v = row.get(c)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        try:
                            x = float(v)
                            if x == x:
                                return x, c
                        except Exception:
                            pass
        elif isinstance(row, dict):
            for c in VOL_COLUMNS:
                if c in row and row[c] is not None:
                    try:
                        x = float(row[c])
                        if x == x:
                            return x, c
                    except Exception:
                        pass
    except Exception:
        pass
    return None, ""


def volatility_proxy_from_row(row: Any) -> Optional[float]:
    """Backward compat: value only (first column)."""
    v, _ = pick_volatility_from_row(row)
    return v


def normalize_volatility_for_adjustment(raw: float, col: str) -> float:
    """
    Map raw column value to a non-negative scale v for vol_adjustment.
    iv_rank: 0-100 -> 0-1
    atr_pct / realized_vol: stored as percent (e.g. 3.5 = 3.5%) -> /100
    volatility_proxy: unitless, capped
    """
    x = abs(float(raw))
    if col == "iv_rank":
        return min(1.0, max(0.0, x / 100.0))
    if col in ("atr_pct", "realized_vol"):
        # Values are percent of price (e.g. 3.5 = 3.5%); higher -> smaller vol_adjustment
        return min(2.0, x / 100.0)
    return min(5.0, max(0.0, x))


def compute_vol_adjustment(vol_norm: float, cfg: Optional[Dict[str, Any]] = None) -> float:
    """
    Stronger risk response: 1 / (1 + k * v) or 1 / (1 + k * (s*v)^2) when use_quadratic_vol.
    quadratic_vol_scale (default 10) maps typical normalized v (~0.05 for 5% ATR) into a range
    where v^2 materially reduces size. Floor vol_adjustment at vol_adjustment_floor.
    """
    cfg = dict(cfg or {})
    v = max(0.0, float(vol_norm))
    k = float(cfg.get("volatility_impact_strength", 3.0))
    if k < 0:
        k = 0.0
    if bool(cfg.get("use_quadratic_vol", True)):
        s = max(0.0, float(cfg.get("quadratic_vol_scale", 10.0)))
        vv = v * s
        term = k * vv * vv
    else:
        term = k * v
    adj = 1.0 / (1.0 + term)
    floor_v = float(cfg.get("vol_adjustment_floor", 0.3))
    return max(floor_v, min(1.0, adj))


def _confidence_size_factor(confidence: float, delta_pct: float, cfg: Dict[str, Any]) -> float:
    lo = float(cfg.get("size_factor_min", 0.5))
    hi = float(cfg.get("size_factor_max", 1.5))
    c = max(0.0, min(1.0, float(confidence)))
    sf = lo + c * (hi - lo)
    d_scale = float(cfg.get("delta_pct_boost_scale", 50.0))
    if d_scale > 0:
        bump = max(-0.1, min(0.1, float(delta_pct) * d_scale * 0.01))
        sf *= 1.0 + bump
    return max(lo * 0.8, min(hi * 1.1, sf))


def compute_size_factor_breakdown(
    confidence: float,
    delta_pct: float,
    row: Any,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    size_factor_confidence from confidence (+ delta tweak);
    vol_adjustment = 1/(1+k*v) or 1/(1+k*v^2), floored;
    size_factor_final = clamp(size_factor_confidence * vol_adjustment, final_min..final_max).
    """
    cfg = dict(cfg or {})
    out: Dict[str, Any] = {
        "size_factor_confidence": 1.0,
        "volatility_used": None,
        "volatility_column": "",
        "vol_adjustment": 1.0,
        "size_factor_final": 1.0,
    }
    if not bool(cfg.get("adaptive_sizing_enabled", True)):
        return out

    sf_conf = _confidence_size_factor(confidence, delta_pct, cfg)
    out["size_factor_confidence"] = sf_conf

    raw, col = pick_volatility_from_row(row)
    vol_adj = 1.0
    if raw is not None and col:
        vn = normalize_volatility_for_adjustment(raw, col)
        vol_adj = compute_vol_adjustment(vn, cfg)
        out["volatility_used"] = raw
        out["volatility_column"] = col
    out["vol_adjustment"] = vol_adj

    sf_final = sf_conf * vol_adj
    lo = float(cfg.get("size_factor_final_min", 0.3))
    hi = float(cfg.get("size_factor_final_max", 1.5))
    sf_final = max(lo, min(hi, sf_final))
    out["size_factor_final"] = sf_final
    return out


def compute_size_factor(
    confidence: float,
    delta_pct: float = 0.0,
    *,
    volatility_proxy: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
    row: Optional[Any] = None,
) -> float:
    """
    Returns size_factor_final for backward compatibility.
    Prefer compute_size_factor_breakdown(..., row=row) when row is available.
    """
    if row is not None:
        return float(
            compute_size_factor_breakdown(confidence, delta_pct, row, cfg)["size_factor_final"]
        )
    # Legacy: no row — optional deprecated volatility_proxy scaling (match old behavior loosely)
    cfg = dict(cfg or {})
    if not bool(cfg.get("adaptive_sizing_enabled", True)):
        return 1.0
    sf = _confidence_size_factor(confidence, delta_pct, cfg)
    if volatility_proxy is not None and float(volatility_proxy) > 0:
        vn = normalize_volatility_for_adjustment(float(volatility_proxy), "volatility_proxy")
        sf *= compute_vol_adjustment(vn, cfg)
    lo = float(cfg.get("size_factor_final_min", 0.3))
    hi = float(cfg.get("size_factor_final_max", 1.5))
    return max(lo, min(hi, sf))
