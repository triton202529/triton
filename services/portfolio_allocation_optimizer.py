"""Joint portfolio allocation: caps, diversification floors, optional inverse-vol weighting."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _safe_float(x: Any, default: float = 1.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def volatilities_from_row_dicts(rows: List[Dict[str, Any]]) -> List[float]:
    """Annualized vol from reallocation rows (volatility_used from adaptive sizing)."""
    out: List[float] = []
    for r in rows:
        vu = r.get("volatility_used")
        if vu is None or vu == "":
            out.append(1.0)
        else:
            v = max(_safe_float(vu, 1.0), 1e-6)
            out.append(v)
    return out


def normalize_portfolio_notionals(
    scaled_notionals: List[float],
    volatilities: List[float],
    budget: float,
    cfg: Dict[str, Any],
) -> Tuple[List[float], List[float]]:
    """
    Jointly refine notionals that already sum to ``budget`` (or positive weights to renormalize).

    Returns (normalized_notional, portfolio_weight_pct) per row; weights sum to ~100.
    """
    if not scaled_notionals:
        return [], []
    n = len(scaled_notionals)
    if n != len(volatilities):
        volatilities = volatilities[:n] + [1.0] * max(0, n - len(volatilities))

    w = np.array(scaled_notionals, dtype=float)
    if budget <= 0 or w.sum() <= 1e-12:
        eq = round(budget / n, 2) if n else 0.0
        nn = [eq] * n
        pw = [100.0 / n] * n if n else []
        return nn, [round(p, 4) for p in pw]

    w = w / w.sum() * budget

    if not cfg.get("portfolio_optimizer_enabled", True):
        norm = w
        pct = norm / budget * 100.0 if budget > 0 else np.zeros_like(norm)
        return [round(float(x), 2) for x in norm], [round(float(x), 4) for x in pct]

    v = np.array(
        [
            max(float(volatilities[i]), float(cfg.get("risk_parity_vol_floor", 0.01)))
            for i in range(n)
        ],
        dtype=float,
    )

    if cfg.get("risk_parity_enabled", False) and float(cfg.get("risk_parity_strength", 0.0)) > 0:
        st = float(cfg.get("risk_parity_strength", 0.5))
        rp = 1.0 / (v**st)
        w = w * rp
        if w.sum() <= 1e-12:
            w = np.ones(n) / n * budget
        else:
            w = w / w.sum() * budget

    cap_pct = float(cfg.get("max_position_weight_pct", 0.35))
    cap = cap_pct * budget
    use_min = bool(cfg.get("min_diversification_enforce", True))
    min_pct = float(cfg.get("min_position_weight_pct", 0.0)) if use_min else 0.0
    floor = min_pct * budget

    if floor * n > budget + 1e-9:
        floor = budget / n

    if cap * n < budget - 1e-9:
        cap = budget / n

    for _ in range(500):
        w = np.clip(w, floor, cap)
        s = float(w.sum())
        if abs(s - budget) < 1e-4:
            break
        if s > budget + 1e-9:
            w *= budget / s
        else:
            room = np.maximum(cap - w, 0.0)
            rs = float(room.sum())
            if rs < 1e-12:
                break
            w += (budget - s) * (room / rs)

    s = float(w.sum())
    if abs(s - budget) > 1e-3 and s > 1e-12:
        w = w * (budget / s)
        w = np.clip(w, floor, cap)
        s2 = float(w.sum())
        if abs(s2 - budget) > 1e-3 and s2 > 1e-12:
            w = w * (budget / s2)

    norm = w
    pct = norm / budget * 100.0
    return [round(float(x), 2) for x in norm], [round(float(x), 4) for x in pct]
