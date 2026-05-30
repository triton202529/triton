# services/capital_engine.py
"""Capital deployment engine helpers — deploy ratio, borderline rescue selection (execute_trades integration)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd


def deploy_ratio_metrics(
    planned: List[Any],
    acct: Any,
    cfg: Dict[str, Any],
) -> Tuple[float, float, float]:
    """Returns (planned_notional_sum, available_bp, deploy_ratio)."""
    pn = sum(float(getattr(p, "planned_notional", 0.0) or 0.0) for p in planned)
    reserve = float(cfg.get("reserve_cash_pct", 0.2))
    bp = float(getattr(acct, "buying_power", 0.0) or 0.0)
    eq = float(getattr(acct, "equity", 0.0) or 0.0)
    avail = bp * (1.0 - reserve) if bp > 0 else eq * (1.0 - reserve)
    dr = (pn / avail) if avail > 1e-9 else 0.0
    return pn, avail, dr


def find_row_for_symbol(df: pd.DataFrame, sym: str) -> Optional[pd.Series]:
    if df is None or df.empty or "ticker" not in df.columns:
        return None
    su = str(sym or "").strip().upper()
    for _idx, row in df.iterrows():
        t = str(row.get("ticker") or "").strip().upper()
        if t == su:
            return row
    return None


def pick_borderline_rescues(
    plan_lines: List[Any],
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    row_final_score_fn: Callable[[pd.Series], float],
) -> Tuple[Set[str], Set[str], Set[str], int, List[str], int]:
    """Select up to max_borderline_trades rescues from skipped BELOW_MIN_SCORE /
    ADD_BELOW_SCORE_THRESHOLD / SCORE_SIZED_TOO_SMALL.

    Returns (rescue_global, rescue_add, rescue_sizing, borderline_candidates_count,
    rescued_symbols_ordered, rescued_count).
    """
    strict_min = float(cfg.get("min_final_score_threshold", 0.65) or 0.0)
    relaxed_min = float(cfg.get("capital_engine_relaxed_min_final_score_threshold", 0.55) or 0.0)
    add_thr = float(cfg.get("add_min_final_score_threshold", 0.67) or 0.0)
    relaxed_add = float(cfg.get("capital_engine_relaxed_add_score_threshold", 0.60) or 0.0)
    max_n = max(0, int(cfg.get("capital_engine_max_borderline_trades", 1) or 0))

    pool_valid: List[Tuple[float, str, str, int]] = []
    for pl in plan_lines:
        if getattr(pl, "status", None) != "skipped":
            continue
        sr = str(getattr(pl, "skip_reason", "") or "").strip()
        if sr not in ("BELOW_MIN_SCORE", "ADD_BELOW_SCORE_THRESHOLD", "SCORE_SIZED_TOO_SMALL"):
            continue
        sym = str(getattr(pl, "symbol", "") or "").strip().upper()
        if not sym:
            continue
        row = find_row_for_symbol(df, sym)
        if row is None:
            continue
        fs = float(row_final_score_fn(row))
        if fs < 0:
            continue
        base_qty_val = 0
        try:
            base_qty_val = int(row.get("shares") or row.get("qty") or 0)
        except Exception:
            base_qty_val = 0
        if sr == "BELOW_MIN_SCORE":
            if relaxed_min <= fs < strict_min:
                pool_valid.append((fs, sym, sr, base_qty_val))
        elif sr == "ADD_BELOW_SCORE_THRESHOLD":
            st = str(getattr(pl, "stance", "") or "").strip().upper()
            if st == "ADD" and relaxed_add <= fs < add_thr:
                pool_valid.append((fs, sym, sr, base_qty_val))
        elif sr == "SCORE_SIZED_TOO_SMALL":
            if fs >= relaxed_min:
                pool_valid.append((fs, sym, sr, base_qty_val))

    pool_valid.sort(key=lambda x: x[0], reverse=True)
    borderline_n = len(pool_valid)
    for fs, sym, src, bq in pool_valid:
        print(
            f"[CAPITAL_ENGINE_CANDIDATE] symbol={sym} final_score={fs:.4f} "
            f"base_qty={bq} reason=borderline_candidate source={src}"
        )
    pool = pool_valid[:max_n] if max_n else []

    rg: Set[str] = set()
    ra: Set[str] = set()
    rs: Set[str] = set()
    rescued_syms: List[str] = []
    for fs, sym, src, _bq in pool:
        rescued_syms.append(sym)
        if src == "BELOW_MIN_SCORE":
            rg.add(sym)
        elif src == "ADD_BELOW_SCORE_THRESHOLD":
            ra.add(sym)
        else:
            rs.add(sym)

    return rg, ra, rs, borderline_n, rescued_syms, len(rescued_syms)
