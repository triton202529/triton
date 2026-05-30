# services/build_open_position_risk_diagnostics.py
"""
Open-position risk ranking from local Triton result files only (diagnostics; no strategy changes).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

PATH_TRADE_OUTCOMES = RESULTS / "trade_outcomes.csv"
PATH_PNL_BY_SYM = RESULTS / "pnl_diagnostics_by_symbol.csv"
PATH_POSITIONS = RESULTS / "positions_snapshot.csv"
PATH_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"

OUT_CSV = RESULTS / "open_position_risk_diagnostics.csv"
OUT_JSON = RESULTS / "open_position_risk_summary.json"

# --- Conservative action thresholds (USD and fraction P/L) ---
EXIT_USD = 40.0
EXIT_USD_SEVERE = 80.0
TRIM_USD = 8.0
PCT_STRONG_EXIT = 0.08
PCT_MOD_TRIM = 0.025
HOLDADD_PENALTY = 22.0


def _warn(msg: str) -> None:
    print(f"[build_open_position_risk_diagnostics] {msg}", file=sys.stderr)


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def safe_read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        _warn(f"missing or empty: {path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        _warn(f"could not read {label} ({path.name}): {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _f(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, str) and not str(x).strip()):
        return None
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _ulpc_to_fraction(ul: Optional[float], raw: Any) -> Optional[float]:
    """
    unrealized as fraction: Alpaca & snapshots often -0.05; outcomes may have return_pct in %.
    We store fraction in ( -1, 1 ) for scoring.
    """
    v = _f(raw)
    if v is None:
        if ul is not None and ul != 0:
            return None
        return None
    # Heuristic: if |v| < 0.2 treat as fraction; if |v| > 0.2 could be % already
    if abs(v) <= 0.2:
        return v
    if abs(v) <= 2.0:
        return v / 100.0
    return v / 100.0  # e.g. -2.77 from return_pct in outcomes


def _open_symbols_from_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    tc = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
    if not tc:
        return pd.DataFrame()
    o = df.copy()
    o["_sym"] = o[tc].map(_norm_sym)
    if "qty" in o.columns:
        o["_qty"] = pd.to_numeric(o["qty"], errors="coerce").fillna(0.0)
    elif "qty_available" in o.columns:
        o["_qty"] = pd.to_numeric(o["qty_available"], errors="coerce").fillna(0.0)
    else:
        o["_qty"] = 0.0
    o = o[o["_qty"] > 1e-6]
    return o


def _open_from_trade_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "outcome_type" not in df.columns or "symbol" not in df.columns:
        return pd.DataFrame()
    o = df[df["outcome_type"].astype(str).str.strip().str.upper() == "OPEN"].copy()
    if o.empty:
        return pd.DataFrame()
    o["_sym"] = o["symbol"].map(_norm_sym)
    return o


def _lifecycle_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    lc = df.copy()
    lc["ticker"] = lc["ticker"].map(_norm_sym)
    return lc.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")


def _row_from_position(r: pd.Series) -> Dict[str, Any]:
    sym = _norm_sym(r.get("ticker") or r.get("symbol"))
    q = _f(r.get("qty") or r.get("qty_available")) or 0.0
    avg = _f(r.get("avg_entry_price"))
    mkt = _f(r.get("current_price") or r.get("lastday_price"))
    ul = _f(r.get("unrealized_pl") or r.get("unrealized_pnl"))
    ulpc = _f(r.get("unrealized_plpc") or r.get("unrealized_intraday_plpc"))
    return {
        "symbol": sym,
        "open_qty": q,
        "avg_entry_price": avg,
        "market_price": mkt,
        "unrealized_pl": ul,
        "unrealized_pl_fraction": _ulpc_to_fraction(ul, ulpc),
    }


def _row_from_to_open(r: pd.Series) -> Dict[str, Any]:
    sym = _norm_sym(r.get("symbol"))
    q = _f(r.get("qty")) or 0.0
    return {
        "symbol": sym,
        "open_qty": q,
        "avg_entry_price": _f(r.get("entry_price")),
        "market_price": _f(r.get("exit_price")),
        "unrealized_pl": _f(r.get("unrealized_pl")),
        "unrealized_pl_fraction": _ulpc_to_fraction(
            _f(r.get("unrealized_pl")),
            r.get("return_pct"),
        ),
    }


def _hold_add_penalty(stance: str) -> bool:
    s = (stance or "").strip().upper()
    return s in ("HOLD", "ADD", "BUY", "FLAT", "WAIT")


def _compute_risk_score(
    ul: Optional[float],
    ulpc_frac: Optional[float],
    drag: bool,
    hold_add_pessimistic: bool,
) -> float:
    s = 0.0
    if ul is not None and ul < 0:
        s += min(200.0, -ul) * 0.35
    if ulpc_frac is not None and ulpc_frac < 0:
        s += min(80.0, -ulpc_frac * 200.0)
    if drag:
        s += 18.0
    if (
        hold_add_pessimistic
        and ul is not None
        and ul < 0
        and ulpc_frac is not None
        and ulpc_frac < -0.02
    ):
        s += HOLDADD_PENALTY
    return float(s)


def _severity_from_score(score: float) -> str:
    if score < 10:
        return "LOW"
    if score < 40:
        return "MED"
    return "HIGH"


def _candidate_action(
    ul: Optional[float],
    ulpc_frac: Optional[float],
    drag: bool,
) -> str:
    if ul is None and ulpc_frac is None:
        return "REVIEW"
    up = 0.0 if ul is None else float(ul)
    pneg = 0.0
    if ulpc_frac is not None and ulpc_frac < 0:
        pneg = float(ulpc_frac)

    if up >= 0.0 and pneg >= 0.0:
        return "HOLD_OK"

    if pneg <= -PCT_STRONG_EXIT or up <= -EXIT_USD_SEVERE or (drag and up <= -EXIT_USD):
        return "EXIT_CANDIDATE"
    if up <= -TRIM_USD or pneg <= -PCT_MOD_TRIM:
        return "TRIM_CANDIDATE"
    if up < 0.0 or pneg < 0.0:
        return "REVIEW"
    return "HOLD_OK"


def build() -> int:
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    RESULTS.mkdir(parents=True, exist_ok=True)

    df_to = safe_read_csv(PATH_TRADE_OUTCOMES, "trade_outcomes")
    df_pnl = safe_read_csv(PATH_PNL_BY_SYM, "pnl_diagnostics_by_symbol")
    df_pos = safe_read_csv(PATH_POSITIONS, "positions")
    df_lc = safe_read_csv(PATH_LIFECYCLE, "lifecycle")

    pnl_drag: Dict[str, bool] = {}
    if not df_pnl.empty and "symbol" in df_pnl.columns and "drag_flag" in df_pnl.columns:
        for _, r in df_pnl.iterrows():
            s = _norm_sym(r.get("symbol"))
            pnl_drag[s] = bool(r.get("drag_flag"))

    lc_ix = _lifecycle_index(df_lc)
    to_open = _open_from_trade_outcomes(df_to)
    pos_open = _open_symbols_from_positions(df_pos)

    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for _, r in pos_open.iterrows():
        d = _row_from_position(r)
        sym = d["symbol"]
        if not sym:
            continue
        seen.add(sym)
        drag = pnl_drag.get(sym, False)
        stance = ""
        eff_st = ""
        if not lc_ix.empty and sym in lc_ix.index:
            row = lc_ix.loc[sym]
            stance = str(
                row.get("effective_stance")
                or row.get("stance")
                or row.get("lifecycle_action")
                or ""
            )
            eff_st = str(row.get("effective_position_state") or row.get("position_state") or "")
        ulpc_f = d.get("unrealized_pl_fraction")
        ul = d.get("unrealized_pl")
        hnp = _hold_add_penalty(stance) and ul is not None and ul < 0
        sc = _compute_risk_score(ul, ulpc_f, drag, hnp)
        sev = _severity_from_score(sc)
        act = _candidate_action(ul, ulpc_f, drag)
        rows.append(
            {
                "symbol": sym,
                "open_qty": d.get("open_qty"),
                "avg_entry_price": d.get("avg_entry_price"),
                "market_price": d.get("market_price"),
                "unrealized_pl": ul,
                "unrealized_pl_pct": (ulpc_f * 100.0) if ulpc_f is not None else None,
                "lifecycle_stance": stance,
                "effective_position_state": eff_st,
                "drag_flag": drag,
                "risk_score": round(sc, 3),
                "severity_bucket": sev,
                "candidate_action": act,
            }
        )

    if not to_open.empty:
        for _, r in to_open.iterrows():
            sym = _norm_sym(r.get("symbol"))
            if not sym or sym in seen:
                continue
            d = _row_from_to_open(r)
            seen.add(sym)
            drag = pnl_drag.get(sym, False)
            stance, eff_st = "", ""
            if not lc_ix.empty and sym in lc_ix.index:
                row = lc_ix.loc[sym]
                stance = str(
                    row.get("effective_stance")
                    or row.get("stance")
                    or row.get("lifecycle_action")
                    or ""
                )
                eff_st = str(row.get("effective_position_state") or row.get("position_state") or "")
            ulpc_f = d.get("unrealized_pl_fraction")
            ul = d.get("unrealized_pl")
            hnp = _hold_add_penalty(stance) and ul is not None and ul < 0
            sc = _compute_risk_score(ul, ulpc_f, drag, hnp)
            sev = _severity_from_score(sc)
            act = _candidate_action(ul, ulpc_f, drag)
            rows.append(
                {
                    "symbol": sym,
                    "open_qty": d.get("open_qty"),
                    "avg_entry_price": d.get("avg_entry_price"),
                    "market_price": d.get("market_price"),
                    "unrealized_pl": ul,
                    "unrealized_pl_pct": (ulpc_f * 100.0) if ulpc_f is not None else None,
                    "lifecycle_stance": stance,
                    "effective_position_state": eff_st,
                    "drag_flag": drag,
                    "risk_score": round(sc, 3),
                    "severity_bucket": sev,
                    "candidate_action": act,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("risk_score", ascending=False)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    n = len(out)
    drag_n = int(out["drag_flag"].sum()) if n and "drag_flag" in out.columns else 0
    w_sym = None
    w_pl: Optional[float] = None
    if n and "unrealized_pl" in out.columns:
        uu = out["unrealized_pl"].apply(lambda x: _f(x))
        if uu.notna().any():
            imin = uu.idxmin()
            w_sym = str(out.loc[imin, "symbol"])
            w_pl = _f(out.loc[imin, "unrealized_pl"])
    n_exit = int((out["candidate_action"] == "EXIT_CANDIDATE").sum()) if n else 0
    n_trim = int((out["candidate_action"] == "TRIM_CANDIDATE").sum()) if n else 0
    n_hold = int((out["candidate_action"] == "HOLD_OK").sum()) if n else 0
    drag_list = (
        out.loc[out["drag_flag"] == True, "symbol"].astype(str).tolist()  # noqa: E712
        if n and "drag_flag" in out.columns
        else []
    )

    def _j(v: Any) -> Any:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    summ: Dict[str, Any] = {
        "open_symbols": n,
        "drag_symbols": int(drag_n),
        "drag_symbol_list": drag_list,
        "worst_open_symbol": w_sym,
        "worst_open_pl": _j(w_pl) if w_pl is not None else None,
        "exit_candidates": n_exit,
        "trim_candidates": n_trim,
        "hold_ok_count": n_hold,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, allow_nan=False)

    print(
        "[OPEN_POSITION_RISK] "
        f"open_symbols={n} "
        f"drag_symbols={drag_n} "
        f"worst_open_symbol={w_sym or ''} "
        f"exit_candidates={n_exit} "
        f"trim_candidates={n_trim} "
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build open position risk diagnostics from result CSVs."
    )
    p.parse_args(argv)
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
