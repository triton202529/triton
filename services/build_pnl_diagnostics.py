# services/build_pnl_diagnostics.py
"""
PnL diagnostics from trade_outcomes* artifacts (read-only; no trading logic).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

PATH_TRADE_OUTCOMES = RESULTS / "trade_outcomes.csv"
PATH_BY_SYMBOL = RESULTS / "trade_outcomes_by_symbol.csv"
PATH_SUMMARY = RESULTS / "trade_outcomes_summary.json"

OUT_JSON = RESULTS / "pnl_diagnostics.json"
OUT_BY_SYMBOL = RESULTS / "pnl_diagnostics_by_symbol.csv"


def _warn(msg: str) -> None:
    print(f"[build_pnl_diagnostics] {msg}", file=sys.stderr)


def _to_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, str) and not str(x).strip()):
        return None
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def safe_read_csv(path: Path, *, label: str) -> pd.DataFrame:
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


def safe_read_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        _warn(f"missing or empty: {path.name}")
        return {}
    try:
        o = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return o if isinstance(o, dict) else {}
    except Exception as e:
        _warn(f"could not read JSON ({path.name}): {e}")
        return {}


def _count_outcome_types(df: pd.DataFrame) -> Tuple[int, int]:
    if df is None or df.empty or "outcome_type" not in df.columns:
        return 0, 0
    ot = df["outcome_type"].astype(str).str.strip().str.upper()
    n_open = int((ot == "OPEN").sum())
    n_re = int((ot == "REALIZED").sum() + (ot == "UNKNOWN").sum())
    return n_open, n_re


def _loss_magnitude(x: Optional[float]) -> float:
    if x is None:
        return 0.0
    v = float(x)
    return -v if v < 0.0 else 0.0


def _loss_share(
    tr: Optional[float], tu: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Share of portfolio loss *components* (splitting negative realized vs negative unrealized)."""
    if tr is None and tu is None:
        return None, None
    a = 0.0 if tr is None else float(tr)
    b = 0.0 if tu is None else float(tu)
    lr = -min(0.0, a)
    lu = -min(0.0, b)
    s = lr + lu
    if s <= 1e-12:
        return None, None
    return lr / s, lu / s


def _severity_bucket(val: float) -> str:
    a = abs(val)
    if a < 1e-6:
        return "NONE"
    if a < 5.0:
        return "LOW"
    if a < 25.0:
        return "MED"
    return "HIGH"


def _loss_source_row(r: float, u: float) -> str:
    """r=realized_pl, u=unrealized_pl (numeric, default 0)."""
    t = r + u
    if t >= -1e-9:
        return "NONE"
    re_loss = r < 0
    un_loss = u < 0
    if re_loss and not un_loss:
        return "REALIZED"
    if un_loss and not re_loss:
        return "UNREALIZED"
    if re_loss and un_loss:
        # both negative -> MIXED if comparable
        if abs(r) > 1e-6 and abs(u) > 1e-6:
            rmg = -r
            umg = -u
            if min(rmg, umg) / max(rmg, umg) > 0.25:
                return "MIXED"
        return "REALIZED" if abs(r) >= abs(u) else "UNREALIZED"
    return "NONE"


def _drag_per_symbol(r: float, u: float, open_qty: float) -> bool:
    if open_qty <= 0:
        return False
    u_loss = u < 0
    r_loss = r < 0
    if not u_loss:
        return False
    if u_loss and not r_loss:
        return True
    return -u > -r + 1e-9


def _aggregate_from_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "symbol" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["symbol"] = d["symbol"].map(_norm_sym)
    d["_r"] = d["realized_pl"].map(lambda x: _to_float(x) or 0.0)
    d["_u"] = d["unrealized_pl"].map(lambda x: _to_float(x) or 0.0)
    g = d.groupby("symbol", dropna=False).agg(
        realized_pl=("_r", "sum"),
        unrealized_pl=("_u", "sum"),
        trade_rows=("symbol", "count"),
    )
    g = g.reset_index()
    oq = d[d["outcome_type"].astype(str).str.strip().str.upper() == "OPEN"]
    if not oq.empty and "qty" in oq.columns:
        qg = oq.groupby("symbol")["qty"].apply(
            lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())
        )
        g = g.merge(qg.to_frame("open_qty"), on="symbol", how="left")
    else:
        g["open_qty"] = 0.0
    g["open_qty"] = g["open_qty"].fillna(0.0)
    g["total_pl"] = g["realized_pl"] + g["unrealized_pl"]
    return g


def build() -> int:
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    RESULTS.mkdir(parents=True, exist_ok=True)

    summary = safe_read_summary(PATH_SUMMARY)
    df_sym = safe_read_csv(PATH_BY_SYMBOL, label="by_symbol")
    df_to = safe_read_csv(PATH_TRADE_OUTCOMES, label="trade_outcomes")

    n_open_r, n_real_r = _count_outcome_types(df_to)
    if df_sym.empty and not df_to.empty:
        _warn("trade_outcomes_by_symbol.csv empty; aggregating from trade_outcomes.csv")
        df_sym = _aggregate_from_outcomes(df_to)

    tr = _to_float(summary.get("total_realized_pl")) if summary else None
    tu = _to_float(summary.get("total_unrealized_pl")) if summary else None
    tc = _to_float(summary.get("total_combined_pl")) if summary else None
    if df_sym is not None and not df_sym.empty:
        for c in ("realized_pl", "unrealized_pl", "total_pl"):
            if c in df_sym.columns:
                df_sym[c] = pd.to_numeric(df_sym[c], errors="coerce")
        s_tr = (
            float(df_sym["realized_pl"].fillna(0).sum())
            if "realized_pl" in df_sym.columns
            else None
        )
        s_tu = (
            float(df_sym["unrealized_pl"].fillna(0).sum())
            if "unrealized_pl" in df_sym.columns
            else None
        )
        s_tc = s_tr + s_tu if s_tr is not None and s_tu is not None else None
        if tr is None and s_tr is not None:
            tr = s_tr
        if tu is None and s_tu is not None:
            tu = s_tu
        if tc is None and s_tc is not None:
            tc = s_tc

    trf = 0.0 if tr is None else float(tr)
    tuf = 0.0 if tu is None else float(tu)
    tcf = 0.0 if tc is None else float(tc)
    if (
        tr is None
        and tu is None
        and tc is None
        and not df_sym.empty
        and "total_pl" in df_sym.columns
    ):
        tcf = float(df_sym["total_pl"].fillna(0).sum())

    rs, us = _loss_share(tr, tu)
    l_u = _loss_magnitude(tu)
    l_r = _loss_magnitude(tr)
    open_drag = bool(l_u > l_r and l_u > 1e-9)
    rel_loss = tr is not None and trf < 0.0

    worst = _norm_sym(summary.get("worst_symbol")) or None
    best = _norm_sym(summary.get("best_symbol")) or None
    if (
        (not worst or not best)
        and not df_sym.empty
        and "total_pl" in df_sym.columns
        and "symbol" in df_sym.columns
    ):
        s = df_sym.dropna(subset=["total_pl"])
        if not s.empty:
            imin = s["total_pl"].idxmin()
            imax = s["total_pl"].idxmax()
            worst = worst or _norm_sym(s.loc[imin, "symbol"])
            best = best or _norm_sym(s.loc[imax, "symbol"])

    w_open, b_open = None, None
    if not df_sym.empty and "unrealized_pl" in df_sym.columns and "open_qty" in df_sym.columns:
        op = df_sym[pd.to_numeric(df_sym["open_qty"], errors="coerce").fillna(0) > 0].copy()
        if not op.empty and "symbol" in op.columns:
            u = op["unrealized_pl"]
            w_open = _norm_sym(op.loc[u.idxmin(), "symbol"]) if len(op) else None
            b_open = _norm_sym(op.loc[u.idxmax(), "symbol"]) if len(op) else None

    out_rows: List[Dict[str, Any]] = []
    if not df_sym.empty:
        for _, row in df_sym.iterrows():
            sym = _norm_sym(row.get("symbol"))
            if not sym:
                continue
            r = _to_float(row.get("realized_pl")) or 0.0
            u = _to_float(row.get("unrealized_pl")) or 0.0
            t = _to_float(row.get("total_pl"))
            if t is None:
                t = r + u
            oq = _to_float(row.get("open_qty")) or 0.0
            trc = int(row.get("trade_rows") or 0)
            lsrc = _loss_source_row(r, u)
            sev = _severity_bucket(float(t) if t is not None else 0.0)
            dflag = _drag_per_symbol(r, u, oq)
            out_rows.append(
                {
                    "symbol": sym,
                    "realized_pl": r,
                    "unrealized_pl": u,
                    "total_pl": float(t) if t is not None else 0.0,
                    "loss_source": lsrc,
                    "severity_bucket": sev,
                    "open_qty": oq,
                    "trade_rows": trc,
                    "drag_flag": dflag,
                }
            )
    by_df = (
        pd.DataFrame(out_rows)
        if out_rows
        else pd.DataFrame(
            columns=[
                "symbol",
                "realized_pl",
                "unrealized_pl",
                "total_pl",
                "loss_source",
                "severity_bucket",
                "open_qty",
                "trade_rows",
                "drag_flag",
            ]
        )
    )
    if not by_df.empty:
        by_df.to_csv(OUT_BY_SYMBOL, index=False, encoding="utf-8")
    else:
        by_df.to_csv(OUT_BY_SYMBOL, index=False, encoding="utf-8")

    diag: Dict[str, Any] = {
        "total_realized_pl": tr,
        "total_unrealized_pl": tu,
        "total_combined_pl": (
            tc if tc is not None else (trf + tuf if (tr is not None or tu is not None) else None)
        ),
        "realized_loss_share": rs,
        "unrealized_loss_share": us,
        "open_position_drag_flag": open_drag,
        "realized_loss_flag": rel_loss,
        "worst_symbol": worst,
        "best_symbol": best,
        "worst_open_symbol": w_open,
        "best_open_symbol": b_open,
        "total_open_rows": n_open_r,
        "total_realized_rows": n_real_r,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    tcp = tc if tc is not None else (tcf if (tr is not None or tu is not None) else 0.0)
    if tcp is None:
        tcp = 0.0
    print(
        "[PNL_DIAGNOSTICS] "
        f"worst_symbol={worst or ''} "
        f"best_symbol={best or ''} "
        f"open_position_drag_flag={open_drag} "
        f"realized_loss_flag={rel_loss} "
        f"total_combined_pl={tcp} "
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build PnL diagnostics from trade_outcomes* files.")
    p.parse_args(argv)
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
