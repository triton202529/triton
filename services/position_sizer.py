# services/position_sizer.py
"""
Standalone capital allocation: trade_opportunities.csv -> orders_sized.csv.
Enriches prices from the opportunities row and local result CSVs (no broker calls).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_IN = RESULTS_DIR / "trade_opportunities.csv"
DEFAULT_OUT = RESULTS_DIR / "orders_sized.csv"
DEFAULT_PLAN = RESULTS_DIR / "execution_plan.csv"
DEFAULT_EI = RESULTS_DIR / "execution_intelligence.csv"
DEFAULT_CAPITAL = 10_000.0
DEFAULT_NORM_MIN_CONF: Optional[float] = None
DEFAULT_NORM_MAX_CONF: Optional[float] = None
DEFAULT_MAX_NOTIONAL: Optional[float] = None
DEFAULT_MAX_POSITION_WEIGHT: Optional[float] = None
DEFAULT_MIN_ORDER_SIZE: float = 0.0

REQUIRED_COLS = ("confidence", "delta_pct")
# Order for picking the first valid numeric price
PRICE_COLUMN_ORDER: Tuple[str, ...] = (
    "close",
    "price",
    "entry_price",
    "limit_price",
    "last_price",
    "mark",
    "current_price",
    "px",
    # also used when scanning external CSVs
    "intended_price",
    "submitted_limit_price",
    "decision_mid_price",
    "quote_mid",
    "fill_price",
)


def display_ticker(x: Any) -> str:
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none", ""):
        return ""
    return s.upper()


def join_key(x: Any) -> str:
    """
    Key for safe joins: strip, upper, map BRK.B <-> BRK-B via '.' -> '-'.
    """
    d = display_ticker(x)
    if not d:
        return ""
    return d.replace(".", "-")


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _first_valid_price_in_row(
    row: pd.Series, cols: Tuple[str, ...]
) -> Tuple[Optional[float], Optional[str]]:
    for c in cols:
        if c not in row.index:
            continue
        v = pd.to_numeric(row.get(c), errors="coerce")
        if pd.isna(v) or float(v) <= 0.0:
            continue
        return float(v), c
    return None, None


def _read_aux_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return None


def _price_map_from_frame(
    df: pd.DataFrame, label: str, cols: Tuple[str, ...] = PRICE_COLUMN_ORDER
) -> Dict[str, Tuple[float, str]]:
    """
    One price per join_key, first valid price column in `cols` order.
    Iterate from the bottom of the file so later rows (often fresher) win.
    """
    out: Dict[str, Tuple[float, str]] = {}
    sym_col = None
    for c in ("ticker", "symbol", "TICKER", "Symbol"):
        if c in df.columns:
            sym_col = c
            break
    if sym_col is None:
        return out
    for _, row in df.iloc[::-1].iterrows():
        jk = join_key(row.get(sym_col))
        if not jk:
            continue
        if jk in out:
            continue
        px, colname = _first_valid_price_in_row(row, cols)
        if px is not None and colname is not None:
            out[jk] = (px, f"{label}.{colname}")
    return out


def enrich_prices(
    work: pd.DataFrame,
    *,
    execution_plan_path: Path = DEFAULT_PLAN,
    execution_intelligence_path: Path = DEFAULT_EI,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Set price_used and price_source per row from (1) direct opportunity columns,
    (2) execution_plan.csv, (3) execution_intelligence.csv. No fabricated prices.
    """
    w = work.copy()
    w.columns = [str(c).strip() for c in w.columns]
    n_plan = 0
    n_ei = 0

    if "ticker" not in w.columns and "symbol" in w.columns:
        w["ticker"] = w["symbol"]
    if "ticker" not in w.columns:
        w["ticker"] = ""

    w["_join_key"] = w["ticker"].map(join_key)

    w["price_used"] = np.nan
    w["price_source"] = ""

    # 1) Direct columns on the opportunity file
    for i in w.index:
        row = w.loc[i]
        px, cname = _first_valid_price_in_row(row, PRICE_COLUMN_ORDER)
        if px is not None and cname is not None:
            w.at[i, "price_used"] = px
            w.at[i, "price_source"] = f"trade_opportunities.{cname}"

    _plan = _read_aux_csv(execution_plan_path)
    _ei = _read_aux_csv(execution_intelligence_path)
    plan_map = _price_map_from_frame(
        _plan if _plan is not None else pd.DataFrame(), "execution_plan"
    )
    ei_map = _price_map_from_frame(
        _ei if _ei is not None else pd.DataFrame(), "execution_intelligence"
    )

    for i in w.index:
        pcur = w.at[i, "price_used"]
        if np.isfinite(pcur) and float(pcur) > 0:
            continue
        jk = w.at[i, "_join_key"]
        if not jk or not isinstance(jk, str):
            continue
        if jk in plan_map:
            px, src = plan_map[jk]
            w.at[i, "price_used"] = px
            w.at[i, "price_source"] = src
            n_plan += 1
        elif jk in ei_map:
            px, src = ei_map[jk]
            w.at[i, "price_used"] = px
            w.at[i, "price_source"] = src
            n_ei += 1

    # Count how many still lack price
    pu = _numeric(w["price_used"])
    missing_price_rows = int((~pu.notna() | (pu <= 0)).sum())

    debug = {
        "missing_price_rows": missing_price_rows,
        "n_from_trade_opportunities": int(
            (w["price_source"].astype(str).str.startswith("trade_opportunities.")).sum()
        ),
        "n_from_plan_merge": n_plan,
        "n_from_ei_merge": n_ei,
    }
    return w, debug


def _format_price_sources_used(series: pd.Series) -> str:
    if series is None or series.empty:
        return ""
    vc = series.astype(str)
    vc = vc[vc.str.len() > 0]
    if vc.empty:
        return ""
    c = Counter(vc)
    return ";".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda x: x[0]))


def _confidence_sizing_multipliers(
    conf_arr: np.ndarray,
    *,
    min_conf: Optional[float],
    max_conf: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    conf_norm in [0,1] from (confidence - min_conf) / (max_conf - min_conf) when both bounds
    are provided and valid; else use the batch min/max of confidence. sizing_weight = conf_norm ** 2.
    """
    c = np.asarray(conf_arr, dtype=float)
    if min_conf is not None and max_conf is not None and float(max_conf) > float(min_conf):
        lo, hi = float(min_conf), float(max_conf)
        conf_norm = (c - lo) / (hi - lo)
    else:
        lo = float(np.nanmin(c)) if c.size else 0.0
        hi = float(np.nanmax(c)) if c.size else 1.0
        if hi > lo:
            conf_norm = (c - lo) / (hi - lo)
        else:
            # All confidences equal: same normalized value for every row (no false spread)
            conf_norm = np.ones_like(c, dtype=float)
    conf_norm = np.clip(conf_norm, 0.0, 1.0)
    sizing_w = conf_norm**2
    return conf_norm, sizing_w, lo, hi


def _require_columns(df: pd.DataFrame) -> Optional[str]:
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        return f"missing required columns: {', '.join(miss)}"
    return None


def _print_sizing_summary(st: Dict[str, Any]) -> None:
    """[POSITION_SIZING] summary line; extras (price sources) available in stats only."""
    print(
        f"[POSITION_SIZING] total_capital={float(st.get('total_capital', 0) or 0):g} "
        f"num_trades={int(st.get('num_trades', 0) or 0)} "
        f"top_weight={float(st.get('top_weight', 0.0) or 0.0):.6f} "
        f"skipped_rows={int(st.get('skipped_rows', 0) or 0)}",
        flush=True,
    )


def size_opportunities(
    df: pd.DataFrame,
    *,
    deployable_capital: float = DEFAULT_CAPITAL,
    execution_plan_path: Path = DEFAULT_PLAN,
    execution_intelligence_path: Path = DEFAULT_EI,
    min_conf: Optional[float] = DEFAULT_NORM_MIN_CONF,
    max_conf: Optional[float] = DEFAULT_NORM_MAX_CONF,
    max_notional: Optional[float] = DEFAULT_MAX_NOTIONAL,
    max_position_weight: Optional[float] = DEFAULT_MAX_POSITION_WEIGHT,
    min_order_size: float = DEFAULT_MIN_ORDER_SIZE,
) -> Tuple[pd.DataFrame, dict]:
    """
    Returns (sized_df, stats) with total_capital, num_trades, top_weight, skipped_rows,
    price_sources_used, missing_price_rows, warning (optional).
    Sizing: conf_norm, weight=conf_norm**2, mult=0.5+weight*1.5, base=cap/n,
    position_value = cap * (mult / sum(mult)) to deploy full total_capital (literal base*mult would not
    add to cap), then cap max_notional / max_position_weight, min_order_size; score kept as c*(1+d).
    """
    empty_stats: Dict[str, Any] = {
        "total_capital": float(deployable_capital),
        "num_trades": 0,
        "top_weight": 0.0,
        "skipped_rows": 0,
        "price_sources_used": "",
        "missing_price_rows": 0,
        "warning": None,
    }
    if df is None or df.empty:
        return pd.DataFrame(), {**empty_stats, "warning": "empty input"}

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    if "score" in work.columns:
        work = work.rename(columns={"score": "upstream_score"})

    err = _require_columns(work)
    if err is not None:
        return pd.DataFrame(), {**empty_stats, "warning": err}

    n_in = len(work)
    work, enr_debug = enrich_prices(
        work,
        execution_plan_path=execution_plan_path,
        execution_intelligence_path=execution_intelligence_path,
    )
    missing_price_rows = int(enr_debug.get("missing_price_rows", 0))

    conf = _numeric(work["confidence"])
    dlt = _numeric(work["delta_pct"])
    pr = _numeric(work["price_used"])

    base_ok = conf.notna() & dlt.notna() & pr.notna() & (pr > 0)
    wv = work.loc[base_ok].copy()
    if wv.empty:
        return (
            pd.DataFrame(),
            {
                "total_capital": float(deployable_capital),
                "num_trades": 0,
                "top_weight": 0.0,
                "skipped_rows": n_in,
                "price_sources_used": "",
                "missing_price_rows": missing_price_rows,
                "warning": "all rows skipped (missing/invalid price after enrichment, NaN confidence, or NaN delta_pct)",
            },
        )

    c = _numeric(wv["confidence"])
    d = _numeric(wv["delta_pct"])
    px = _numeric(wv["price_used"])
    scores = c * (1.0 + d)
    n = int(len(wv))
    cap = float(deployable_capital)
    c_np = c.to_numpy(dtype=float)
    conf_norm, sizing_w, _lo, _hi = _confidence_sizing_multipliers(
        c_np, min_conf=min_conf, max_conf=max_conf
    )
    mult = 0.5 + sizing_w * 1.5
    s_mult = float(np.nansum(mult))
    if not np.isfinite(s_mult) or s_mult <= 0.0:
        mult = np.ones(n, dtype=float)
        s_mult = float(n)
    # PART 3: final_size = base_size * (0.5 + weight*1.5) with weight=conf_norm^2; then scale
    # notionals to sum to total_capital so the portfolio uses full deployable capital.
    target_positions = max(n, 1)
    base_size = cap / target_positions
    position_value = (cap * (mult / s_mult)).astype(float)
    if max_position_weight is not None and max_position_weight > 0.0:
        cap_tr = min(float(max_position_weight) * cap, cap)
        position_value = np.minimum(position_value, cap_tr)
    if max_notional is not None and max_notional > 0.0:
        position_value = np.minimum(position_value, float(max_notional))

    if "ticker" in wv.columns:
        tcol = wv["ticker"]
    elif "symbol" in wv.columns:
        tcol = wv["symbol"]
    else:
        tcol = pd.Series([""] * n, index=wv.index)
    for j in range(n):
        tkr = display_ticker(tcol.iloc[j])
        print(
            f"[POSITION_SIZING_DETAIL] ticker={tkr} "
            f"confidence={float(c_np[j]):.6f} "
            f"weight={float(sizing_w[j]):.6f} "
            f"final_size={float(position_value[j]):.4f}",
            flush=True,
        )

    wv["score"] = scores
    wv["conf_norm"] = conf_norm
    wv["sizing_weight"] = sizing_w
    wv["position_value"] = position_value
    wv["price_used"] = px
    # Keep price_source from enrichment
    if min_order_size and float(min_order_size) > 0.0:
        keep = wv["position_value"].astype(float) >= float(min_order_size)
        wv = wv.loc[keep].copy()
        px = _numeric(wv["price_used"])
    if not wv.empty and cap > 0:
        wv["weight"] = wv["position_value"].astype(float) / cap
    rs = wv["position_value"] / wv["price_used"]
    wv["shares"] = np.floor(rs.astype(float) * 100.0) / 100.0
    wv = wv.loc[wv["shares"].notna() & np.isfinite(wv["shares"]) & (wv["shares"] > 0)].copy()
    if not wv.empty and cap > 0:
        wv["weight"] = wv["position_value"].astype(float) / cap

    if wv.empty:
        return (
            pd.DataFrame(),
            {
                "total_capital": cap,
                "num_trades": 0,
                "top_weight": 0.0,
                "skipped_rows": n_in,
                "price_sources_used": _format_price_sources_used(
                    work.loc[base_ok, "price_source"]
                    if "price_source" in work.columns
                    else pd.Series()
                ),
                "missing_price_rows": missing_price_rows,
                "warning": "all rows removed after share rounding (shares <= 0 or invalid)",
            },
        )

    top_w = float(wv["weight"].max()) if len(wv) else 0.0
    psrc_used = _format_price_sources_used(wv["price_source"])
    stats: Dict[str, Any] = {
        "total_capital": cap,
        "num_trades": int(len(wv)),
        "top_weight": top_w,
        "skipped_rows": int(n_in - len(wv)),
        "price_sources_used": psrc_used,
        "missing_price_rows": int(missing_price_rows),
        "warning": None,
    }
    tail = [
        "ticker",
        "price_used",
        "price_source",
        "score",
        "conf_norm",
        "sizing_weight",
        "weight",
        "position_value",
        "shares",
    ]
    ordered_tail = [c for c in tail if c in wv.columns]
    other = [c for c in wv.columns if c not in ordered_tail and c != "_join_key"]
    wv = wv[other + ordered_tail]
    wv = wv.drop(columns=[c for c in ("_join_key",) if c in wv.columns], errors="ignore")
    return wv, stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Allocate capital from trade_opportunities.csv to orders_sized.csv (standalone).",
    )
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--execution-plan",
        type=Path,
        default=DEFAULT_PLAN,
        help="CSV for price enrichment (default data/results/execution_plan.csv)",
    )
    ap.add_argument(
        "--execution-intelligence",
        type=Path,
        default=DEFAULT_EI,
        help="CSV for price enrichment (default data/results/execution_intelligence.csv)",
    )
    ap.add_argument(
        "--deployable-capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help="Placeholder deployable notional (default 10000)",
    )
    ap.add_argument(
        "--min-conf",
        type=float,
        default=None,
        help="Override lower bound for confidence normalization (default: batch min)",
    )
    ap.add_argument(
        "--max-conf",
        type=float,
        default=None,
        help="Override upper bound for confidence normalization (default: batch max)",
    )
    ap.add_argument(
        "--max-notional",
        type=float,
        default=None,
        help="Per-order notional cap (dollars), optional",
    )
    ap.add_argument(
        "--max-position-weight",
        type=float,
        default=None,
        help="Max fraction of total deployable for one name (0-1), optional",
    )
    ap.add_argument(
        "--min-order-size",
        type=float,
        default=0.0,
        help="Drop rows with position_value below this notional (dollars, default 0)",
    )
    args = ap.parse_args(argv)

    p = args.in_path
    if not p.is_file() or p.stat().st_size == 0:
        _print_sizing_summary(
            {
                "total_capital": float(args.deployable_capital),
                "num_trades": 0,
                "top_weight": 0.0,
                "skipped_rows": 0,
            }
        )
        print("  (no input file)", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("", encoding="utf-8")
        print(
            "[position_sizer] warning: no input; wrote empty output",
            file=sys.stderr,
        )
        return 0

    df = pd.read_csv(p)
    out_df, st = size_opportunities(
        df,
        deployable_capital=args.deployable_capital,
        execution_plan_path=args.execution_plan,
        execution_intelligence_path=args.execution_intelligence,
        min_conf=args.min_conf,
        max_conf=args.max_conf,
        max_notional=args.max_notional,
        max_position_weight=args.max_position_weight,
        min_order_size=float(args.min_order_size or 0.0),
    )
    wmsg = st.get("warning")

    if wmsg and "missing required columns" in str(wmsg):
        _print_sizing_summary(
            {
                "total_capital": float(args.deployable_capital),
                "num_trades": 0,
                "top_weight": 0.0,
                "skipped_rows": 0,
            }
        )
        print(f"[position_sizer] error: {wmsg}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    if wmsg:
        print(f"[position_sizer] warning: {wmsg}", file=sys.stderr)

    _print_sizing_summary(st)
    if st.get("price_sources_used") or st.get("missing_price_rows", 0):
        print(
            f"  price_sources_used={st.get('price_sources_used', '')!s} "
            f"missing_price_rows={st.get('missing_price_rows', 0)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
