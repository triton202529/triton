#!/usr/bin/env python3
"""
run_daily.py — minimal daily orchestrator

What it does (standalone-friendly):
1) Ensures results dir exists.
2) Optionally (re)builds market_by_ticker.csv from per-ticker parquet files.
3) Builds and saves Smart-Weight baseline curves/weights/turnover using signals and market returns.

You can run:
  python run_daily.py
  python run_daily.py --results-dir data/results --rebuild-market
  python run_daily.py --config config/baseline.smart_weight.json

Outputs:
  results/baseline/smart_weight_baseline_curves.csv
  results/baseline/smart_weight_daily_weights.csv
  results/baseline/smart_weight_daily_turnover.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Defaults / locations
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_RESULTS_DIR = Path("data/results")
DEFAULT_CFG_PATH = Path("config/baseline.smart_weight.json")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight IO helpers (standalone; no project imports required)
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(name_or_path: str | Path, base_dir: Path) -> pd.DataFrame:
    """Read CSV (if exists) and return DataFrame, else empty DF."""
    p = Path(name_or_path)
    if not p.is_absolute():
        p = base_dir / p
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        return df
    except Exception as e:
        print(f"[warn] Failed to read CSV: {p} — {e}", file=sys.stderr)
        return pd.DataFrame()


def ensure_date(df: pd.DataFrame, candidates=None, normalize=True) -> pd.DataFrame:
    """Coerce a best-effort date column; normalize to midnight if requested."""
    if df.empty:
        return df
    candidates = candidates or ["date", "as_of", "timestamp", "time", "datetime", "Date"]
    date_col = next((c for c in candidates if c in df.columns), None)
    if date_col is None:
        return df
    d = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    d = d.dt.tz_localize(None)
    if normalize:
        d = d.dt.normalize()
    df = df.copy()
    df["date"] = d
    return df


def to_numeric(df: pd.DataFrame, cols) -> None:
    """In-place numeric coercion; missing columns ignored."""
    if df is None or df.empty:
        return
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_parquet(path: Path) -> pd.DataFrame:
    """Read parquet if exists; else empty DF."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[warn] Failed to read parquet: {path} — {e}", file=sys.stderr)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Smart-Weight baseline builder (matches Tab 26 math, net of costs)
# ─────────────────────────────────────────────────────────────────────────────
def _load_baseline_cfg(cfg_path: Path) -> Dict:
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # sensible defaults if config missing
    return {
        "baseline": "smart_weight",
        "smart_weight": {
            "score_column": "confidence",
            "alpha": 0.30,
            "max_weight_cap": 0.15,           # 15%
            "trading_cost_bps": 5,            # 5 bps per $ traded
            "ema_smoothing_days": 1,          # 1 = off
            "min_hold_days": 1                # 1 = off
        },
        "benchmark": "avg_market",
        "chart_normalize_to_one": True,
    }


def _ema_smooth_wide(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """EMA smooth each column independently; span<=1 returns df unchanged."""
    span = int(span or 0)
    if span <= 1 or df.empty:
        return df
    return df.apply(lambda s: pd.to_numeric(s, errors="coerce").ewm(span=span, adjust=False).mean())


def _stabilize_weights(scores_wide: pd.DataFrame, max_pct: float) -> pd.DataFrame:
    """
    Row-normalize nonnegative scores, cap per-column at max_pct, renormalize; eq-weight fallback.
    """
    W = scores_wide.copy()
    W = W.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    W[W < 0] = 0.0

    # initial normalize
    row_sum = W.sum(axis=1)
    eq = pd.Series(1.0 / max(len(W.columns), 1), index=W.columns)
    for i in W.index:
        s = float(row_sum.loc[i])
        if s <= 0 or not np.isfinite(s):
            W.loc[i] = eq
        else:
            W.loc[i] = W.loc[i] / s

    # cap + renormalize remaining mass
    cap = float(max_pct)
    for i in W.index:
        w = W.loc[i].clip(lower=0.0)
        over = w > cap
        if over.any():
            capped = w.copy()
            capped[over] = cap
            rem = 1.0 - float(capped.sum())
            if rem > 1e-12:
                room = cap - capped
                room[room < 0] = 0.0
                room_sum = float(room.sum())
                if room_sum > 0:
                    capped += rem * (room / room_sum)
                else:
                    capped += rem / len(capped)
            total = float(capped.sum())
            W.loc[i] = capped / (total if total > 0 else 1.0)
        else:
            total = float(w.sum())
            W.loc[i] = w / (total if total > 0 else 1.0)
    return W


def _daily_turnover(W: pd.DataFrame) -> pd.Series:
    """0.5 * Σ|Δw| across columns."""
    if W is None or W.empty:
        return pd.Series(dtype=float)
    dW = W.diff().abs()
    return 0.5 * dW.sum(axis=1)


def build_market_by_ticker_if_missing(results_dir: Path) -> None:
    """
    If data/results/market_by_ticker.csv is missing, try to build it by scanning
    per-ticker parquet files under results_dir (e.g., results_dir / '{T}.parquet').
    Requires parquet files with columns ['date','close'].
    """
    out_csv = results_dir / "market_by_ticker.csv"
    if out_csv.exists():
        print("[info] market_by_ticker.csv already present.")
        return

    rets_list = []
    # Find all *.parquet directly under results_dir
    for pq in results_dir.glob("*.parquet"):
        ticker = pq.stem
        px = load_parquet(pq)
        if px.empty or "date" not in px.columns or "close" not in px.columns:
            continue
        px = px.copy()
        px["date"] = pd.to_datetime(px["date"], errors="coerce", utc=True).dt.tz_localize(None)
        px = px.dropna(subset=["date", "close"]).sort_values("date")
        px["ret"] = pd.to_numeric(px["close"], errors="coerce").pct_change()
        tmp = px[["date", "ret"]].copy()
        tmp["ticker"] = str(ticker)
        rets_list.append(tmp)

    if not rets_list:
        print("[warn] Could not build market_by_ticker.csv (no suitable parquet files found).")
        return

    mkt = pd.concat(rets_list, ignore_index=True)
    mkt.to_csv(out_csv, index=False)
    print(f"[ok] Built {out_csv} from {len(rets_list)} parquet files.")


def build_and_save_baseline(results_dir: Path, cfg_path: Path) -> None:
    """Build Smart-Weight baseline using signals + market returns; write CSVs used by the app."""
    cfg = _load_baseline_cfg(cfg_path)
    if cfg.get("baseline", "smart_weight") != "smart_weight":
        print("[warn] Config 'baseline' is not 'smart_weight'; proceeding anyway.")

    sw = cfg["smart_weight"]
    score_col   = sw.get("score_column", "confidence")
    alpha       = float(sw.get("alpha", 0.30))
    max_cap     = float(sw.get("max_weight_cap", 0.15))
    cost_bps    = float(sw.get("trading_cost_bps", 5)) / 1e4
    ema_span    = int(sw.get("ema_smoothing_days", 1))
    min_hold    = int(sw.get("min_hold_days", 1))

    # signals
    sig = load_csv("signals_with_rationale.csv", results_dir)
    if sig.empty:
        sig = load_csv("signals.csv", results_dir)
    if sig.empty or "ticker" not in sig.columns:
        print("[baseline] No signals available; skipping baseline build.")
        return

    sig = ensure_date(sig, candidates=["date", "as_of", "timestamp", "time", "datetime", "Date"], normalize=True)
    to_numeric(sig, [score_col])
    sig = sig.dropna(subset=["date", "ticker"])
    sig["ticker"] = sig["ticker"].astype(str)

    sc = (
        sig[["date", "ticker", score_col]]
        .pivot_table(index="date", columns="ticker", values=score_col, aggfunc="last")
        .sort_index()
    )
    sc = _ema_smooth_wide(sc, ema_span)

    # returns
    mkt = load_csv("market_by_ticker.csv", results_dir)
    if mkt.empty:
        print("[baseline] market_by_ticker.csv missing; skipping baseline build.")
        return
    mkt = mkt[["date", "ticker", "ret"]].copy()
    mkt["ticker"] = mkt["ticker"].astype(str)
    mkt["ret"] = pd.to_numeric(mkt["ret"], errors="coerce").fillna(0.0)
    mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce", utc=True).dt.tz_localize(None)
    R = mkt.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last").sort_index()

    # overlap
    cols = sorted(set(sc.columns) & set(R.columns))
    if not cols:
        print("[baseline] No overlap between signals and returns universe; skipping baseline build.")
        return
    sc = sc[cols]
    R  = R[cols]

    # equal weights for blend
    eq = pd.DataFrame(np.full((len(sc.index), len(cols)), 1.0 / len(cols)), index=sc.index, columns=cols)

    # smart weights
    W_s = _stabilize_weights(sc, max_cap)
    # optional min-hold: force weights to persist for N days by averaging recent rows
    if min_hold and min_hold > 1:
        W_s = W_s.rolling(min_hold, min_periods=1).mean()

    # blend
    Wf = alpha * W_s + (1.0 - alpha) * eq
    Wf = Wf.div(Wf.sum(axis=1), axis=0)

    # align with returns
    idx = Wf.index.intersection(R.index)
    Wf = Wf.loc[idx, cols]
    Rt = R.loc[idx, cols]

    # returns + costs
    gross_ret = (Wf * Rt).sum(axis=1).fillna(0.0)
    tvr = _daily_turnover(Wf)
    cost = cost_bps * tvr.reindex(gross_ret.index).fillna(0.0)
    net_ret = gross_ret - cost
    bench_ret = Rt.mean(axis=1).fillna(0.0)

    curves = pd.DataFrame({
        "portfolio_gross": (1.0 + gross_ret).cumprod(),
        "portfolio_net":   (1.0 + net_ret).cumprod(),
        "benchmark":       (1.0 + bench_ret).cumprod(),
    })
    curves.index.name = "date"

    out_dir = results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_dir / "smart_weight_baseline_curves.csv", index=True)
    Wf.to_csv(out_dir / "smart_weight_daily_weights.csv", index=True)
    tvr.to_csv(out_dir / "smart_weight_daily_turnover.csv", index=True)
    print(f"[ok] Saved smart-weight baseline to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Minimal daily runner for Triton.")
    ap.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                    help="Path to results directory (default: data/results)")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CFG_PATH),
                    help="Path to baseline config JSON (default: config/baseline.smart_weight.json)")
    ap.add_argument("--rebuild-market", action="store_true",
                    help="If set, rebuild market_by_ticker.csv from per-ticker parquet files.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    cfg_path = Path(args.config).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_market:
        build_market_by_ticker_if_missing(results_dir)

    build_and_save_baseline(results_dir, cfg_path)
    print("[done] run_daily complete.")


if __name__ == "__main__":
    main()
