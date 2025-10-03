# scripts/build_baseline.py
"""
Dynamic Baseline Builder for Triton
----------------------------------
Builds data/results/baseline/weights.csv from market_by_ticker.csv (+ optional fundamentals.csv).

Upgrades (2025-09-30/10-02)
- Broad market-cap detection + parsing (K/M/B/T, $, commas, sci-notation).
- Robust enrichment from fundamentals.csv:
    1) direct market_cap
    2) shares_outstanding * fundamentals price
    3) shares_outstanding * universe price   (CROSS-SOURCE BOOST)
    4) AUM proxies (for ETFs/Funds/Trusts)
- --drop-missing-cap: drop tickers with NaN/zero cap before Top-N.
- --keep-missing-equal: keep Top-N even if caps are missing; allocate an equal fallback slice to the missing group.
- --missing-share: fraction of total weight for the missing-cap group (default 0.20).
- --max-weight: cap any single name's final weight and re-distribute excess.
- --min-weight: optional minimum weight per name before rounding (e.g., 0.0005 = 5 bps).
- --fund-whitelist: force certain tickers to be treated as funds (for AUM proxy).
- NEW: --turnover-cap guardrail blends with prior weights to keep turnover under a threshold.
- Clear diagnostics for which columns were used + cap feasibility printout.
- Per-row notes include run flags and the row’s group (known/missing).
- Exact-sum rounding: CSV weights sum to exactly 1.0000 (largest-remainder).

Defaults:
- method=cap, top=20, min_price=1.0, min_volume=100_000, excludes {UNG,WFC,GE}
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import re

# ---------- Config defaults ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = REPO_ROOT / "data" / "results" / "market_by_ticker.csv"
FUNDAMENTALS_CSV = REPO_ROOT / "data" / "results" / "fundamentals.csv"
BASELINE_DIR = REPO_ROOT / "data" / "results" / "baseline"
BASELINE_FILE = BASELINE_DIR / "weights.csv"

# Permanent excludes (adjust as you like)
DEFAULT_TICKERS_EXCLUDE = {"UNG", "WFC", "GE"}

# ---------- Column candidates ----------
_CAP_CANDIDATES = [
    "market_cap",
    "Market Cap",
    "marketCap",
    "mktcap",
    "mktCap",
    "free_float_market_cap",
    "freeFloatMarketCap",
    "market_capitalization",
    "MarketCapitalization",
    "cap",
    "Cap",
]
_PRICE_CANDIDATES = [
    "last",
    "close",
    "price",
    "adj_close",
    "Adj Close",
    "Close",
    "Last",
    "Price",
    "regularMarketPrice",
    "regular_market_price",
    "nav",
    "NAV",
    "navPrice",
]
_VOL_CANDIDATES = [
    "avg_volume",
    "average_volume",
    "volume",
    "avg_dollar_volume",
    "avgVolume",
    "Average Volume",
]
_TICKER_CANDIDATES = ["ticker", "symbol", "Symbol", "Ticker", "SYMBOL"]

# ETF / fund detection
_QUOTETYPE_CANDIDATES = [
    "quoteType",
    "quote_type",
    "securityType",
    "type",
    "asset_type",
    "assetType",
    "category",
]
_ISETF_CANDIDATES = ["is_etf", "isETF", "ETF", "is_fund", "isFund", "fundFlag"]

# AUM / shares candidates (for enrichment)
_AUM_CANDIDATES = [
    "totalAssets",
    "total_assets",
    "Total Assets",
    "netAssets",
    "net_assets",
    "Net Assets",
    "AUM",
    "aum",
    "assetsUnderManagement",
    "fundAssets",
    "totalNetAssets",
    "total_net_assets",
]
_SHARES_CANDIDATES = [
    "shares_outstanding",
    "Shares Outstanding",
    "sharesOutstanding",
    "shares_out",
    "sharesOut",
    "float_shares",
    "Float",
]

_SUFFIX_MAP = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


# ---------- Helpers ----------
def coalesce_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    return None


def parse_num(x):
    """Parse $, commas, 1.2T/350B/25M/100k, sci-notation, floats/ints."""
    if pd.isna(x):
        return math.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    if not s:
        return math.nan
    m = re.match(r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*([KMBTkmbt]))?\s*$", s)
    if m:
        base = float(m.group(1))
        suf = m.group(2).upper() if m.group(2) else None
        return base * _SUFFIX_MAP.get(suf, 1.0)
    try:
        return float(s)
    except ValueError:
        return math.nan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Triton baseline weights.csv from market_by_ticker.csv"
    )
    p.add_argument(
        "--universe-csv",
        type=Path,
        default=DEFAULT_UNIVERSE,
        help="Path to market_by_ticker.csv",
    )
    p.add_argument(
        "--method",
        choices=["equal", "cap", "risk"],
        default="cap",
        help="Weighting method",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Take top-N by market cap (if available) or alphabetically",
    )
    p.add_argument("--min-price", type=float, default=1.0, help="Minimum last/close price filter")
    p.add_argument(
        "--min-volume",
        type=float,
        default=100_000,
        help="Minimum average daily volume filter",
    )
    p.add_argument("--exclude", nargs="*", default=[], help="Tickers to exclude (space-separated)")
    p.add_argument(
        "--drop-missing-cap",
        action="store_true",
        help="Drop tickers with NaN/zero cap before Top-N",
    )
    p.add_argument(
        "--keep-missing-equal",
        action="store_true",
        help="Keep Top-N even if caps are missing; allocate fallback equal slice to missing group",
    )
    p.add_argument(
        "--missing-share",
        type=float,
        default=0.20,
        help="Fraction of total weight reserved for missing-cap group (0..1). Used with --keep-missing-equal.",
    )
    p.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help="Cap any single name's final weight at this fraction (e.g., 0.15) and re-distribute excess.",
    )
    p.add_argument(
        "--min-weight",
        type=float,
        default=None,
        help="Minimum weight per name before rounding (e.g., 0.0005 for 5 bps).",
    )
    p.add_argument(
        "--fund-whitelist",
        nargs="*",
        default=[],
        help="Tickers to force-treat as funds (AUM allowed as cap proxy).",
    )
    p.add_argument(
        "--turnover-cap",
        type=float,
        default=None,
        help="Max allowed turnover vs previous baseline (e.g., 0.05 = 5%). If exceeded, blend with prior.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing baseline file",
    )
    return p.parse_args()


def normalize_series_to_weights(vals: pd.Series) -> pd.Series:
    vals = vals.clip(lower=0).astype(float)
    total = vals.sum()
    if total <= 0:
        n = len(vals)
        return pd.Series([1.0 / n] * n, index=vals.index)
    return vals / total


def round_and_fix(weights: pd.Series, decimals: int = 4) -> pd.Series:
    """
    Quantize to `decimals` with exact sum==1 using the largest-remainder method.
    """
    w = weights.clip(lower=0).astype(float)
    n = len(w)
    if n == 0:
        return w
    total = float(w.sum())
    if total <= 0:
        return pd.Series([1.0 / n] * n, index=w.index)

    w = w / total
    scale = 10**decimals

    raw = w.values * scale
    floors = np.floor(raw)
    rema = raw - floors

    deficit = int(round(scale - floors.sum()))
    if deficit > 0:
        order = np.argsort(-rema)  # biggest remainders first
        floors[order[:deficit]] += 1
    elif deficit < 0:
        order = np.argsort(rema)  # smallest remainders first
        take = min(-deficit, len(order))
        for i in range(take):
            j = order[i]
            if floors[j] > 0:
                floors[j] -= 1

    rounded = floors / scale
    return pd.Series(rounded, index=w.index)


def cap_and_renormalize(
    weights: pd.Series, cap: float, tol: float = 1e-12, max_iter: int = 1000
) -> pd.Series:
    """
    Iteratively cap weights at `cap` and re-distribute excess to uncapped names.
    If the requested cap is infeasible (cap < 1/n), it is relaxed to 1/n.
    """
    w = weights.astype(float).clip(lower=0).copy()
    n = len(w)
    if n == 0:
        return w
    s = w.sum()
    if not pd.isna(s) and s > 0:
        w /= s
    else:
        w[:] = 1.0 / n

    cap = float(cap)
    if cap <= 0 or cap >= 1:
        return w
    cap = max(cap, 1.0 / n)  # relax infeasible

    it = 0
    while True:
        it += 1
        over = w > cap + tol
        if not over.any() or it > max_iter:
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        under_count = int(under.sum())

        if under_count == 0:
            w[:] = 1.0 / n
            break

        pool = w[under].sum()
        if pool <= tol:
            w[under] = (1.0 - (over.sum() * cap)) / under_count
        else:
            w[under] *= 1.0 + (excess / pool)

        w /= max(w.sum(), 1e-12)

    w /= max(w.sum(), 1e-12)
    return w


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def backup_existing(target: Path):
    if target.exists():
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = target.with_name(f"{target.stem}.{ts}.bak.csv")
        backup.write_bytes(target.read_bytes())
        print(f"[backup] Existing baseline backed up -> {backup}")


def load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Universe CSV not found: {path}")
    return pd.read_csv(path)


def tag_is_fund(
    index: pd.Index,
    etf_flag: Optional[pd.Series] = None,
    qtype: Optional[pd.Series] = None,
) -> pd.Series:
    """Heuristic: ETF/FUND/TRUST detection (aligned to provided index)."""
    is_f = pd.Series(False, index=index)
    if etf_flag is not None:
        tmp = (
            etf_flag.reindex(index, fill_value=False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["1", "true", "yes", "y"])
        )
        is_f = is_f | tmp
    if qtype is not None:
        s = qtype.reindex(index, fill_value="").astype(str).str.upper()
        is_f = is_f | s.str.contains("ETF|FUND|TRUST|ETP|INDEX", regex=True)
    return is_f


def enrich_caps_from_fundamentals(uni: pd.DataFrame, fund_whitelist: set[str]) -> pd.DataFrame:
    """
    Fill missing/zero _mcap using fundamentals + universe price:
      1) use universe _mcap as-is where > 0
      2) else direct fundamentals cap
      3) else fundamentals shares * fundamentals price
      4) else fundamentals shares * universe price  (cross-source)
      5) else (if fund) AUM
    """
    if not FUNDAMENTALS_CSV.exists():
        return uni
    try:
        f = pd.read_csv(FUNDAMENTALS_CSV)
    except Exception:
        return uni
    if f.empty:
        return uni

    c_ftick = coalesce_col(f, _TICKER_CANDIDATES)
    if not c_ftick:
        return uni

    c_fcap = coalesce_col(f, _CAP_CANDIDATES)
    c_fprice = coalesce_col(f, _PRICE_CANDIDATES)
    c_fshares = coalesce_col(f, _SHARES_CANDIDATES)
    c_faum = coalesce_col(f, _AUM_CANDIDATES)
    c_fqtype = coalesce_col(f, _QUOTETYPE_CANDIDATES)
    c_fisetf = coalesce_col(f, _ISETF_CANDIDATES)

    f2 = f.rename(columns={c_ftick: "ticker"}).copy()
    f2["ticker"] = f2["ticker"].astype(str).str.upper()

    cap_direct = f2[c_fcap].map(parse_num) if c_fcap else pd.Series(math.nan, index=f2.index)
    price_f = f2[c_fprice].map(parse_num) if c_fprice else pd.Series(math.nan, index=f2.index)
    shares_f = f2[c_fshares].map(parse_num) if c_fshares else pd.Series(math.nan, index=f2.index)
    aum_f = f2[c_faum].map(parse_num) if c_faum else pd.Series(math.nan, index=f2.index)

    isfund_f = tag_is_fund(
        f2.index,
        f2[c_fisetf] if c_fisetf in f2.columns else None,
        f2[c_fqtype] if c_fqtype in f2.columns else None,
    )

    # Merge core fields needed for cross-source calc
    uni2 = uni.merge(
        f2[["ticker"]].assign(
            _f_cap_direct=cap_direct,
            _f_price=price_f,
            _f_shares=shares_f,
            _f_aum=aum_f,
            _f_isfund=isfund_f,
        ),
        on="ticker",
        how="left",
    )

    # Force-whitelist funds
    if fund_whitelist:
        uni2["_f_isfund"] = uni2["_f_isfund"] | uni2["ticker"].isin(list(fund_whitelist))

    # Start with current universe cap
    cap = uni2["_mcap"]

    # Direct fundamentals cap
    cap = cap.where(cap.notna() & (cap > 0), uni2["_f_cap_direct"])

    # Fundamentals shares * fundamentals price
    cap_from_shares_f = uni2["_f_shares"] * uni2["_f_price"]
    cap = cap.where(cap.notna() & (cap > 0), cap_from_shares_f)

    # Fundamentals shares * UNIVERSE price (cross-source boost)
    cap_from_cross = uni2["_f_shares"] * uni2["_price"]
    cap = cap.where(cap.notna() & (cap > 0), cap_from_cross)

    # AUM for funds
    cap = cap.where(cap.notna() & (cap > 0), uni2["_f_aum"].where(uni2["_f_isfund"], math.nan))

    uni2["_mcap"] = cap
    return uni2


def detect_columns(
    df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    c_ticker = coalesce_col(df, _TICKER_CANDIDATES)
    c_price = coalesce_col(df, _PRICE_CANDIDATES)
    c_vol = coalesce_col(df, _VOL_CANDIDATES)
    c_mcap = coalesce_col(df, _CAP_CANDIDATES)
    return c_ticker, c_price, c_vol, c_mcap


def choose_universe(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, dict]:
    c_ticker, c_price, c_vol, c_mcap = detect_columns(df)
    if not c_ticker:
        raise SystemExit("No 'ticker'/'symbol' column found in universe CSV.")

    uni = df.rename(columns={c_ticker: "ticker"}).copy()
    uni["ticker"] = uni["ticker"].astype(str).str.upper()
    uni["_price"] = uni[c_price].map(parse_num) if c_price else math.nan
    uni["_volume"] = uni[c_vol].map(parse_num) if c_vol else math.nan
    uni["_mcap"] = uni[c_mcap].map(parse_num) if c_mcap else math.nan

    # Filters
    tickers_exclude = set(DEFAULT_TICKERS_EXCLUDE) | set([t.upper() for t in (args.exclude or [])])
    uni = uni.drop_duplicates(subset=["ticker"])
    uni = uni[uni["ticker"].notna() & (uni["ticker"] != "")]

    if not math.isnan(args.min_price):
        uni = uni[(uni["_price"].isna()) | (uni["_price"] >= float(args.min_price))]
    if not math.isnan(args.min_volume):
        uni = uni[(uni["_volume"].isna()) | (uni["_volume"] >= float(args.min_volume))]
    if tickers_exclude:
        uni = uni[~uni["ticker"].isin(tickers_exclude)]

    # Enrich caps ALWAYS (fills gaps even if some caps already exist)
    uni_before = uni.copy()
    uni = enrich_caps_from_fundamentals(
        uni, fund_whitelist=set([t.upper() for t in args.fund_whitelist])
    )

    # Optional: drop missing/zero caps before Top-N
    if args.drop_missing_cap:
        uni = uni[uni["_mcap"].notna() & (uni["_mcap"] > 0)]

    # Ordering
    if uni["_mcap"].notna().any():
        uni = uni.sort_values(by=["_mcap"], ascending=False)
    else:
        uni = uni.sort_values(by=["ticker"])

    # Top-N
    topn = int(args.top) if args.top and args.top > 0 else len(uni)
    uni = uni.head(topn).reset_index(drop=True)

    diag = {
        "ticker_col": c_ticker,
        "price_col": c_price,
        "volume_col": c_vol,
        "cap_col": (
            c_mcap
            if uni_before["_mcap"].notna().any()
            else ("ENRICHED(fundamentals)" if uni["_mcap"].notna().any() else None)
        ),
        "cap_found": bool(uni["_mcap"].notna().any()),
        "rows": len(uni),
        "nonzero_caps": int((uni["_mcap"].fillna(0) > 0).sum()),
    }
    return uni, diag


def compute_weights_cap(uni: pd.DataFrame) -> pd.Series:
    n = len(uni)
    if n == 0:
        raise SystemExit("Universe is empty after filters — nothing to weight.")
    if "_mcap" in uni.columns and uni["_mcap"].notna().any():
        nz = uni["_mcap"].fillna(0) > 0
        if nz.sum() == 0:
            return pd.Series([1.0 / n] * n, index=uni.index)
        w = normalize_series_to_weights(uni.loc[nz, "_mcap"])
        full = pd.Series(0.0, index=uni.index)
        full.loc[nz] = w
        return full
    print("[warn] Market cap missing; falling back to equal.")
    return pd.Series([1.0 / n] * n, index=uni.index)


def compute_weights(uni: pd.DataFrame, method: str) -> pd.Series:
    method = (method or "cap").lower()
    n = len(uni)
    if method == "equal":
        return pd.Series([1.0 / n] * n, index=uni.index)
    if method == "cap":
        return compute_weights_cap(uni)
    if method == "risk":
        if "_mcap" in uni.columns and uni["_mcap"].notna().any() and (uni["_mcap"] > 0).any():
            proxy = 1.0 / (uni["_mcap"].clip(lower=1.0).pow(0.5))
            nz = proxy > 0
            w = normalize_series_to_weights(proxy[nz])
            full = pd.Series(0.0, index=uni.index)
            full.loc[nz] = w
            return full
        print("[warn] Risk proxy unavailable; falling back to equal.")
        return pd.Series([1.0 / n] * n, index=uni.index)
    return pd.Series([1.0 / n] * n, index=uni.index)


def load_previous_weights() -> Optional[pd.Series]:
    """
    Return prior weights as a Series indexed by ticker (upper), or None if not available.
    Searches the live baseline first, then the newest snapshot.
    """
    paths: list[Path] = []
    if BASELINE_FILE.exists():
        paths.append(BASELINE_FILE)
    if BASELINE_DIR.exists():
        snaps = sorted(
            BASELINE_DIR.glob("weights.*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        paths.extend(snaps)
    for path in paths:
        try:
            dfp = pd.read_csv(path)
            if "ticker" in dfp.columns and "target_weight" in dfp.columns:
                s = pd.Series(
                    dfp["target_weight"].values,
                    index=dfp["ticker"].astype(str).str.upper(),
                )
                s = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)
                return s
        except Exception:
            continue
    return None


def main():
    args = parse_args()
    if args.keep_missing_equal and args.drop_missing_cap:
        print(
            "[warn] --keep-missing-equal overrides --drop-missing-cap (we will keep missing and blend weights)."
        )

    df = load_universe(args.universe_csv)
    uni, diag = choose_universe(df, args)

    # Diagnostics
    print("[diagnostics]")
    print(f"  ticker_col = {diag['ticker_col']}")
    print(f"  price_col  = {diag['price_col']}")
    print(f"  volume_col = {diag['volume_col']}")
    print(f"  cap_col    = {diag['cap_col']}")
    print(f"  cap_found  = {diag['cap_found']}")
    print(f"  rows       = {diag['rows']}")
    print(f"  nonzero_caps = {diag['nonzero_caps']}")

    if len(uni) < 5:
        print(
            "[warn] Very small universe (<5). Proceeding, but consider relaxing filters or increasing --top)."
        )

    # Compute base weights (cap-weighting -> zeros for missing caps)
    weights = compute_weights(uni, args.method)

    # ---- Fixed-share missing logic & capping within pool ----
    mshare_eff = 0.0
    known_mask = pd.Series([True] * len(weights), index=weights.index)
    missing_mask = pd.Series([False] * len(weights), index=weights.index)

    if args.keep_missing_equal:
        mshare = max(0.0, min(1.0, float(args.missing_share)))
        nz = weights > 0
        missing_mask = ~nz
        missing_count = int(missing_mask.sum())
        mshare_eff = mshare if missing_count > 0 else 0.0
        known_mask = ~missing_mask

        if known_mask.sum() > 0:
            # Normalize known to pool (1 - mshare_eff)
            known_w = normalize_series_to_weights(weights.loc[known_mask]) * (1.0 - mshare_eff)
            if args.max_weight is not None and (1.0 - mshare_eff) > 0:
                # Cap only within the known pool using pool-relative cap
                pool_cap = float(args.max_weight) / (1.0 - mshare_eff)
                pool_cap = max(0.0, min(1.0, pool_cap))
                known_w = cap_and_renormalize(known_w, pool_cap)
                # Rescale back to exact pool sum after capping
                known_w = normalize_series_to_weights(known_w) * (1.0 - mshare_eff)
            else:
                known_w = normalize_series_to_weights(known_w) * (1.0 - mshare_eff)
            weights.loc[known_mask] = known_w
        else:
            weights.loc[:] = 0.0

        # Fixed equal slice for the missing-cap group
        if missing_count > 0 and mshare_eff > 0.0:
            weights.loc[missing_mask] = mshare_eff / missing_count
        else:
            weights.loc[missing_mask] = 0.0
    else:
        # No fixed missing share: cap across the whole vector if requested
        if args.max_weight is not None:
            weights = cap_and_renormalize(weights, float(args.max_weight))

    # ---- Optional minimum weight per name (pre-rounding) ----
    if args.min_weight is not None and float(args.min_weight) > 0:
        floor = float(args.min_weight)
        n = len(weights)
        if floor * n >= 1.0 - 1e-9:
            print(f"[warn] --min-weight={floor} infeasible for n={n}; ignoring.")
        else:
            weights = weights.clip(lower=floor)
            weights = normalize_series_to_weights(weights)
            if args.max_weight is not None:
                weights = cap_and_renormalize(weights, float(args.max_weight))
                weights = normalize_series_to_weights(weights)

    # ---- Turnover guardrail (blend with prior if needed) ----
    turnover_note = ""
    if args.turnover_cap is not None and args.turnover_cap > 0:
        prev = load_previous_weights()
        if prev is not None and not prev.empty:
            tickers_up = uni["ticker"].astype(str).str.upper()
            prev_aligned = prev.reindex(tickers_up).fillna(0.0).values
            new_base = weights.values.astype(float)

            abs_d = np.abs(new_base - prev_aligned).sum()
            raw_turnover = 0.5 * abs_d
            if raw_turnover > args.turnover_cap and abs_d > 0:
                alpha = min(1.0, (2.0 * float(args.turnover_cap)) / float(abs_d))
                blended = alpha * new_base + (1.0 - alpha) * prev_aligned
                blended = pd.Series(blended, index=weights.index)
                blended = blended.clip(lower=0.0)
                blended = normalize_series_to_weights(blended)
                if args.max_weight is not None:
                    blended = cap_and_renormalize(blended, float(args.max_weight))
                    blended = normalize_series_to_weights(blended)
                weights = blended
                print(
                    f"[turnover] raw={raw_turnover:.4f} > cap={args.turnover_cap:.4f} -> blended alpha={alpha:.4f}"
                )
                turnover_note = f" turnover_raw={raw_turnover:.4f} alpha={alpha:.4f}"
            else:
                turnover_note = f" turnover_raw={raw_turnover:.4f} alpha=1.0000"

    # ---- Cap feasibility diagnostics ----
    known_count = int(known_mask.sum())
    missing_count = int(missing_mask.sum())
    pool = 1.0 - mshare_eff
    feas_min = (pool / known_count) if known_count > 0 else None
    eff_cap = None
    if args.max_weight is not None:
        eff_cap = max(float(args.max_weight), (feas_min if feas_min is not None else 0.0))
    print("[cap-diagnostics]")
    print(f"  known_caps = {known_count}, missing_caps = {missing_count}")
    print(f"  missing_share = {mshare_eff:.2f}, known_pool = {pool:.2f}")
    if args.max_weight is not None:
        if feas_min is not None:
            print(
                f"  requested_cap = {args.max_weight:.4f}, feasible_min = {feas_min:.4f} -> effective_cap = {eff_cap:.4f}"
            )
        else:
            print(f"  requested_cap = {args.max_weight:.4f}, effective_cap = {args.max_weight:.4f}")

    # Exact-sum rounding to 4dp
    weights = round_and_fix(weights, decimals=4)

    # Build per-row notes (includes group)
    ts = dt.datetime.now().isoformat(timespec="seconds")
    base_notes = (
        f"baseline method={args.method}"
        f" top={args.top}"
        f" keep-missing-equal={int(bool(args.keep_missing_equal))}"
        f" missing-share={mshare_eff:.2f}"
        f" max-weight={(args.max_weight if args.max_weight is not None else 'NA')}"
        f" known={known_count} missing={missing_count}"
        f" cap-effective={(f'{eff_cap:.4f}' if eff_cap is not None else 'NA')}"
        f" ts={ts}"
        f"{turnover_note}"
    )
    # If we used --keep-missing-equal, group reflects known/missing; otherwise all are "known"
    row_groups = ["missing" if missing_mask.loc[i] else "known" for i in weights.index]
    row_notes = [f"{base_notes} group={g}" for g in row_groups]

    out = pd.DataFrame(
        {
            "ticker": uni["ticker"].tolist(),
            "target_weight": weights.tolist(),
            "source_weights": weights.apply(lambda x: f"{x:.4f}").astype(str).tolist(),
            "notes": row_notes,
        }
    )

    print("\n[preview] Baseline weights")
    print(out.to_string(index=False))

    if args.dry_run:
        print("\n[dry-run] Skipping write.")
        return

    ensure_dir(BASELINE_DIR)
    backup_existing(BASELINE_FILE)
    out.to_csv(BASELINE_FILE, index=False, float_format="%.4f")

    snapshot = BASELINE_DIR / f"weights.{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    out.to_csv(snapshot, index=False, float_format="%.4f")

    print(f"\n[ok] Baseline written -> {BASELINE_FILE}")
    print(f"[ok] Snapshot saved  -> {snapshot}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
