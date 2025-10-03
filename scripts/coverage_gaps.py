# scripts/coverage_gaps.py
"""
Coverage gaps scanner for Triton
--------------------------------
Scans universe + fundamentals to find tickers that are missing key inputs
needed to compute baseline (e.g., market cap for companies, AUM for funds).

Key features
- Robust boolean parsing (avoids pandas FutureWarning on .fillna with object dtypes)
- Fund detection from fundamentals (is_fund / is_fund_hint), ticker whitelist, and CLI
- Handles numbers like 1.2T / 350B / 25M / $1,234,567 and scientific notation
- Exclude noisy tickers via --exclude (defaults: UNG, WFC, GE)
- Writes timestamped CSV to data/results/baseline/coverage_gaps.YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations
import argparse
import datetime as dt
import math
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import re

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = REPO_ROOT / "data" / "results" / "market_by_ticker.csv"
DEFAULT_FUNDAMENTALS = REPO_ROOT / "data" / "results" / "fundamentals.csv"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "results" / "baseline"

# ---------- Heuristics ----------
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
]
_SHARES_CANDIDATES = [
    "shares_outstanding",
    "Shares Outstanding",
    "sharesOutstanding",
    "float_shares",
    "Float",
]
_TICKER_CANDIDATES = ["ticker", "symbol", "Symbol", "Ticker", "SYMBOL"]
_AUM_CANDIDATES = ["totalAssets", "aum", "AUM", "net_assets", "Net Assets", "netAssets"]

_FUND_FLAG_CANDIDATES_TRUE = [
    "is_fund",
    "isFund",
    "is_etf",
    "isETF",
    "is_index_fund",
    "isIndexFund",
    "is_fund_hint",
    "isFundHint",
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


def truthy(series: pd.Series) -> pd.Series:
    """
    Robust boolean parser:
    - Uses pandas 'string' dtype and avoids .fillna(False) on object arrays
    - Accepts: 1/true/t/yes/y (case-insensitive)
    """
    s = series.astype("string").fillna("false").str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"])


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_columns(
    df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    c_ticker = coalesce_col(df, _TICKER_CANDIDATES)
    c_price = coalesce_col(df, _PRICE_CANDIDATES)
    c_mcap = coalesce_col(df, _CAP_CANDIDATES)
    c_shares = coalesce_col(df, _SHARES_CANDIDATES)
    return c_ticker, c_price, c_mcap, c_shares


# ---------- Core ----------
def load_csv(path: Path, kind: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{kind} CSV not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Failed to read {kind} at {path}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Scan coverage gaps for Triton baseline inputs")
    ap.add_argument(
        "--universe-csv",
        type=Path,
        default=DEFAULT_UNIVERSE,
        help="Path to market_by_ticker.csv",
    )
    ap.add_argument(
        "--fundamentals-csv",
        type=Path,
        default=DEFAULT_FUNDAMENTALS,
        help="Path to fundamentals.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Where to write coverage_gaps.*.csv",
    )
    ap.add_argument(
        "--fund-whitelist", nargs="*", default=[], help="Tickers to force-mark as funds"
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=["UNG", "WFC", "GE"],
        help="Tickers to ignore in the gap report",
    )
    ap.add_argument("--sample", type=int, default=20, help="How many sample rows to print")
    args = ap.parse_args()

    uni = load_csv(args.universe_csv, "Universe")
    fnd = load_csv(args.fundamentals_csv, "Fundamentals")

    c_utick, c_uprice, c_umcap, _ = detect_columns(uni)
    c_ftick, c_fprice, c_fmcap, c_fshares = detect_columns(fnd)
    c_faum = coalesce_col(fnd, _AUM_CANDIDATES)

    if not c_utick:
        raise SystemExit("Universe missing ticker column.")
    if not c_ftick:
        raise SystemExit("Fundamentals missing ticker column.")

    # Normalize & slim
    u = uni.rename(columns={c_utick: "ticker"}).copy()
    f = fnd.rename(columns={c_ftick: "ticker"}).copy()
    u["ticker"] = u["ticker"].astype(str).str.upper()
    f["ticker"] = f["ticker"].astype(str).str.upper()

    # Deduplicate by ticker (keeps first occurrence)
    u = u.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    f = f.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    # Extract numeric inputs
    u["_cap_u"] = u[c_umcap].map(parse_num) if c_umcap else math.nan
    u["_price_u"] = u[c_uprice].map(parse_num) if c_uprice else math.nan

    f["_cap_f_direct"] = f[c_fmcap].map(parse_num) if c_fmcap else math.nan
    f["_price_f"] = f[c_fprice].map(parse_num) if c_fprice else math.nan
    f["_shares"] = f[c_fshares].map(parse_num) if c_fshares else math.nan
    f["_aum"] = f[c_faum].map(parse_num) if c_faum else math.nan

    # Best market cap from fundamentals: direct, or shares*price
    f["_cap_f_est"] = f["_shares"] * f["_price_f"]
    f["_cap_f"] = f["_cap_f_direct"].fillna(f["_cap_f_est"])

    # Merge
    merged = (
        u[["ticker", "_cap_u", "_price_u"]]
        .merge(
            f[
                ["ticker", "_cap_f", "_shares", "_price_f", "_aum"]
                + [c for c in f.columns if c in _FUND_FLAG_CANDIDATES_TRUE]
            ],
            on="ticker",
            how="outer",
        )
        .fillna({"_cap_u": math.nan, "_price_u": math.nan})
    )

    # Exclusions
    exclude_set = {t.upper() for t in (args.exclude or [])}
    if exclude_set:
        merged = merged[~merged["ticker"].isin(exclude_set)].reset_index(drop=True)

    # Fund flags: any truthy fund columns OR CLI whitelist
    fund_flags = pd.Series(False, index=merged.index)
    for c in _FUND_FLAG_CANDIDATES_TRUE:
        if c in merged.columns:
            fund_flags |= truthy(merged[c])
    if args.fund_whitelist:
        wl = {t.upper() for t in args.fund_whitelist}
        fund_flags |= merged["ticker"].isin(wl)
    merged["_is_fund"] = fund_flags

    # Gap logic
    # For funds: need positive _aum
    is_fund = merged["_is_fund"].fillna(False)
    aum_ok = pd.to_numeric(merged["_aum"], errors="coerce").fillna(0) > 0
    fund_gap = is_fund & (~aum_ok)

    # For non-funds: need a usable market cap (either from universe or fundamentals)
    cap_u_ok = pd.to_numeric(merged["_cap_u"], errors="coerce").fillna(0) > 0
    cap_f_ok = pd.to_numeric(merged["_cap_f"], errors="coerce").fillna(0) > 0
    nonfund_gap = (~is_fund) & (~(cap_u_ok | cap_f_ok))

    gaps = merged.loc[fund_gap | nonfund_gap].copy()

    # Reason text
    reasons = []
    for i, row in gaps.iterrows():
        if row.get("_is_fund", False):
            if not (
                pd.to_numeric(pd.Series([row.get("_aum")]), errors="coerce").fillna(0).iloc[0] > 0
            ):
                reasons.append("fund missing AUM")
            else:
                reasons.append("")
        else:
            cu = pd.to_numeric(pd.Series([row.get("_cap_u")]), errors="coerce").fillna(0).iloc[0]
            cf = pd.to_numeric(pd.Series([row.get("_cap_f")]), errors="coerce").fillna(0).iloc[0]
            sh = pd.to_numeric(pd.Series([row.get("_shares")]), errors="coerce").fillna(0).iloc[0]
            pf = pd.to_numeric(pd.Series([row.get("_price_f")]), errors="coerce").fillna(0).iloc[0]
            pu = pd.to_numeric(pd.Series([row.get("_price_u")]), errors="coerce").fillna(0).iloc[0]
            if cu <= 0 and cf <= 0 and sh <= 0 and pf <= 0 and pu <= 0:
                reasons.append("missing all cap inputs")
            elif cu <= 0 and cf <= 0:
                reasons.append("no cap_u and no cap_f")
            elif cf <= 0 and (sh <= 0 or pf <= 0):
                reasons.append("missing shares or price to estimate")
            else:
                reasons.append("incomplete inputs")
    gaps["reason"] = reasons

    # Select output columns
    out_cols = [
        "ticker",
        "_is_fund",
        "_cap_u",
        "_cap_f",
        "_shares",
        "_price_f",
        "_price_u",
        "_aum",
        "reason",
    ]
    for c in out_cols:
        if c not in gaps.columns:
            gaps[c] = pd.NA
    gaps = gaps[out_cols]

    # Write CSV
    ensure_dir(args.output_dir)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"coverage_gaps.{ts}.csv"
    gaps.to_csv(out_path, index=False)

    # Print summary + sample
    print("[coverage-gaps]")
    print(f"  gaps: {len(gaps)}  -> {out_path}")
    if len(gaps) > 0:
        print("\n(sample)")
        # pandas pretty print without index
        try:
            print(gaps.head(args.sample).to_string(index=False))
        except Exception:
            print(gaps.head(args.sample))


if __name__ == "__main__":
    main()
