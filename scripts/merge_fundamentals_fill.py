# scripts/merge_fundamentals_fill.py
import argparse, math, re, datetime as dt
from pathlib import Path
import pandas as pd

FUND_PATH = Path("data/results/fundamentals.csv")

SUF = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_num(x):
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
        return base * SUF.get(suf, 1.0)
    try:
        return float(s)
    except:
        return math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", type=Path, required=True, help="CSV from export_missing_template.py")
    ap.add_argument("--fundamentals", type=Path, default=FUND_PATH)
    args = ap.parse_args()

    if not args.fill.exists():
        raise SystemExit(f"[err] Fill file not found: {args.fill}")

    fill = pd.read_csv(args.fill)
    if "ticker" not in fill.columns:
        raise SystemExit("[err] Fill file needs a 'ticker' column.")
    fill["ticker"] = fill["ticker"].astype(str).str.upper()

    # Normalize numeric fields if present
    for col in ("market_cap", "shares_outstanding", "price", "totalAssets"):
        if col in fill.columns:
            fill[col] = fill[col].apply(parse_num)

    # Load existing fundamentals (or create minimal)
    if args.fundamentals.exists():
        fund = pd.read_csv(args.fundamentals)
    else:
        fund = pd.DataFrame(columns=["ticker"])
    if "ticker" not in fund.columns:
        fund["ticker"] = []
    fund["ticker"] = fund["ticker"].astype(str).str.upper()

    # Prepare columns to merge/update
    # We'll prefer standard names recognized by builder:
    # market_cap, shares_outstanding, (price optional), totalAssets for funds
    keep_cols = [
        c
        for c in ["ticker", "market_cap", "shares_outstanding", "price", "totalAssets"]
        if c in fill.columns
    ]
    f2 = fund.copy()

    # Left-merge to bring in new values (update existing rows, append new tickers)
    merged = f2.merge(fill[keep_cols], on="ticker", how="outer", suffixes=("", "_fill"))

    def choose(a, b):
        # prefer b if finite and >0, else a
        if pd.notna(b) and b > 0:
            return b
        return a

    for col in ("market_cap", "shares_outstanding", "price", "totalAssets"):
        if col in merged.columns and f"{col}_fill" in merged.columns:
            merged[col] = [choose(a, b) for a, b in zip(merged[col], merged[f"{col}_fill"])]

    # Drop helper columns
    merged = merged[[c for c in merged.columns if not c.endswith("_fill")]]

    # Backup + write
    args.fundamentals.parent.mkdir(parents=True, exist_ok=True)
    if args.fundamentals.exists():
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = args.fundamentals.with_name(f"{args.fundamentals.stem}.{ts}.bak.csv")
        bak.write_bytes(args.fundamentals.read_bytes())
        print(f"[backup] Existing fundamentals backed up -> {bak}")

    merged.to_csv(args.fundamentals, index=False)
    print(f"[ok] Fundamentals updated -> {args.fundamentals}")


if __name__ == "__main__":
    main()
