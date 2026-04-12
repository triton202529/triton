# scripts/merge_fundamentals_fill.py
"""
Merge a "fill" CSV (manual/exported missing fundamentals template) into
data/results/fundamentals.csv.

Clean upgrades:
- Safer numeric parsing (handles $, commas, suffixes K/M/B/T)
- Idempotent update logic (only overwrites when fill value is valid > 0)
- Preserves ALL existing fundamentals columns (doesn't drop extra columns)
- Creates fundamentals.csv if missing
- Backs up existing fundamentals before writing

Usage:
  python scripts/merge_fundamentals_fill.py --fill data/results/missing_fundamentals_fill.csv
  python scripts/merge_fundamentals_fill.py --fill <path> --fundamentals data/results/fundamentals.csv
"""

import argparse
import math
import re
import datetime as dt
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DEFAULT_FUND_PATH = Path("data/results/fundamentals.csv")
SUF = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_num(x: Any) -> float:
    """Parse numbers like '1.2B', '$3,400,000', '5.6M' into float."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)

    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return math.nan

    s = s.replace("$", "").replace(",", "")
    m = re.match(r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*([KMBTkmbt]))?\s*$", s)
    if m:
        base = float(m.group(1))
        suf = m.group(2)
        if suf:
            return base * SUF.get(suf.upper(), 1.0)
        return base

    try:
        return float(s)
    except Exception:
        return math.nan


def is_good_value(v: Any) -> bool:
    """Valid numeric value we should use to overwrite: finite and > 0."""
    try:
        if pd.isna(v):
            return False
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return False
        return fv > 0
    except Exception:
        return False


def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_name(f"{path.stem}.{ts}.bak.csv")
    bak.write_bytes(path.read_bytes())
    return bak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fill",
        type=Path,
        required=True,
        help="CSV from export_missing_template.py (must have ticker)",
    )
    ap.add_argument("--fundamentals", type=Path, default=DEFAULT_FUND_PATH)
    args = ap.parse_args()

    if not args.fill.exists():
        raise SystemExit(f"[err] Fill file not found: {args.fill}")

    fill = pd.read_csv(args.fill)
    if "ticker" not in fill.columns:
        raise SystemExit("[err] Fill file needs a 'ticker' column.")
    fill["ticker"] = fill["ticker"].astype(str).str.upper().str.strip()

    # Normalize likely numeric fields if present
    numeric_like = ("market_cap", "shares_outstanding", "price", "totalAssets")
    for col in numeric_like:
        if col in fill.columns:
            fill[col] = fill[col].apply(parse_num)

    # Load existing fundamentals (or create minimal)
    if args.fundamentals.exists():
        fund = pd.read_csv(args.fundamentals)
        if "ticker" not in fund.columns:
            fund["ticker"] = []
    else:
        fund = pd.DataFrame(columns=["ticker"])

    fund["ticker"] = fund["ticker"].astype(str).str.upper().str.strip()

    # Outer merge to include new tickers and preserve all existing columns
    merged = fund.merge(fill, on="ticker", how="outer", suffixes=("", "_fill"))

    # For each column present in fill (excluding ticker), choose fill value if good, else keep existing
    for col in fill.columns:
        if col == "ticker":
            continue
        fill_col = f"{col}_fill"
        if fill_col not in merged.columns:
            continue

        if col in numeric_like:
            # numeric overwrite rule: only use fill if finite and > 0
            merged[col] = [
                b if is_good_value(b) else a
                for a, b in zip(
                    merged.get(col, pd.Series([math.nan] * len(merged))), merged[fill_col]
                )
            ]
        else:
            # non-numeric overwrite rule: use fill if non-empty
            def choose_text(a, b):
                if pd.isna(b):
                    return a
                sb = str(b).strip()
                if sb and sb.lower() not in ("nan", "none", "null"):
                    return b
                return a

            merged[col] = [
                choose_text(a, b)
                for a, b in zip(merged.get(col, pd.Series([pd.NA] * len(merged))), merged[fill_col])
            ]

    # Drop all *_fill helper columns
    merged = merged[[c for c in merged.columns if not c.endswith("_fill")]]

    # Sort for readability
    if "ticker" in merged.columns:
        merged["ticker"] = merged["ticker"].astype(str).str.upper().str.strip()
        merged = merged.sort_values("ticker").reset_index(drop=True)

    # Backup + write
    args.fundamentals.parent.mkdir(parents=True, exist_ok=True)
    bak = backup_file(args.fundamentals)
    if bak:
        print(f"[backup] Existing fundamentals backed up -> {bak}")

    merged.to_csv(args.fundamentals, index=False, encoding="utf-8")
    print(f"[ok] Fundamentals updated -> {args.fundamentals}")


if __name__ == "__main__":
    main()
