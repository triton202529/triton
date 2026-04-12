#!/usr/bin/env python3
"""
Auto-fix price CSVs in data/:
- If a CSV lacks a date column, synthesize business-day dates from the filename
  <TICKER>_YYYY-MM-DD_to_YYYY-MM-DD.csv
- If can't parse dates, try borrowing from a sibling CSV with dates
- Cleans price/volume (commas, $ etc.) and writes <name>.fixed.csv
"""

import re
from pathlib import Path
import sys
import pandas as pd

DATA_DIR = Path("data")
DATE_RE = re.compile(r".*_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})")

# make sure we can import normalize_prices from the same folder as this script
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from normalize_prices import (  # noqa: E402
    _read_with_sniff,
    _pick_date_column,
    _pick_price_column,
    _pick_volume_column,
    _get_dates_from_other,
    _synthesize_dates,
    _clean_numeric_series,
)


def find_sibling_with_dates(ticker: str, self_path: Path) -> Path | None:
    for f in DATA_DIR.glob(f"{ticker}_*.csv"):
        if f == self_path or f.suffix != ".csv":
            continue
        try:
            df = _read_with_sniff(f)
            if _pick_date_column(df):
                return f
        except Exception:
            pass
    return None


def normalize_file(csv_path: Path) -> bool:
    try:
        df = _read_with_sniff(csv_path)
        dcol = _pick_date_column(df)
        pcol = _pick_price_column(df)
        vcol = _pick_volume_column(df)
        if pcol is None:
            print(f"  ❌ {csv_path.name}: no price-like column")
            return False

        # determine dates
        if dcol is not None:
            dates = pd.to_datetime(df[dcol], errors="coerce")
        else:
            m = DATE_RE.match(csv_path.name)
            if m:
                start = m.group(1)
                dates = _synthesize_dates(start, len(df))
            else:
                tkr = csv_path.stem.split("_")[0]
                sib = find_sibling_with_dates(tkr, csv_path)
                if not sib:
                    print(f"  ❌ {csv_path.name}: no date col and no sibling with dates; skip")
                    return False
                dates = _get_dates_from_other(sib, len(df))

        price = _clean_numeric_series(df[pcol])
        vol = _clean_numeric_series(df[vcol]) if vcol else None

        out = pd.DataFrame({"date": dates, "close": price})
        if vol is not None:
            out["volume"] = vol

        out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if out.empty:
            print(f"  ❌ {csv_path.name}: no valid rows after cleaning")
            return False

        out_path = csv_path.with_suffix(".fixed.csv")
        out.to_csv(out_path, index=False)
        print(f"  ✅ wrote {out_path.name} (rows={len(out)})")
        return True
    except Exception as e:
        print(f"  ❌ {csv_path.name}: {e}")
        return False


def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        print("No CSVs found in data/")
        return
    fixed, skipped = 0, 0
    for f in files:
        ok = normalize_file(f)
        fixed += int(ok)
        skipped += int(not ok)
    print(f"\nDone. Fixed: {fixed} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
