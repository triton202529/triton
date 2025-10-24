#!/usr/bin/env python3
# scripts/repair_normalized_missing_date.py
import sys
from pathlib import Path
import pandas as pd
import re
import shutil


def guess_dates_from_filename(fname: str):
    # looks for _YYYY-MM-DD_to_YYYY-MM-DD in filename
    m = re.search(r"_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", fname)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def repair_file(path: Path, overwrite: bool = False):
    print("Processing:", path)
    if not path.exists():
        print("  -> file not found")
        return False

    # backup
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print("  -> backed up to", bak)

    # read, keep raw strings to avoid parse surprises
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    cols = list(df.columns)
    print("  -> parsed columns:", cols)

    # if date already present, nothing to do
    if any(c.strip().lower() == "date" for c in cols):
        print("  -> file already has 'date' column, skipping")
        return True

    # Drop any obviously junk row: row where first cell is empty and many cells equal ticker
    # Heuristic: first row values except first column are identical and match ticker in filename
    first_row = df.iloc[0].astype(str).tolist()
    rest_row = [s.strip() for s in first_row[1:]]
    all_equal = len(rest_row) > 0 and all(x == rest_row[0] for x in rest_row)
    if all_equal and (first_row[0] == "" or first_row[0].strip() == ""):
        print("  -> detected junk header-row with repeated ticker, dropping first data row")
        df = df.iloc[1:].reset_index(drop=True)
    else:
        print("  -> no obvious repeated-ticker junk row detected")

    # Attempt to build dates from filename
    start_s, end_s = guess_dates_from_filename(path.name)
    if not start_s or not end_s:
        print("  -> filename does not contain start/end dates; cannot auto-generate dates")
        out = path.with_name(path.stem + ".fixed" + path.suffix)
        df.to_csv(out, index=False)
        print("  -> wrote repaired file (no 'date' column):", out)
        return True

    # create business day range
    try:
        start = pd.to_datetime(start_s)
        end = pd.to_datetime(end_s)
        dates = pd.bdate_range(start=start, end=end)
    except Exception as e:
        print("  -> error parsing dates:", e)
        out = path.with_name(path.stem + ".fixed" + path.suffix)
        df.to_csv(out, index=False)
        print("  -> wrote repaired file (no 'date' column):", out)
        return True

    # If lengths match, insert date column
    if len(dates) == len(df):
        print(
            f"  -> dates range length {len(dates)} matches data rows {len(df)}; inserting 'date' column"
        )
        df.insert(0, "date", dates.strftime("%Y-%m-%d"))
        out = path.with_name(path.stem + ".fixed" + path.suffix)
        df.to_csv(out, index=False)
        print("  -> wrote fixed file:", out)
        return True
    else:
        # If mismatch, try len(dates) >= len(df) and take last N or first N depending on alignment
        if len(dates) >= len(df):
            print(
                f"  -> date-range length {len(dates)} >= data rows {len(df)}; taking last {len(df)} business days"
            )
            chosen = dates[-len(df) :].strftime("%Y-%m-%d")
            df.insert(0, "date", chosen)
            out = path.with_name(path.stem + ".fixed" + path.suffix)
            df.to_csv(out, index=False)
            print("  -> wrote fixed file with last-N dates:", out)
            return True
        else:
            print("  -> not enough dates in filename range to cover rows; aborting auto-insert")
            out = path.with_name(path.stem + ".fixed" + path.suffix)
            df.to_csv(out, index=False)
            print("  -> wrote repaired file (no 'date' column):", out)
            return True


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="normalized CSV files to repair")
    args = p.parse_args()
    for f in args.files:
        repair_file(Path(f))


if __name__ == "__main__":
    main()
