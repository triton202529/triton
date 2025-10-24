# scripts/normalize_csv_headers_v2.py
"""
Normalize CSV headers in data/ for Triton pipeline.

- Maps common header variants (Date, PRICE, Close, etc.) to canonical names:
  date, open, high, low, close, volume, symbol
- Collapses duplicate columns after mapping by preferring left-most non-null values.
- Writes <original>.normalized.csv by default. Use --overwrite to replace originals.
- Skips files already ending with .normalized.csv to avoid double-processing.
"""
import argparse
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
REQUIRED = {"date", "close"}

# mapping of common variants -> normalized name (case-insensitive)
COL_MAP_LOWER = {
    "date": "date",
    "timestamp": "date",
    "datetime": "date",
    "time": "date",
    "day": "date",
    "close": "close",
    "price": "close",
    "adjclose": "close",
    "adj_close": "close",
    "adj close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "symbol": "symbol",
    "ticker": "symbol",
}


def normalize_columns(cols):
    """
    Map each original column name to a normalized name (or keep a safe fallback).
    Returns a list of normalized names in the same order as `cols`.
    """
    normalized = []
    for c in cols:
        if c is None:
            normalized.append(c)
            continue
        key = str(c).strip()
        # primary match on lower()
        nm = COL_MAP_LOWER.get(key.lower())
        if nm:
            normalized.append(nm)
            continue
        # fallback: remove spaces/underscores and try again
        key2 = key.replace(" ", "").replace("_", "").lower()
        nm = COL_MAP_LOWER.get(key2)
        if nm:
            normalized.append(nm)
            continue
        # default to safe lowercased original (so we can still detect/inspect)
        normalized.append(key.lower())
    return normalized


def collapse_duplicates_preserve_order(df, original_cols, normalized_cols):
    """
    Collapse duplicate normalized column names.

    - original_cols: list of original column names (in same order as df.columns)
    - normalized_cols: list of normalized names corresponding to original_cols
    - For each original column (left to right), take its series (by index) and:
       - if the normalized name is new, set it
       - if it already exists, update missing values in existing with values from this column
    """
    new_df = pd.DataFrame(index=df.index)
    # iterate by position to avoid pandas returning a DataFrame for duplicate labels
    for idx, (orig_col, norm_col) in enumerate(zip(original_cols, normalized_cols)):
        series = df.iloc[:, idx]  # guarantees a Series even if labels duplicate
        # keep original name on series for traceability (not necessary)
        if norm_col not in new_df.columns:
            # assign the series (copy)
            new_df[norm_col] = series.copy()
        else:
            # fill existing NaNs with values from this series (left-to-right precedence)
            existing = new_df[norm_col]
            # prefer existing when not null, else take the incoming value
            combined = existing.where(existing.notna(), series)
            new_df[norm_col] = combined
    return new_df


def process_file(p: Path, overwrite=False):
    # Skip already-normalized files
    if p.name.endswith(".normalized.csv") and not overwrite:
        print(f"SKIP (already normalized): {p.name}")
        return False

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"SKIP: {p.name} -> read error: {e}")
        return False

    if df.empty:
        print(f"SKIP (empty): {p.name}")
        return False

    original_cols = list(df.columns)
    normalized_cols = normalize_columns(original_cols)

    # Rename columns temporarily to their normalized equivalents (not collapsing duplicates yet)
    # We'll collapse duplicates in a controlled way using positional indexing
    # Create a copy to preserve original df while we reconstruct columns
    try:
        reconstructed = collapse_duplicates_preserve_order(df, original_cols, normalized_cols)
    except Exception as ex:
        print(f"ERROR collapsing duplicates for {p.name}: {ex}")
        return False

    df = reconstructed

    # small heuristic: rename date_ -> date if present
    if "date_" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"date_": "date"})

    # coerce types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    missing = REQUIRED - set(df.columns)
    out_path = p.with_suffix(".normalized.csv")
    if overwrite:
        out_path = p

    try:
        df.to_csv(out_path, index=False)
    except Exception as e:
        print(f"ERROR writing {out_path.name}: {e}")
        return False

    if missing:
        print(
            f"WARNING: {p.name}: missing required columns after normalization (found: {list(df.columns)}) -> wrote {out_path.name}"
        )
    else:
        print(f"Normalized -> {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Normalize CSV headers in data/ for Triton pipeline."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original CSV files with normalized output",
    )
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern (default: *.csv)")
    args = parser.parse_args()

    files = sorted(DATA_DIR.glob(args.pattern))
    if not files:
        print("No CSV files found in data/")
        return

    for p in files:
        # skip .normalized.csv unless overwrite specified
        if p.name.endswith(".normalized.csv") and not args.overwrite:
            # don't double process
            continue
        process_file(p, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
