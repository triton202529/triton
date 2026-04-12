# scripts/normalize_csv_headers.py
import csv
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

# mapping of common variants -> normalized name
COL_MAP = {
    "date": "date",
    "Date": "date",
    "timestamp": "date",
    "close": "close",
    "Close": "close",
    "price": "close",
    "Price": "close",
    "open": "open",
    "Open": "open",
    "high": "high",
    "High": "high",
    "low": "low",
    "Low": "low",
    "volume": "volume",
    "Volume": "volume",
    "symbol": "symbol",
}


def normalize_file(p: Path):
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"SKIP: {p} -> read error: {e}")
        return

    if df.empty:
        print(f"REMOVE (empty): {p}")
        # optionally remove empty file
        # p.unlink()
        return

    # rename columns based on map (case-sensitive keys in COL_MAP)
    new_cols = []
    for c in df.columns:
        new_cols.append(COL_MAP.get(c, c).lower())

    df.columns = new_cols

    # Ensure 'date' and 'close' present
    if "date" not in df.columns or "close" not in df.columns:
        print(
            f"WARNING: {p.name}: missing required columns after normalization (found: {list(df.columns)})"
        )
        # still write back normalized header for consistency
    else:
        # coerce types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    out = p.with_suffix(".normalized.csv")
    df.to_csv(out, index=False)
    print(f"Normalized -> {out.name}")


def main():
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        print("No CSVs found in data/")
        return
    for p in csvs:
        normalize_file(p)


if __name__ == "__main__":
    main()
