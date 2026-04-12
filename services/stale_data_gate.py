# services/stale_data_gate.py
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import sys

RAW_DIR = Path("data/raw")


def main(max_age_days: int = 3):
    # max_age_days=3 gives cushion for weekends/holidays/data delays
    csvs = list(RAW_DIR.glob("*.csv"))
    if not csvs:
        print("[stale_gate] NO RAW CSVs FOUND -> FAIL")
        sys.exit(2)

    newest = None
    newest_file = None

    for f in csvs:
        try:
            df = pd.read_csv(f)
            if "date" not in df.columns or df.empty:
                continue
            d = pd.to_datetime(df["date"], errors="coerce").dropna()
            if d.empty:
                continue
            m = d.max().date()
            if newest is None or m > newest:
                newest = m
                newest_file = f.name
        except Exception:
            continue

    if newest is None:
        print("[stale_gate] COULD NOT DETERMINE NEWEST RAW DATE -> FAIL")
        sys.exit(2)

    today = datetime.now(timezone.utc).date()
    age = (today - newest).days

    print(
        f"[stale_gate] newest_raw_date={newest} (from {newest_file}) age_days={age} today_utc={today}"
    )

    if age > max_age_days:
        print(f"[stale_gate] STALE DATA -> FAIL (age_days={age} > {max_age_days})")
        sys.exit(3)

    print("[stale_gate] OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
