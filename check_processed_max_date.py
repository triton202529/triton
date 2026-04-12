import pandas as pd
from pathlib import Path

proc = Path("data/processed")
if not proc.exists():
    print("NO data/processed folder")
    raise SystemExit(0)

files = sorted(proc.glob("*.parquet"))
print("processed parquet files:", len(files))

best = None
for p in files[:50]:  # limit prints
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print("FAIL", p.name, e)
        continue

    col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if not col:
        continue

    mx = pd.to_datetime(df[col], errors="coerce").max()
    print(p.name, "max_date:", mx)

    if pd.notna(mx) and (best is None or mx > best[0]):
        best = (mx, p.name)

print("LATEST_PROCESSED_DATE:", best)
