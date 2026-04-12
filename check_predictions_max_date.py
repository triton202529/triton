import pandas as pd
from pathlib import Path

pred_dir = Path("data/predictions")
files = sorted(
    list(pred_dir.glob("*_predictions.parquet")) + list(pred_dir.glob("*_predictions.csv"))
)
print("files:", len(files))

best = None

for p in files:
    try:
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    except Exception as e:
        print("FAIL", p.name, e)
        continue

    col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if not col:
        print("NO_DATE_COL", p.name, "cols:", list(df.columns)[:12])
        continue

    s = pd.to_datetime(df[col], errors="coerce")
    mx = s.max()
    print(p.name, "max_date:", mx)

    if pd.notna(mx) and (best is None or mx > best[0]):
        best = (mx, p.name)

print("LATEST_PREDICTION_DATE:", best)
