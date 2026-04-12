from pathlib import Path
import pandas as pd

p = Path("data/results/portfolio_history.csv")
if not p.exists():
    raise SystemExit(f"Missing file: {p}")

df = pd.read_csv(p)

if "timestamp" in df.columns and "date" not in df.columns:
    df = df.rename(columns={"timestamp": "date"})
    df.to_csv(p, index=False)
    print("✅ Renamed 'timestamp' → 'date'")
else:
    print("ℹ️ No rename needed. Columns:", df.columns.tolist())
