import pandas as pd
from services.preprocess_data import preprocess_stock_csv

fp = r"data/raw/AAPL_2020-07-08_to_2025-07-07.csv"

raw = pd.read_csv(fp)
print("RAW shape:", raw.shape)
print("RAW cols:", list(raw.columns))
print(
    "RAW date min/max:",
    pd.to_datetime(raw["date"], errors="coerce").min(),
    pd.to_datetime(raw["date"], errors="coerce").max(),
)
print(raw.head(3))

out = preprocess_stock_csv(fp)
print("\nOUT shape:", out.shape)
print("OUT cols:", list(out.columns))
if "date" in out.columns:
    print(
        "OUT date min/max:",
        pd.to_datetime(out["date"], errors="coerce").min(),
        pd.to_datetime(out["date"], errors="coerce").max(),
    )
print(out.head(3))
print(out.tail(3))
