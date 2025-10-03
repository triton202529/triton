import pandas as pd
from pathlib import Path

fpath = Path("data/results/fundamentals.csv")
df = pd.read_csv(fpath)

funds = {
    "ARKK": 7174e6,
    "BITO": 3673.83e6,
    "GBTC": 18735e6,
    "GLD": 124443.81e6,
    "DIA": 39443.50e6,
    "DBA": 815.94e6,
}

# Normalize tickers
if "ticker" not in df.columns:
    raise SystemExit("fundamentals.csv missing 'ticker' column")
df["ticker"] = df["ticker"].astype(str).str.upper()

mask = df["ticker"].isin(funds.keys())

# Ensure the boolean columns exist, then set them
if "is_fund" not in df.columns:
    df["is_fund"] = False
if "is_fund_hint" not in df.columns:
    df["is_fund_hint"] = False

df.loc[mask, "is_fund"] = True
df.loc[mask, "is_fund_hint"] = True

# Make sure AUM is present for those funds (keep existing if already set)
if "totalAssets" not in df.columns:
    df["totalAssets"] = pd.NA
for t, aum in funds.items():
    m = df["ticker"].eq(t)
    if m.any():
        if not m.any() or df.loc[m, "totalAssets"].isna().all():
            df.loc[m, "totalAssets"] = aum
        else:
            # If present but zero/negative, set it too
            df.loc[
                m & (pd.to_numeric(df["totalAssets"], errors="coerce").fillna(0) <= 0),
                "totalAssets",
            ] = aum

df.to_csv(fpath, index=False)
print(
    "Set fund flags for:", ", ".join(sorted(k for k in funds if k in set(df["ticker"])))
)
