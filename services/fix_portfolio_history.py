import pandas as pd

path = r"data\results\portfolio_history.csv"
df = pd.read_csv(path)

if "total_value" not in df.columns:
    if "equity" in df.columns:
        df["total_value"] = pd.to_numeric(df["equity"], errors="coerce")
    elif "portfolio_value" in df.columns:
        df["total_value"] = pd.to_numeric(df["portfolio_value"], errors="coerce")
    else:
        raise SystemExit("No equity or portfolio_value column found")

df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
df = df[df["total_value"].notna()].copy()

df.to_csv(path, index=False)
print("OK: portfolio_history.csv normalized")
