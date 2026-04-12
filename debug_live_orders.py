import pandas as pd

df = pd.read_csv("data/results/live_orders.csv")
s = "2026-01-25_AM_R1"

d = df[df["session"].astype(str).str.strip() == s]

print("rows for session:", len(d))
print("action counts:", d["action"].value_counts(dropna=False).to_dict())
print("submit rows:", int((d["action"].astype(str).str.lower() == "submit").sum()))
