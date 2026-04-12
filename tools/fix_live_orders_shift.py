import pandas as pd

p = "data/results/live_orders.csv"
df = pd.read_csv(p)

# rows where status contains triton-* and client_order_id is empty -> shift fix
status = df.get("status", "").astype(str)
client = df.get("client_order_id", "")

m = status.str.startswith("triton-") & (client.isna() | (client.astype(str).str.strip() == ""))

df.loc[m, "client_order_id"] = df.loc[m, "status"]
df.loc[m, "status"] = ""

df.to_csv(p, index=False)
print("fixed_rows", int(m.sum()))
print(df[["action", "status", "client_order_id"]].tail(10).to_string(index=False))
