import pandas as pd

p = "data/live/orders_today.csv"
df = pd.read_csv(p)

# normalize column names
df.columns = [str(c).strip().lower() for c in df.columns]

# locate symbol column
sym_col = None
for c in ["symbol", "ticker", "sym"]:
    if c in df.columns:
        sym_col = c
        break
if sym_col is None:
    raise SystemExit(f"[BAD_SCHEMA] No symbol/ticker/sym column. cols={df.columns.tolist()}")

# locate limit column
limit_col = None
for c in ["limit_price", "close", "price", "limit"]:
    if c in df.columns:
        limit_col = c
        break
if limit_col is None:
    # create limit_price if missing
    limit_col = "limit_price"
    df[limit_col] = ""

# update UNG
mask = df[sym_col].astype(str).str.strip().str.upper().eq("UNG")
if mask.sum() == 0:
    raise SystemExit("[NOT_FOUND] UNG not found in orders_today.csv")

df.loc[mask, limit_col] = 17.00

df.to_csv(p, index=False)
print(f"[OK] Updated UNG {limit_col}=17.00 in {p}. rows_updated={int(mask.sum())}")
