import sys
from pathlib import Path

# Ensure project root is importable (works even when script is under /tools)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.broker_alpaca import AlpacaBroker
import pandas as pd

dups = {"BITO", "UNG", "XLRE", "XLU", "GBTC"}
b = AlpacaBroker(mode="paper")

open_orders = b.list_orders(status="open", nested=True, limit=500) or []
rows = []
for o in open_orders:
    sym = o.get("symbol")
    if sym not in dups:
        continue
    rows.append(
        {
            "id": o.get("id"),
            "symbol": sym,
            "side": (o.get("side") or "").lower(),
            "status": o.get("status"),
            "submitted_at": o.get("submitted_at") or o.get("created_at"),
            "qty": o.get("qty"),
            "filled_qty": o.get("filled_qty"),
            "type": o.get("type"),
        }
    )

if not rows:
    print(
        "No OPEN orders found for duplicate symbols. (They may have filled or been canceled already.)"
    )
    raise SystemExit(0)

df = pd.DataFrame(rows)
df["submitted_at"] = pd.to_datetime(df["submitted_at"], errors="coerce", utc=True)
df = df.sort_values(["symbol", "side", "submitted_at"])

print("OPEN orders for duplicate symbols:")
print(df.to_string(index=False))

to_cancel = []
for (sym, side), g in df.groupby(["symbol", "side"]):
    g = g.sort_values("submitted_at")
    if len(g) <= 1:
        continue
    to_cancel += list(g.iloc[1:]["id"].astype(str).values)

print("\nCancelling extras:", len(to_cancel))
for oid in to_cancel:
    try:
        b.cancel_order(oid)
        print("CANCEL OK:", oid)
    except Exception as e:
        print("CANCEL FAIL:", oid, e)
