from datetime import datetime, timezone, time as dtime
import pandas as pd, os
from services.broker_alpaca import AlpacaBroker

EXEC_LOG = "data/results/live_orders.csv"
mode = os.getenv("ALPACA_ENV","paper")
key  = os.getenv("ALPACA_KEY_ID","")
sec  = os.getenv("ALPACA_SECRET_KEY","")
b = AlpacaBroker(key, sec, mode)

d = pd.read_csv(EXEC_LOG)
d = d[d["status"].isin(["ACCEPTED","NEW","OPEN"]) & d["broker_order_id"].notna()]
d = d.sort_values("timestamp").groupby("broker_order_id").tail(1)
cut = dtime(15, 0)  # 3:00pm local
now = datetime.now().time()
if now >= cut:
    for oid in d["broker_order_id"]:
        try: b.cancel_order(str(oid))
        except: pass
print(f"Checked {len(d)} stale orders; cancel attempted after {cut}.")
