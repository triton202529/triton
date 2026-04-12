import os, requests, math, time

TP_PCT = float(os.getenv("TRITON_TP_PCT", "0.08"))
SL_PCT = float(os.getenv("TRITON_SL_PCT", "0.05"))
TIF = os.getenv("TRITON_OCO_TIF", "gtc")

H = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET"),
}
B = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets").rstrip("/")

open_orders = requests.get(
    f"{B}/v2/orders", headers=H, params={"status": "open", "nested": "true", "limit": 500}
).json()
already_protected = {o["symbol"] for o in open_orders if (o.get("side") == "sell")}

positions = requests.get(f"{B}/v2/positions", headers=H).json()

created = 0
for p in positions:
    try:
        sym = p["symbol"]
        if sym in already_protected:
            continue
        qty_f = float(p["qty"])
        qty = int(qty_f)
        if qty <= 0:
            continue
        avg = float(p["avg_entry_price"])
        tp = round(avg * (1.0 + TP_PCT), 2)
        sl = round(avg * (1.0 - SL_PCT), 2)

        payload = {
            "symbol": sym,
            "qty": str(qty),
            "side": "sell",
            "type": "limit",
            "time_in_force": TIF,
            "order_class": "oco",
            "take_profit": {"limit_price": str(tp)},
            "stop_loss": {"stop_price": str(sl)},
        }
        r = requests.post(f"{B}/v2/orders", headers=H, json=payload)
        if r.status_code in (200, 201):
            created += 1
            print(f"[OK] {sym} OCO placed (qty={qty}, tp={tp}, sl={sl})")
        else:
            print(f"[ERR] {sym} {r.status_code}: {r.text[:300]}")
        time.sleep(0.15)
    except Exception as e:
        print(f"[EXC] {p.get('symbol','?')}: {e}")

print(f"Done. Created {created} OCO protection orders.")
