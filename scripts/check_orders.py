import os, requests, pprint

H = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET"),
}
B = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets").rstrip("/")
orders = requests.get(f"{B}/v2/orders", headers=H, params={"status": "open", "nested": True}).json()
print(f"Open orders: {len(orders)}")
pp = pprint.PrettyPrinter()
pp.pprint(
    [
        {
            k: o.get(k)
            for k in (
                "symbol",
                "side",
                "type",
                "time_in_force",
                "extended_hours",
                "order_class",
                "legs",
            )
        }
        for o in orders
    ][:10]
)
