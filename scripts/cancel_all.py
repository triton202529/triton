import os, requests

H = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET"),
}
B = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets").rstrip("/")
r = requests.delete(f"{B}/v2/orders", headers=H)
print("Cancel all status:", r.status_code, r.text[:200])
