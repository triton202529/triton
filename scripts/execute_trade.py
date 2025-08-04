import os
import requests

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

def submit_order(symbol, qty, side, type="market", time_in_force="gtc"):
    url = f"{BASE_URL}/v2/orders"
    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side.lower(),
        "type": type,
        "time_in_force": time_in_force
    }

    response = requests.post(url, json=order, headers=HEADERS)
    if response.status_code == 200 or response.status_code == 201:
        return {"status": "success", "data": response.json()}
    else:
        return {"status": "error", "message": response.text}
