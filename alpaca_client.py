import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

def get_account():
    url = f"{BASE_URL}/v2/account"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def get_positions():
    url = f"{BASE_URL}/v2/positions"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def submit_order(symbol, qty, side, type="market", time_in_force="gtc"):
    url = f"{BASE_URL}/v2/orders"
    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }
    response = requests.post(url, headers=HEADERS, json=order)
    return response.json()

def cancel_all_orders():
    url = f"{BASE_URL}/v2/orders"
    response = requests.delete(url, headers=HEADERS)
    return response.status_code
