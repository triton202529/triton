import os, sys, json


def show(title, f):
    try:
        v = f()
        print(f"[{title}] OK -> {v}")
    except Exception as e:
        print(f"[{title}] ERR -> {e.__class__.__name__}: {e}")


base = os.environ.get("APCA_API_BASE_URL", "").rstrip("/")
key = os.environ.get("APCA_API_KEY_ID")
sec = os.environ.get("APCA_API_SECRET_KEY")
print("ENV base:", base, "| key set:", bool(key), "| secret set:", bool(sec))


def via_requests():
    import requests

    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    r = requests.get(f"{base}/v2/account", headers=h, timeout=10)
    r.raise_for_status()
    return r.json().get("buying_power")


def via_alpaca_trade_api():
    import alpaca_trade_api as tradeapi

    api = tradeapi.REST(key_id=key, secret_key=sec, base_url=base)
    acct = api.get_account()
    return acct.buying_power


def via_alpaca_py():
    from alpaca.trading.client import TradingClient

    client = TradingClient(key, sec, paper=("paper-api" in base))
    acct = client.get_account()
    return acct.buying_power


show("requests", via_requests)
show("alpaca_trade_api", via_alpaca_trade_api)
show("alpaca-py", via_alpaca_py)
