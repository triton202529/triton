# services/broker_alpaca.py
import os, time, json, logging
import requests
from dataclasses import dataclass
from typing import Optional, Dict

ALPACA_PAPER = "https://paper-api.alpaca.markets"
ALPACA_LIVE  = "https://api.alpaca.markets"

@dataclass
class OrderResult:
    ok: bool
    id: Optional[str]
    status: str
    raw: Dict

class AlpacaBroker:
    def __init__(self, key: str, secret: str, env: str = "paper"):
        base = ALPACA_PAPER if env.lower() == "paper" else ALPACA_LIVE
        self.base = base
        self.trading = f"{base}/v2"
        self.headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
        }

    def get_account(self) -> Dict:
        r = requests.get(f"{self.trading}/account", headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> Dict:
        r = requests.get(f"{self.trading}/positions", headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def submit_order(self, symbol: str, qty: int, side: str, tif="day", type_="market") -> OrderResult:
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side.lower(),         # "buy" | "sell"
            "type": type_.lower(),        # "market"
            "time_in_force": tif.lower(), # "day"
        }
        try:
            r = requests.post(f"{self.trading}/orders", headers=self.headers, data=json.dumps(payload), timeout=20)
            if r.status_code >= 400:
                return OrderResult(False, None, f"HTTP {r.status_code}", {"error": r.text})
            data = r.json()
            return OrderResult(True, data.get("id"), data.get("status", ""), data)
        except Exception as e:
            logging.exception("submit_order failed")
            return OrderResult(False, None, "exception", {"error": str(e)})
