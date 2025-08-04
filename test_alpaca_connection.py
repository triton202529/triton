from alpaca_trade_api.rest import REST, APIError
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_API_SECRET")
base_url = os.getenv("ALPACA_BASE_URL")

try:
    api = REST(api_key, api_secret, base_url)
    account = api.get_account()
    print("✅ Connected to Alpaca!")
    print("🆔 ID:", account.id)
    print("💰 Cash:", account.cash)
    print("📊 Status:", account.status)
except APIError as e:
    print("❌ Connection failed:", e)
    print("🔍 Full Error Body:", e._error)
