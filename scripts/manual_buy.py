import os
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

# Connect to Alpaca
api = REST(API_KEY, API_SECRET, BASE_URL)

# Prompt for ticker and quantity
symbol = input("Enter the ticker symbol (e.g., AAPL): ").upper()
qty = int(input("Enter the number of shares to buy: "))

# Submit market order
try:
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side="buy",
        type="market",
        time_in_force="gtc"  # Using string instead of TimeInForce.GTC
    )
    print(f"✅ Order placed: BUY {qty} x {symbol}")
    print(f"📄 Order ID: {order.id}")
except Exception as e:
    print(f"❌ Failed to place order: {e}")
