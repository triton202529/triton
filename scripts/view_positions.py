import os
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

api = REST(API_KEY, API_SECRET, BASE_URL)

print("📡 Fetching current positions...\n")

try:
    positions = api.list_positions()

    if not positions:
        print("📭 No current positions.")
    else:
        data = []
        for pos in positions:
            data.append({
                "Ticker": pos.symbol,
                "Qty": pos.qty,
                "Avg Price": float(pos.avg_entry_price),
                "Current Price": float(pos.current_price),
                "Unrealized P&L": float(pos.unrealized_pl),
                "Market Value": float(pos.market_value)
            })

        df = pd.DataFrame(data)
        df["Unrealized P&L"] = df["Unrealized P&L"].round(2)
        df["Market Value"] = df["Market Value"].round(2)
        print(df.to_string(index=False))

except Exception as e:
    print(f"❌ Error fetching positions: {e}")
