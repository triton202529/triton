import sys
import os

# Make sure root folder is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpaca_client import get_account, get_positions


def main():
    print("🔌 Connecting to live Alpaca account...\n")

    try:
        account = get_account()
        print("💼 ACCOUNT INFO:")
        print(account)

        print("\n📊 CURRENT POSITIONS:")
        positions = get_positions()
        if positions:
            for p in positions:
                print(f"{p['symbol']}: {p['qty']} shares at ${p['avg_entry_price']}")
        else:
            print("No open positions.")
    except Exception as e:
        print("❌ Failed to connect:", e)


if __name__ == "__main__":
    main()
