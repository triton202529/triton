# scripts/attach_oco.py
import argparse, os, requests


def alpaca_base():
    return os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def alpaca_headers():
    return {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
    }


def has_open_sell(b, h, sym):
    r = requests.get(
        f"{b}/v2/orders",
        params={"status": "open", "nested": "true", "symbols": sym, "limit": 200},
        headers=h,
        timeout=15,
    )
    r.raise_for_status()
    for o in r.json():
        if o.get("side") == "sell":
            return True
    return False


def attach_oco(sym, qty, tp_pct, sl_pct, tif="gtc"):
    b, h = alpaca_base(), alpaca_headers()
    # fetch position for avg price
    r = requests.get(f"{b}/v2/positions/{sym}", headers=h, timeout=10)
    if r.status_code != 200:
        return {"symbol": sym, "attached": False, "reason": "no position"}
    pos = r.json()
    avg = float(pos["avg_entry_price"])
    tp = round(avg * (1 + tp_pct), 2)
    sl = round(avg * (1 - sl_pct), 2)

    # skip if a sell leg is already open
    if has_open_sell(b, h, sym):
        return {"symbol": sym, "attached": False, "reason": "sell leg already open"}

    payload = {
        "symbol": sym,
        "qty": str(int(qty)),
        "side": "sell",
        "type": "limit",  # required for OCO parent
        "time_in_force": tif,
        "order_class": "oco",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
    }
    rr = requests.post(f"{b}/v2/orders", headers=h, json=payload, timeout=15)
    ok = rr.status_code in (200, 201)
    return {"symbol": sym, "attached": ok, "status": rr.status_code, "resp": rr.text[:200]}


def main():
    p = argparse.ArgumentParser(description="Attach OCO exits to open positions.")
    p.add_argument(
        "--symbols", default="", help="Comma-separated symbols (leave empty = all positions)"
    )
    p.add_argument("--qty", type=int, default=1, help="Exit quantity per symbol")
    p.add_argument("--tp", type=float, default=0.08, help="Take-profit pct (e.g. 0.08)")
    p.add_argument("--sl", type=float, default=0.05, help="Stop-loss pct (e.g. 0.05)")
    p.add_argument("--tif", default="gtc", help="time_in_force (default gtc)")
    args = p.parse_args()

    b, h = alpaca_base(), alpaca_headers()

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        # load all current position symbols
        r = requests.get(f"{b}/v2/positions", headers=h, timeout=15)
        r.raise_for_status()
        syms = [p["symbol"].upper() for p in r.json()]

    results = []
    for s in syms:
        res = attach_oco(s, args.qty, args.tp, args.sl, args.tif)
        results.append(res)
        print(res)
    # no JSON file write; prints line-by-line dicts


if __name__ == "__main__":
    main()
