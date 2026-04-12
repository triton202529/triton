# scripts/oco_watch.py
# Watches for new BUY fills and auto-attaches OCO exits (TP/SL).
# Usage:
#   python scripts/oco_watch.py --tp 0.08 --sl 0.05 --qty 1 --minutes 60
#   python scripts/oco_watch.py --symbols AMZN,NVDA --tp 0.08 --sl 0.05 --qty 1

import argparse, os, time, datetime as dt, requests


def base():
    return os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def H():
    k = os.getenv("APCA_API_KEY_ID")
    s = os.getenv("APCA_API_SECRET_KEY")
    if not k or not s:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def fills_since(ts_iso: str):
    r = requests.get(
        f"{base()}/v2/account/activities/FILL",
        params={"after": ts_iso, "direction": "asc"},
        headers=H(),
        timeout=30,
    )
    r.raise_for_status()
    acts = r.json()
    return acts if isinstance(acts, list) else [acts]


def pos(sym):
    r = requests.get(f"{base()}/v2/positions/{sym}", headers=H(), timeout=15)
    return r.json() if r.status_code == 200 else None


def has_open_sell(sym):
    r = requests.get(
        f"{base()}/v2/orders",
        params={"status": "open", "nested": "true", "symbols": sym, "limit": 500},
        headers=H(),
        timeout=30,
    )
    r.raise_for_status()
    for o in r.json():
        if str(o.get("side")).lower() == "sell":
            return True
        for leg in o.get("legs") or []:
            if str(leg.get("side")).lower() == "sell":
                return True
    return False


def attach_oco(sym, qty, tp_pct, sl_pct, tif="gtc"):
    p = pos(sym)
    if not p:
        return {"symbol": sym, "attached": False, "reason": "no position"}
    if has_open_sell(sym):
        return {"symbol": sym, "attached": False, "reason": "sell leg already open"}
    avg = float(p["avg_entry_price"])
    tp = round(avg * (1 + tp_pct), 2)
    sl = round(avg * (1 - sl_pct), 2)
    payload = {
        "symbol": sym,
        "qty": str(int(qty)),
        "side": "sell",
        "type": "limit",
        "time_in_force": tif,
        "order_class": "oco",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
    }
    r = requests.post(f"{base()}/v2/orders", headers=H(), json=payload, timeout=30)
    ok = r.status_code in (200, 201)
    return {
        "symbol": sym,
        "attached": ok,
        "status": r.status_code,
        "tp": tp,
        "sl": sl,
        "resp": r.text[:200],
    }


def main():
    ap = argparse.ArgumentParser(description="Watch for BUY fills and attach OCO exits.")
    ap.add_argument("--tp", type=float, default=0.08)
    ap.add_argument("--sl", type=float, default=0.05)
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--tif", default="gtc")
    ap.add_argument("--minutes", type=int, default=60, help="Watch window")
    ap.add_argument("--symbols", default="", help="Optional whitelist: e.g. AMZN,NVDA")
    args = ap.parse_args()

    whitelist = set(s.strip().upper() for s in args.symbols.split(",") if s.strip()) or None
    start = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    print(
        f"[watch] start={start} window={args.minutes}m tp={args.tp} sl={args.sl} qty={args.qty}"
        + (f" symbols={','.join(sorted(whitelist))}" if whitelist else "")
    )

    seen = set()
    end = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=args.minutes)
    while dt.datetime.now(dt.timezone.utc) < end:
        try:
            for a in fills_since(start):
                if str(a.get("side")).lower() != "buy":
                    continue
                sym = (a.get("symbol") or "").upper()
                if not sym:
                    continue
                if whitelist and sym not in whitelist:
                    continue
                key = (a.get("id"), sym)
                if key in seen:
                    continue
                seen.add(key)
                res = attach_oco(sym, args.qty, args.tp, args.sl, args.tif)
                print(res)
        except Exception as e:
            print("warn:", repr(e))
        time.sleep(5)


if __name__ == "__main__":
    main()
