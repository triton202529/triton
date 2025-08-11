# services/place_live_orders.py
import os, csv, logging
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv

from services.broker_alpaca import AlpacaBroker

RESULTS_DIR = "data/results"
LOG_DIR = "logs"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

load_dotenv(override=True)

# --- config / env ---
LIVE_MODE = os.getenv("TRITON_LIVE_MODE", "false").lower() == "true"
BROKER = os.getenv("TRITON_BROKER", "alpaca").lower()
ALPACA_ENV = os.getenv("ALPACA_ENV", "paper")
ALPACA_KEY = os.getenv("ALPACA_KEY_ID", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

MAX_POSITION_PCT   = float(os.getenv("MAX_POSITION_PCT", "0.10"))
MAX_DAILY_NOTIONAL = float(os.getenv("MAX_DAILY_NOTIONAL", "20000"))
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "0.0"))
DO_NOT_TRADE       = {t.strip().upper() for t in os.getenv("DO_NOT_TRADE", "").split(",") if t.strip()}
COOLDOWN_MIN       = int(os.getenv("COOLDOWN_MIN", "5"))

SIGNALS_FILE = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
EXEC_LOG     = os.path.join(RESULTS_DIR, "live_orders.csv")
SESSION_TAG  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"live_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def load_signals() -> pd.DataFrame:
    if not os.path.exists(SIGNALS_FILE):
        raise FileNotFoundError(f"Missing {SIGNALS_FILE}")
    df = pd.read_csv(SIGNALS_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # latest day only
    latest = df["date"].max()
    df = df[df["date"] == latest].copy()
    # keep only actionable
    if "signal" in df.columns:
        df = df[df["signal"].isin(["BUY","SELL"])]
    else:
        df = pd.DataFrame(columns=["ticker","close","signal","date"])
    # basic sanity
    for c in ["ticker","close","signal"]:
        if c not in df.columns: 
            df[c] = None
    return df

def init_broker():
    if BROKER == "alpaca":
        return AlpacaBroker(ALPACA_KEY, ALPACA_SECRET, ALPACA_ENV)
    raise ValueError(f"Unsupported broker: {BROKER}")

def load_exec_log() -> pd.DataFrame:
    if not os.path.exists(EXEC_LOG):
        return pd.DataFrame(columns=["timestamp","session","ticker","side","qty","price","status","broker_order_id","note"])
    return pd.read_csv(EXEC_LOG)

def append_exec_rows(rows: list[dict]):
    """Buffered write of many rows -> avoids concat warning spam."""
    if not rows:
        return
    file_exists = os.path.exists(EXEC_LOG)
    with open(EXEC_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","session","ticker","side","qty","price","status","broker_order_id","note"])
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def within_cooldown(exec_df: pd.DataFrame, ticker: str) -> bool:
    if exec_df.empty: 
        return False
    sub = exec_df[exec_df["ticker"] == ticker]
    if sub.empty: 
        return False
    # Parse timestamps as UTC-aware and compare to now(UTC)
    last_ts = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce").max()
    if pd.isna(last_ts):
        return False
    seconds = (datetime.now(timezone.utc) - last_ts.to_pydatetime()).total_seconds()
    return seconds < COOLDOWN_MIN * 60

def main():
    print(f"{'🚀 LIVE' if LIVE_MODE else '🧪 DRY-RUN'} | Broker={BROKER} ({ALPACA_ENV}) | Session={SESSION_TAG}")
    signals = load_signals()
    if signals.empty:
        print("⚠️ No actionable signals today.")
        return

    # connect broker + account
    broker = init_broker()
    acct = {}
    try:
        acct = broker.get_account()
    except Exception as e:
        logging.exception("Account fetch failed")
        if LIVE_MODE:
            print(f"❌ Broker account fetch failed: {e}")
            return
        else:
            print("ℹ️ DRY-RUN: continuing without account info.")

    buying_power = float(acct.get("buying_power", 0)) if acct else 0.0
    equity = float(acct.get("equity", 100000)) if acct else 100000.0

    per_position_budget = equity * MAX_POSITION_PCT
    daily_notional_left = MAX_DAILY_NOTIONAL

    # load existing exec log into memory once
    exec_df = load_exec_log()

    # buffer rows then append once
    out_rows = []
    acted = 0

    for _, r in signals.iterrows():
        tkr = str(r.get("ticker", "")).upper()
        if not tkr or tkr in DO_NOT_TRADE:
            continue

        conf = r.get("confidence", 0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        if abs(conf) < CONF_THRESHOLD:
            continue

        try:
            price = float(r.get("close", 0) or 0)
        except Exception:
            price = 0.0
        if price <= 0:
            continue

        sig = str(r.get("signal","")).upper()
        if sig not in ("BUY","SELL"):
            continue
        side = "buy" if sig == "BUY" else "sell"

        # cooldown / idempotency
        if within_cooldown(exec_df, tkr):
            continue

        # sizing (simple)
        qty = int(max(0, per_position_budget // price))
        if side == "sell" and qty == 0:
            qty = 1  # optional: allow a small reduction attempt

        if qty <= 0:
            continue

        notional = qty * price
        if notional > daily_notional_left:
            continue

        note = "dry-run"
        broker_id = ""
        status = "SKIPPED"

        if LIVE_MODE:
            try:
                res = broker.submit_order(symbol=tkr, qty=qty, side=side)
                status = "SUBMITTED" if getattr(res, "ok", False) else f"ERROR:{getattr(res, 'status', 'UNKNOWN')}"
                broker_id = getattr(res, "id", "") or ""
                note = "" if getattr(res, "ok", False) else str(getattr(res, "raw", ""))[:180]
            except Exception as e:
                status = "ERROR"
                note = str(e)[:180]
        else:
            status = "DRY-RUN"

        # log row (UTC-aware timestamp)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session": SESSION_TAG,
            "ticker": tkr,
            "side": side.upper(),
            "qty": qty,
            "price": price,
            "status": status,
            "broker_order_id": broker_id,
            "note": note,
        }
        out_rows.append(row)

        # update in-memory exec_df for cooldown checks in same run
        exec_df = pd.concat([exec_df, pd.DataFrame([row])], ignore_index=True)

        daily_notional_left -= notional
        acted += 1

    # single append to file
    append_exec_rows(out_rows)

    print(f"✅ Done. Actions: {acted} | Mode: {'LIVE' if LIVE_MODE else 'DRY-RUN'}")
    print(f"📄 Log: {EXEC_LOG}")

if __name__ == "__main__":
    main()
