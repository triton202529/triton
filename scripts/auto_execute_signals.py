#!/usr/bin/env python3
"""
Execute trade signals from CSV using Alpaca (alpaca-py).

Improvements:
- Recomputes buying power before each BUY
- Supports fractional/notional BUYs via --dollars-per-trade (falls back to whole shares if not fractionable)
- SELL logic:
    * Cancels any open orders for the symbol before closing
    * Fractional SELLs use TimeInForce.DAY (required by Alpaca) and floor quantity to avoid 'requested > available'
    * Whole-share SELLs use GTC
- Processes SELLs first, then BUYs (frees buying power early)
- New knobs:
    * --max-buys N           : cap number of BUYs per run
    * --min-alloc $X         : skip BUYs below this allocation
    * --autoscale            : spread buying power across remaining BUYs to avoid running out mid-run
    * --cancel-all-open      : cancel all open orders at start of run
    * --force-close-all      : close ALL positions before processing signals (danger!)
    * --close-tickers T1,T2  : force-close specific tickers before processing signals
- Maps Yahoo class shares (e.g. BRK-B) to Alpaca’s dot form (BRK.B)
- Skips index symbols (^GSPC, ^DJI, ^IXIC, ^VIX)
- QoL: ALPACA_JSON env var can override config path
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List
from decimal import Decimal, ROUND_DOWN

import pandas as pd
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest

# ---- risk_control import ----
try:
    from risk_control import risk_check  # noqa: F401
except ImportError:
    from scripts.risk_control import risk_check  # type: ignore  # noqa: F401

DEFAULT_SIGNALS = Path("data/predictions/signals.csv")
DEFAULT_LOG = Path("data/results/executed_trades.csv")
ALPACA_JSON = Path(os.getenv("ALPACA_JSON", "config/alpaca.json"))
DEFAULT_TRADE_PCT = 0.05  # 5% of current buying power per trade


def read_json_utf8sig(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def ensure_log_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        cols = ["timestamp", "ticker", "action", "quantity", "price", "status", "note"]
        pd.DataFrame(columns=cols).to_csv(log_path, index=False)


def latest_price_yf(symbol: str) -> Optional[float]:
    """Get a recent price using yfinance (Yahoo symbol)."""
    try:
        ti = yf.Ticker(symbol)
        p = ti.fast_info.get("last_price")
        if p and p > 0:
            return float(p)
    except Exception:
        pass
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1m", auto_adjust=True)
        if not df.empty:
            return float(df["Close"].to_numpy()[-1])
    except Exception:
        pass
    return None


def coerce_side(signal: str) -> str:
    s = (signal or "").strip().upper()
    if s in ("BUY", "B", "LONG"):
        return "BUY"
    if s in ("SELL", "S", "SHORT"):
        return "SELL"
    return "HOLD"


def map_symbols(ticker: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (alpaca_symbol, reason_if_skipped)."""
    t = str(ticker).strip().upper()
    if t.startswith("^"):
        return None, "index symbol"
    if "-" in t:
        return t.replace("-", "."), None
    return t, None


def cancel_open_orders_for_symbol(client: TradingClient, symbol: str) -> int:
    """Cancel any open orders for `symbol`. Returns count canceled."""
    orders = []
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        orders = client.get_orders(filter=req)
    except Exception:
        # Fallback: fetch all opens and filter locally
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = [o for o in client.get_orders(filter=req) if getattr(o, "symbol", None) == symbol]
        except Exception:
            orders = []
    canceled = 0
    for o in orders:
        try:
            client.cancel_order_by_id(o.id)
            canceled += 1
        except Exception:
            pass
    return canceled


def cancel_all_open_orders(client: TradingClient) -> int:
    """Cancel ALL open orders across the account. Returns count canceled (best-effort)."""
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = client.get_orders(filter=req)
    except Exception:
        orders = []
    canceled = 0
    for o in orders:
        try:
            client.cancel_order_by_id(o.id)
            canceled += 1
        except Exception:
            pass
    return canceled


def is_fractional_qty(qty) -> bool:
    """Accept str/Decimal/float; treat nearly-integers as integers."""
    d = Decimal(str(qty))
    return (d % 1) != 0


def fmt_frac_qty_floor(qty_str: str, places: int = 6) -> str:
    """
    Floor (never round up) to `places` decimals for safety.
    Keeps within Alpaca fractional precision while avoiding 'requested > available'.
    """
    d = Decimal(qty_str)
    q = Decimal("1").scaleb(-places)   # 10^-places
    return str(d.quantize(q, rounding=ROUND_DOWN))


def close_position_safely(
    client: TradingClient,
    alpaca_symbol: str,
    disp: str,
    yahoo_symbol: str,
    logs: List[list],
    timestamp: str,
    dry_run: bool = False,
) -> bool:
    """Close position in `alpaca_symbol` using fractional-safe logic. Returns True if an action was taken."""
    if dry_run:
        try:
            pos = client.get_open_position(alpaca_symbol)
            qty = float(getattr(pos, "qty", 0.0))
            if qty > 0:
                logs.append([timestamp, yahoo_symbol, "SELL", qty, None, "SIMULATED", "Would CLOSE position"])
                print(f"📝 Simulated SELL (close position): {qty:g} x {disp}")
                return True
        except Exception:
            pass
        print(f"⏸️ No position to SELL in {disp}")
        logs.append([timestamp, yahoo_symbol, "SELL", 0, None, "SKIPPED", "No position"])
        return False

    # Live close
    canceled = cancel_open_orders_for_symbol(client, alpaca_symbol)
    if canceled:
        print(f"🧹 Canceled {canceled} open order(s) for {disp} before closing")
    try:
        pos = client.get_open_position(alpaca_symbol)
    except Exception:
        print(f"⏸️ No position to SELL in {disp}")
        logs.append([timestamp, yahoo_symbol, "SELL", 0, None, "SKIPPED", "No position"])
        return False

    raw_qty = getattr(pos, "qty", "0")
    dqty = Decimal(str(raw_qty))
    if dqty <= 0:
        print(f"⏸️ No position to SELL in {disp}")
        logs.append([timestamp, yahoo_symbol, "SELL", 0, None, "SKIPPED", "No position"])
        return False

    try:
        if is_fractional_qty(dqty):
            safe_qty = fmt_frac_qty_floor(str(raw_qty), places=6)
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                side=OrderSide.SELL,
                qty=safe_qty,                   # string fractional
                time_in_force=TimeInForce.DAY   # fractional must be DAY
            )
            resp = client.submit_order(order)
            logs.append([timestamp, yahoo_symbol, "SELL", float(safe_qty), None, "EXECUTED",
                         f"Order ID: {resp.id} (fractional DAY)"])
            print(f"✅ Closed {safe_qty} x {disp} (fractional, DAY)")
        else:
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                side=OrderSide.SELL,
                qty=int(dqty),
                time_in_force=TimeInForce.GTC
            )
            resp = client.submit_order(order)
            logs.append([timestamp, yahoo_symbol, "SELL", int(dqty), None, "EXECUTED", f"Order ID: {resp.id}"])
            print(f"✅ Closed {int(dqty)} x {disp} (GTC)")
        return True
    except Exception as e:
        logs.append([timestamp, yahoo_symbol, "SELL", 0, None, "FAILED", str(e)])
        print(f"❌ Error closing position for {disp}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Execute signals file via Alpaca (alpaca-py).")
    ap.add_argument("--signals", default=str(DEFAULT_SIGNALS), help=f"Signals CSV (default: {DEFAULT_SIGNALS})")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help=f"Trade log CSV (default: {DEFAULT_LOG})")
    ap.add_argument("--trade-pct", type=float, default=DEFAULT_TRADE_PCT,
                    help="Allocation %% of current buying power per BUY (ignored if --dollars-per-trade > 0)")
    ap.add_argument("--dollars-per-trade", type=float, default=0.0,
                    help="If > 0, spend this dollar amount per BUY using fractional/notional orders when supported")
    ap.add_argument("--max-buys", type=int, default=None,
                    help="Max number of BUY orders to place in this run (default: no cap)")
    ap.add_argument("--min-alloc", type=float, default=25.0,
                    help="Skip BUYs if computed allocation is below this dollar amount (default: 25)")
    ap.add_argument("--autoscale", action="store_true",
                    help="Scale per-trade allocation to buying_power / remaining_BUYs so you don't run out mid-run")
    ap.add_argument("--cancel-all-open", action="store_true",
                    help="Cancel ALL open orders at start of run")
    ap.add_argument("--force-close-all", action="store_true",
                    help="Close ALL positions before processing signals (dangerous; ignores signals).")
    ap.add_argument("--close-tickers", type=str, default="",
                    help="Comma-separated tickers to force-close before processing signals (e.g. 'AAPL,MSFT,SPY')")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", help="Force paper mode")
    mode.add_argument("--live", action="store_true", help="Force live mode")
    ap.add_argument("--dry-run", action="store_true", help="Preview only; do not place orders")
    args = ap.parse_args()

    signals_path = Path(args.signals)
    log_path = Path(args.log)

    conf = read_json_utf8sig(ALPACA_JSON)
    key_id = conf.get("key_id")
    secret_key = conf.get("secret_key")
    if not key_id or not secret_key:
        print(f"❌ Missing key_id/secret_key in {ALPACA_JSON}", file=sys.stderr)
        return 2

    paper = bool(conf.get("paper", True))
    if args.paper:
        paper = True
    if args.live:
        paper = False

    client = TradingClient(key_id, secret_key, paper=paper)
    acct = client.get_account()
    print(f"Repo root: {Path.cwd()}")
    print("\n🚀 Starting trade execution..." if not args.dry_run else "\n🧪 Starting trade simulation...")
    print(f"🔌 Connected: status={acct.status} cash=${float(acct.cash):,.2f} equity=${float(acct.equity):,.2f} mode={'paper' if paper else 'live'}")
    print(f"💰 Buying power: ${float(acct.buying_power):,.2f}\n")

    ensure_log_file(log_path)
    logs: List[list] = []

    # ---------- Pre-run hygiene ----------
    timestamp0 = datetime.now(timezone.utc).isoformat()

    if args.cancel_all_open:
        canceled = cancel_all_open_orders(client)
        print(f"🗑️  Canceled {canceled} open order(s) globally")

    # Force-close all positions (before signals)
    if args.force_close_all:
        if args.dry_run:
            try:
                positions = client.get_all_positions()
                total = len(positions)
            except Exception:
                total = 0
            print(f"🧪 Would close ALL positions first (found ~{total})")
            for p in getattr(positions, "__iter__", lambda: [])():
                ysym = p.symbol.replace(".", "-")  # for display, inverse of map
                close_position_safely(client, p.symbol, p.symbol, ysym, logs, timestamp0, dry_run=True)
        else:
            try:
                positions = client.get_all_positions()
            except Exception:
                positions = []
            print(f"⚠️  Closing ALL positions before processing signals (count={len(positions)})")
            for p in positions:
                ysym = p.symbol.replace(".", "-")
                close_position_safely(client, p.symbol, p.symbol, ysym, logs, timestamp0, dry_run=False)

    # Force-close explicit tickers list (before signals)
    if args.close_tickers.strip():
        to_close = [t.strip() for t in args.close_tickers.split(",") if t.strip()]
        if to_close:
            print(f"🔒 Force-closing tickers before signals: {', '.join(to_close)}")
            for ysym in to_close:
                alp, skip = map_symbols(ysym)
                if alp is None:
                    print(f"⏭️  Skipping {ysym} ({skip})")
                    continue
                disp = ysym if ysym.upper() == alp else f"{ysym} (Alpaca: {alp})"
                ts = datetime.now(timezone.utc).isoformat()
                close_position_safely(client, alp, disp, ysym, logs, ts, dry_run=args.dry_run)

    # ---------- Load signals ----------
    if not signals_path.exists():
        print(f"❌ Signals file not found: {signals_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(signals_path, encoding="utf-8-sig")
    need_cols = {"ticker", "signal"}
    if not need_cols.issubset({c.lower() for c in df.columns}):
        raise ValueError("❌ 'ticker' and 'signal' columns are required in the signals file.")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["ticker", "signal"] + [c for c in df.columns if c not in ("ticker", "signal")]]

    # Use only the latest signal per ticker
    latest_signals = df.groupby("ticker", as_index=False).last()

    # SELLs first (0), HOLDs (1), BUYs last (2)
    latest_signals["_side_order"] = latest_signals["signal"].apply(
        lambda s: 0 if coerce_side(s) == "SELL" else (1 if coerce_side(s) == "HOLD" else 2)
    )
    latest_signals = latest_signals.sort_values("_side_order").drop(columns="_side_order")

    # Count BUYs for autoscale math
    total_buys = int((latest_signals["signal"].apply(coerce_side) == "BUY").sum())
    buys_placed = 0

    trade_pct = float(args.trade_pct)
    min_alloc = max(1.00, float(args.min_alloc))

    for _, row in latest_signals.iterrows():
        yahoo_symbol = str(row["ticker"]).strip()
        side_text = coerce_side(row["signal"])
        alpaca_symbol, skip_reason = map_symbols(yahoo_symbol)
        timestamp = datetime.now(timezone.utc).isoformat()

        if alpaca_symbol is None:
            print(f"⏭️  Skipping non-tradeable symbol: {yahoo_symbol} ({skip_reason})")
            continue

        disp = yahoo_symbol if yahoo_symbol.upper() == alpaca_symbol else f"{yahoo_symbol} (Alpaca: {alpaca_symbol})"
        print(f"📊 {disp}: Signal = {side_text}")

        try:
            if side_text == "HOLD":
                print(f"⏸️ HOLD for {disp} — no action")
                logs.append([timestamp, yahoo_symbol, "HOLD", 0, None, "SKIPPED", "No action"])
                continue

            # Risk check
            try:
                ok = risk_check(yahoo_symbol, side_text, client)
            except TypeError:
                ok = risk_check(yahoo_symbol, side_text)  # type: ignore
            if not ok:
                print(f"⚠️ Trade blocked by risk control: {disp} {side_text}")
                logs.append([timestamp, yahoo_symbol, side_text, 0, None, "BLOCKED", "Blocked by risk control"])
                continue

            if side_text == "BUY":
                # Optional cap on number of BUYs per run
                if args.max_buys is not None and buys_placed >= args.max_buys:
                    logs.append([timestamp, yahoo_symbol, "BUY", 0, None, "SKIPPED", "max-buys reached"])
                    print(f"⏸️ Reached --max-buys ({args.max_buys}); skipping BUY in {disp}")
                    continue

                # Refresh current buying power
                acct = client.get_account()
                buying_power_now = float(acct.buying_power)

                # Base allocation (from flags or pct)
                base_alloc = args.dollars_per_trade if args.dollars_per_trade > 0 else (trade_pct * buying_power_now)

                # Autoscale so you don't exhaust BP early
                if args.autoscale and total_buys:
                    remaining_buys = max(total_buys - buys_placed, 1)
                    alloc = min(base_alloc, buying_power_now / remaining_buys)
                else:
                    alloc = min(base_alloc, buying_power_now)

                if alloc < min_alloc:
                    logs.append([timestamp, yahoo_symbol, "BUY", 0, None, "SKIPPED",
                                 f"Allocation ${alloc:.2f} below --min-alloc ${min_alloc:.2f}"])
                    print(f"❌ Allocation ${alloc:.2f} below --min-alloc ${min_alloc:.2f} for {disp}")
                    continue

                # Get latest price for whole-share fallback sizing
                price = latest_price_yf(yahoo_symbol)

                # Check if asset supports fractionals
                try:
                    asset = client.get_asset(alpaca_symbol)
                    fractionable = bool(getattr(asset, "fractionable", False))
                except Exception:
                    fractionable = False  # fail safe

                if fractionable and args.dollars_per_trade > 0:
                    # Fractional/notional path -> MUST be DAY per Alpaca
                    notional = round(alloc, 2)
                    if args.dry_run:
                        logs.append([timestamp, yahoo_symbol, "BUY", notional, price, "SIMULATED", "Would BUY notional (DAY)"])
                        print(f"📝 Simulated BUY: ~${notional:.2f} of {disp} (fractional, DAY)")
                    else:
                        order = MarketOrderRequest(
                            symbol=alpaca_symbol,
                            side=OrderSide.BUY,
                            notional=notional,
                            time_in_force=TimeInForce.DAY,  # critical for fractional
                        )
                        resp = client.submit_order(order)
                        logs.append([timestamp, yahoo_symbol, "BUY", notional, price, "EXECUTED", f"Order ID: {resp.id} (notional DAY)"])
                        print(f"✅ Bought ~${notional:.2f} of {disp} (fractional, DAY)")
                        buys_placed += 1
                else:
                    # Whole-share fallback (or when --dollars-per-trade == 0)
                    if price is None or price <= 0:
                        logs.append([timestamp, yahoo_symbol, "BUY", 0, None, "FAILED", "No price available"])
                        print(f"❌ Could not retrieve price for {disp}")
                        continue
                    qty = int(alloc // price)
                    if qty <= 0:
                        logs.append([timestamp, yahoo_symbol, "BUY", 0, price, "SKIPPED", "Insufficient buying power for 1 share"])
                        print(f"❌ Not enough to buy even 1 share of {disp} (price ~${price:.2f})")
                        continue
                    if args.dry_run:
                        logs.append([timestamp, yahoo_symbol, "BUY", qty, price, "SIMULATED", "Would BUY qty (GTC)"])
                        print(f"📝 Simulated BUY: {qty} x {disp} at ~${price:.2f} (GTC)")
                        buys_placed += 1
                    else:
                        order = MarketOrderRequest(
                            symbol=alpaca_symbol,
                            side=OrderSide.BUY,
                            qty=qty,
                            time_in_force=TimeInForce.GTC,  # whole-share orders can be GTC
                        )
                        resp = client.submit_order(order)
                        logs.append([timestamp, yahoo_symbol, "BUY", qty, price, "EXECUTED", f"Order ID: {resp.id}"])
                        print(f"✅ Bought {qty} x {disp} at ~${price:.2f} (GTC)")
                        buys_placed += 1

            elif side_text == "SELL":
                # Use the same safe-precision close routine
                close_position_safely(client, alpaca_symbol, disp, yahoo_symbol, logs, timestamp, dry_run=args.dry_run)

        except Exception as e:
            logs.append([timestamp, yahoo_symbol, side_text, 0, None, "FAILED", str(e)])
            print(f"❌ Error processing {disp}: {e}")

    with log_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(logs)
    print("\n📄 Trade log saved ->", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
