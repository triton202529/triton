# services/place_live_orders.py
"""
Place live/paper orders with guardrails, weights-sizing, and duplicate / BP / risk checks.

What's new (2025-09-01):
- Symbol filters: --exclude / --only (plus DEFAULT_TICKERS_EXCLUDE you can edit)
- Per-run liquidation cap: --max-sell-fraction (e.g., 0.33 caps SELL to 33% of current position)
- Batch limiter: --max-orders to stop after N submissions (or dry-run logs)
- More explicit skip logging (exclude-skip, only-skip, sell-cap-skip)
- Daily notional cap now applies to BUYs only (SELLs are allowed after the buy budget is exhausted)
- NEW: --sell-only convenience flag (= --reconcile --reconcile-mode sells_only + safe SELL reductions)
- NEW: SELL availability reduction is ON by default; use --no-reduce-sell-to-available to opt out
- NEW: --weights-file lets you choose a specific weights CSV (defaults to data/results/weights.csv)
- FIX: --bp-buffer-pct now treats 5 as 5% (not 500%); same normalization applied to risk.json
- FIX: Weight renormalization is now down-only (if sum > 1), so capped names (e.g., 10%) don't get inflated
- FIX: Risk check uses projected position cap and auto-shrinks BUY qty to fit under per-name % of equity
"""

import argparse
import csv
import os
import sys
import json
import shutil
import uuid
import time
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone, date
from typing import Dict, Any, List, Optional, Tuple, Set

# Path bootstrap
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from services.broker_alpaca import AlpacaBroker, AlpacaError
from services.notify import notify

RESULTS_DIR = os.path.join("data", "results")
ORDERS_DIR = os.path.join("data", "orders")

# produced by consolidate_orders.py (or legacy flows)
TODAY_ORDERS_CSV = os.path.join(ORDERS_DIR, "orders_today.csv")
LIVE_ORDERS_LOG = os.path.join(RESULTS_DIR, "live_orders.csv")
DEFAULT_WEIGHTS_CSV = os.path.join(RESULTS_DIR, "weights.csv")

EXPECTED_COLS = [
    "timestamp", "session", "action", "symbol", "side", "qty",
    "type", "limit_price", "order_id", "status",
    "filled_qty", "filled_avg_price", "client_order_id",
    "tp_limit", "sl_stop"
]

# Edit this if you want a permanent default blocklist:
DEFAULT_TICKERS_EXCLUDE = {"UNG", "WFC", "GE"}

# -------------------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _upgrade_log_schema_if_needed():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(LIVE_ORDERS_LOG):
        return
    try:
        with open(LIVE_ORDERS_LOG, "r", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            with open(LIVE_ORDERS_LOG, "w", newline="") as f:
                csv.writer(f).writerow(EXPECTED_COLS)
            return
        if rows[0] == EXPECTED_COLS:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = os.path.join(RESULTS_DIR, f"live_orders.backup.{ts}.csv")
        shutil.copyfile(LIVE_ORDERS_LOG, backup)
        with open(LIVE_ORDERS_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(EXPECTED_COLS)
            for row in rows[1:]:
                new_row = (row + [""] * len(EXPECTED_COLS))[:len(EXPECTED_COLS)]
                w.writerow(new_row)
        print("Upgraded live_orders.csv schema. Backup:", backup)
    except Exception as e:
        print("Log schema upgrade skipped:", e)

def _ensure_log():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(LIVE_ORDERS_LOG):
        with open(LIVE_ORDERS_LOG, "w", newline="") as f:
            csv.writer(f).writerow(EXPECTED_COLS)
    else:
        _upgrade_log_schema_if_needed()

def _append_log(row: List[Any]):
    with open(LIVE_ORDERS_LOG, "a", newline="") as f:
        csv.writer(f).writerow(row)

def _quantize(price: Optional[float], tick_size: float) -> Optional[float]:
    if price is None:
        return None
    q = Decimal(str(tick_size))
    p = Decimal(str(price))
    return float(p.quantize(q, rounding=ROUND_HALF_UP))

def _load_ticks_map() -> Dict[str, float]:
    path = os.path.join(ROOT, "config", "ticks.csv")
    out: Dict[str, float] = {}
    try:
        import csv as _csv
        with open(path, "r", newline="") as f:
            for r in _csv.DictReader(f):
                sym = (r.get("symbol") or "").upper().strip()
                ts = float(r.get("tick_size") or 0)
                if sym and ts > 0:
                    out[sym] = ts
    except Exception:
        pass
    return out

def _load_risk() -> Dict[str, Any]:
    """
    config/risk.json keys:
      - max_daily_notional (float, dollars)
      - max_position_pct (float, 0.10 = 10% of equity)
      - bp_buffer (float, dollars)
      - bp_buffer_pct (float, either 0.05 or 5 means 5%)
    """
    risk = {
        "max_daily_notional": 20000.0,
        "max_position_pct": 0.10,
        "bp_buffer": 0.0,
        "bp_buffer_pct": 0.0,
    }
    try:
        with open(os.path.join(ROOT, "config", "risk.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in ("max_daily_notional", "max_position_pct", "bp_buffer", "bp_buffer_pct"):
            if k in data:
                risk[k] = float(data[k])
    except Exception:
        pass
    return risk

def _normalize_pct(x: Optional[float]) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
    except Exception:
        return 0.0
    # Accept both 0.05 and 5 as 5%
    if v > 1.0:
        v *= 0.01
    return max(0.0, v)

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    synonyms = {
        "symbol": ["symbol", "ticker", "asset", "code"],
        "side": ["side", "action", "signal", "direction"],
        "qty": ["qty", "quantity", "shares", "size", "amount"],
        "order_type": ["order_type", "type", "ordertype"],
        "limit_price": ["limit_price", "limit", "price", "limitprice", "entry_price", "entry"],
    }
    def ensure_col(target: str):
        if target in df.columns: return
        for syn in synonyms.get(target, []):
            if syn in df.columns:
                df.rename(columns={syn: target}, inplace=True)
                return
    for tgt in ["symbol", "side", "qty", "order_type", "limit_price"]:
        ensure_col(tgt)

    if "symbol" not in df.columns: df["symbol"] = pd.NA
    if "side" not in df.columns: df["side"] = pd.NA
    if "qty" not in df.columns: df["qty"] = 0
    if "order_type" not in df.columns: df["order_type"] = "market"
    if "limit_price" not in df.columns: df["limit_price"] = pd.NA

    df["side"] = df["side"].astype(str).str.strip().str.lower().replace(
        {"long": "buy", "bull": "buy", "short": "sell", "bear": "sell", "neutral": "hold"}
    )
    df["order_type"] = df["order_type"].astype(str).str.strip().str.lower().replace(
        {"lmt": "limit", "mkt": "market"}
    )
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    df["limit_price"] = pd.to_numeric(df["limit_price"], errors="coerce")

    side_series = df["side"].astype(str).str.lower()
    qty_series = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    mask_valid = side_series.isin(["buy", "sell"]) & (qty_series > 0)

    df = df.loc[mask_valid].copy()
    df["qty"] = qty_series.loc[df.index].astype(int)
    df["side"] = side_series.loc[df.index]
    return df

def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"No CSV found at {path}; skipping placement.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Could not read CSV {path}: {e}")
        return pd.DataFrame()

def _load_today_orders_raw() -> pd.DataFrame:
    return _load_csv(TODAY_ORDERS_CSV)

def _is_weights_table(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return {"ticker", "target_weight"}.issubset(cols) and not {"side", "qty"}.issubset(cols)

def _load_positions_map(broker: AlpacaBroker) -> Dict[str, int]:
    try:
        pos = broker.get_positions()
    except AlpacaError as e:
        print("Positions fetch failed:", e)
        return {}
    out: Dict[str, int] = {}
    for p in pos:
        sym = (p.get("symbol") or "").upper()
        qty = int(float(p.get("qty", 0)))
        side = (p.get("side", "long") or "long").lower()
        out[sym] = qty if side == "long" else -qty
    return out

def _today_submitted_notional() -> float:
    if not os.path.exists(LIVE_ORDERS_LOG):
        return 0.0
    try:
        df = pd.read_csv(LIVE_ORDERS_LOG)
        if df.empty:
            return 0.0
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        today_utc = pd.Timestamp(date.today()).tz_localize("UTC")
        df = df[(df["action"] == "submit") & (df["timestamp"] >= today_utc)]
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
        df["limit_price"] = pd.to_numeric(df["limit_price"], errors="coerce")
        df["notional"] = df["qty"] * df["limit_price"].fillna(0)
        return float(df["notional"].sum())
    except Exception:
        return 0.0

def _build_open_duplicates_set(open_orders: List[Dict[str, Any]]) -> Set[Tuple[str, str, int, str, Optional[float]]]:
    def _norm_price(v) -> Optional[float]:
        if v in (None, ""):
            return None
        try:
            return float(v)
        except Exception:
            return None
    dup: Set[Tuple[str, str, int, str, Optional[float]]] = set()
    for o in open_orders:
        key = (
            (o.get("symbol") or "").upper(),
            (o.get("side") or "").lower(),
            int(float(o.get("qty", 0) or 0)),
            (o.get("type") or o.get("order_type") or "").lower(),
            _norm_price(o.get("limit_price", None)),
        )
        dup.add(key)
    return dup

def _has_same_order_today(broker: AlpacaBroker, symbol: str, side: str, qty: int, otype: str, q_limit: Optional[float]) -> bool:
    try:
        after = datetime.now(timezone.utc).date().isoformat() + "T00:00:00Z"
        orders = broker.list_orders(status="all", limit=200, after=after)
    except Exception:
        return False
    for o in orders:
        if (o.get("symbol") or "").upper() != symbol: continue
        if (o.get("side") or "").lower() != side: continue
        try:
            oq = int(float(o.get("qty", 0) or 0))
        except Exception:
            oq = 0
        if oq != qty: continue
        otype_o = (o.get("type") or o.get("order_type") or "").lower()
        if otype_o != otype: continue
        olimit = o.get("limit_price", None)
        if otype == "limit":
            try:
                if float(olimit) != float(q_limit if q_limit is not None else "nan"):
                    continue
            except Exception:
                continue
        if (o.get("status") or "").lower() in {"new", "accepted", "pending_new", "partially_filled", "filled"}:
            return True
    return False

# --- Weights → orders sizing helpers -------------------------------------------------

def _latest_price(broker: AlpacaBroker, symbol: str) -> Optional[float]:
    try:
        p = broker.get_latest_price(symbol)
        return float(p) if p is not None else None
    except Exception:
        return None

def _read_weights_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    if "ticker" not in df.columns or "target_weight" not in df.columns:
        return pd.DataFrame()
    df = df[["ticker", "target_weight"]].dropna()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0.0)
    df = df[df["target_weight"] >= 0]
    return df

def _compute_equity(broker: AlpacaBroker, equity_source: str) -> float:
    eq = 0.0
    try:
        acct = broker.get_account()
        if equity_source == "account":
            eq = float(acct.get("equity", 0) or 0)
        else:
            pos = broker.get_positions()
            eq = sum(float(p.get("market_value", 0) or 0) for p in pos)
    except Exception:
        pass
    return max(0.0, eq)

def _weights_to_orders(
    broker: AlpacaBroker,
    weights_df: pd.DataFrame,
    equity_source: str,
    cash_buffer_pct: float,
    min_qty: int,
    verbose: bool
) -> pd.DataFrame:
    """Return normalized orders DataFrame with ['symbol','side','qty','order_type','limit_price']."""
    pos_map = _load_positions_map(broker)

    last_price: Dict[str, float] = {}
    cur_value: Dict[str, float] = {}

    equity = _compute_equity(broker, equity_source)
    # cash_buffer_pct is already expressed as fraction (0.05 means 5%); do not renormalize here
    equity_effective = equity * (1.0 - max(0.0, cash_buffer_pct))

    weights_df = weights_df.copy()
    s = float(weights_df["target_weight"].sum())
    # Down-only renorm: if someone sends weights summing > 1, scale down; otherwise preserve caps
    if s > 1.0 + 1e-6:
        weights_df["target_weight"] = weights_df["target_weight"] / s

    for _, r in weights_df.iterrows():
        sym = r["ticker"]
        lp = _latest_price(broker, sym)
        if lp is None or lp <= 0:
            lp = 0.0
        last_price[sym] = lp

    orders: List[Dict[str, Any]] = []
    for _, r in weights_df.iterrows():
        sym = r["ticker"]
        tw = float(r["target_weight"])
        target_dollars = equity_effective * tw
        lp = last_price.get(sym, 0.0)

        pq = int(pos_map.get(sym, 0))
        cur_val = abs(pq) * lp
        cur_value[sym] = cur_val

        delta = target_dollars - cur_val
        action = None
        qty = 0

        if lp > 0:
            if delta > 0:
                qty = int(delta // lp)
                if qty >= min_qty:
                    action = "buy"
            elif delta < 0:
                qty = int((-delta) // lp)
                if qty >= min_qty:
                    action = "sell"

        if verbose:
            flag = action if action else "lt_1_share"
            print(("   (%r, %.16f, %s, %s, %s, %r)" % (
                sym, tw, f"{target_dollars:.3f}", f"{cur_val:.3f}", f"{lp:.3f}", (f"{action}:{qty}" if action else flag)
            )))

        if action and qty > 0:
            orders.append({
                "symbol": sym,
                "side": action,
                "qty": qty,
                "order_type": "market",
                "limit_price": None
            })

    if not orders:
        return pd.DataFrame()

    df = pd.DataFrame(orders)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df

# --- Optional: cancel helpers ---------------------------------------------------------

def _cancel_orders(broker: AlpacaBroker, orders: List[Dict[str, Any]]):
    for o in orders:
        oid = o.get("id")
        if not oid:
            continue
        try:
            broker.cancel_order(oid)
        except Exception as e:
            print(f"Cancel failed for {o.get('symbol','?')} id={oid}: {e}")

# --- Filtering helpers ----------------------------------------------------------------

def _parse_sym_list(arg: Optional[str]) -> Set[str]:
    if not arg:
        return set()
    return {s.strip().upper() for s in str(arg).split(",") if s.strip()}

# -------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Place live orders (paper/live) with risk caps, ticks, notifications.")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--live", action="store_true", help="If set, actually submit to broker.")
    parser.add_argument("--session", default=None, help="Optional session label; default is YYYYMMDD-HHMMSS.")
    parser.add_argument("--use-bracket", action="store_true", help="Attach TP/SL bracket orders (BUY-only by default).")
    parser.add_argument("--tp-pct", type=float, default=0.03, help="Take profit percent for brackets (default 0.03 = 3%).")
    parser.add_argument("--sl-pct", type=float, default=0.015, help="Stop loss percent for brackets (default 0.015 = 1.5%).")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile intended qty vs current position.")
    parser.add_argument("--reconcile-mode", default="all", choices=["all", "sells_only", "buys_only", "none"],
                        help="Which sides to reconcile (default all).")
    parser.add_argument("--tick-size", type=float, default=0.01, help="Default price tick size (overridden by config/ticks.csv).")
    parser.add_argument("--tif", default="day", choices=["day", "gtc"], help="time_in_force for orders.")
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow identical orders more than once today.")
    parser.add_argument("--no-bp-check", action="store_true", help="Disable buying power pre-check.")
    parser.add_argument("--bp-buffer", type=float, default=None, help="Override absolute BP buffer (dollars).")
    parser.add_argument("--bp-buffer-pct", type=float, default=None, help="Override fractional BP buffer (5 or 0.05 both mean 5%).")

    # ---- Order-behavior patch flags ----
    parser.add_argument("--min-notional", type=float, default=200.0, help="Skip orders with est_notional below this ($).")
    parser.add_argument("--large-cap-price-threshold", type=float, default=100.0,
                        help="Price threshold to treat as large/mega cap for buy-limit offset policy.")
    parser.add_argument("--buy-limit-offset-large", type=float, default=0.001,
                        help="BUY limit offset for large caps (e.g., 0.001 = 0.1%% below last).")
    parser.add_argument("--buy-limit-offset-small", type=float, default=0.005,
                        help="BUY limit offset for mid/small caps (e.g., 0.005 = 0.5%% below last).")
    parser.add_argument("--respect-input-types", action="store_true",
                        help="If set, do NOT convert BUY orders to limit automatically.")
    parser.add_argument("--bracket-on-sell", action="store_true",
                        help="If set, also attach brackets on SELLs (default: BUY-only).")

    # ---- Weights table mode ----
    parser.add_argument("--from-weights", action="store_true",
                        help="Interpret CSV as weights (ticker,target_weight) and size orders automatically.")
    parser.add_argument("--weights-file", default=DEFAULT_WEIGHTS_CSV,
                        help=f"Path to weights CSV when using --from-weights (default: {DEFAULT_WEIGHTS_CSV}).")
    parser.add_argument("--equity-source", default="account", choices=["account", "positions"],
                        help="How to compute portfolio equity for weights sizing.")
    parser.add_argument("--cash-buffer-pct", type=float, default=0.00,
                        help="Keep this fraction of equity in cash when sizing from weights (0.05 = 5%).")
    parser.add_argument("--min-qty", type=int, default=1,
                        help="Minimum whole shares per order when sizing from weights.")
    parser.add_argument("--verbose", action="store_true", help="More diagnostics.")

    # ---- Broker friction guards ----
    parser.add_argument("--cancel-opposites", action="store_true",
                        help="Cancel opposite-side open orders for the same symbol to avoid wash trades.")

    # SELL availability reduction: default ON; allow opt-out
    parser.add_argument("--reduce-sell-to-available", dest="reduce_sell_to_available",
                        action="store_true", default=None,
                        help="Reduce SELL qty to free shares after accounting for open SELLs. Default: enabled. "
                             "Use --no-reduce-sell-to-available to disable.")
    parser.add_argument("--no-reduce-sell-to-available", dest="no_reduce_sell_to_available",
                        action="store_true",
                        help="Disable automatic reduction of SELL qty to available shares.")

    # ---- New filters / caps ----
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated tickers to exclude (in addition to DEFAULT_TICKERS_EXCLUDE).")
    parser.add_argument("--only", type=str, default="",
                        help="If set, only allow this comma-separated list of tickers.")
    parser.add_argument("--max-sell-fraction", type=float, default=1.0,
                        help="Cap each SELL to this fraction of current position (e.g., 0.33). 1.0 = no cap.")
    parser.add_argument("--max-orders", type=int, default=0,
                        help="If >0, stop after logging/submitting this many orders in this run.")

    # ---- Convenience mode ----
    parser.add_argument("--sell-only", action="store_true",
                        help="Convenience: same as --reconcile --reconcile-mode sells_only; "
                             "SELL reduction is enabled by default.")

    args = parser.parse_args()

    # Convenience: --sell-only config
    if args.sell_only:
        args.reconcile = True
        args.reconcile_mode = "sells_only"
        # reduction stays default-enabled unless explicitly disabled

    # Compute effective SELL reduction flag (default True, unless explicitly disabled)
    if args.no_reduce_sell_to_available:
        reduce_sells = False
    elif args.reduce_sell_to_available is not None:
        reduce_sells = bool(args.reduce_sell_to_available)
    else:
        reduce_sells = True  # default ON

    session = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    _ensure_log()

    # Init broker
    try:
        broker = AlpacaBroker(mode=args.mode)
    except Exception as e:
        print("Broker init failed:", e)
        return

    # Load risk & ticks
    ticks_map = _load_ticks_map()
    risk_cfg = _load_risk()

    # Account state
    account_equity = 0.0
    buying_power = 0.0
    try:
        acct = broker.get_account()
        account_equity = float(acct.get("equity", 0) or 0)
        buying_power = float(acct.get("buying_power", 0) or 0)
    except Exception:
        pass

    # Buying power reserve (normalize pct so 5 == 0.05)
    bp_abs = args.bp_buffer if args.bp_buffer is not None else float(risk_cfg.get("bp_buffer", 0.0) or 0.0)
    bp_pct_raw = args.bp_buffer_pct if args.bp_buffer_pct is not None else float(risk_cfg.get("bp_buffer_pct", 0.0) or 0.0)
    bp_pct = _normalize_pct(bp_pct_raw)
    bp_effective_reserve = max(bp_abs, buying_power * bp_pct)
    bp_effective = max(0.0, buying_power - bp_effective_reserve)
    bp_remaining = bp_effective

    # Load CSV according to mode
    if args.from_weights:
        wpath = args.weights_file or DEFAULT_WEIGHTS_CSV
        if args.verbose:
            print(f"Loading weights from: {wpath}")
        raw_csv = _load_csv(wpath)
    else:
        raw_csv = _load_today_orders_raw()

    if raw_csv.empty:
        return

    from_weights = bool(args.from_weights) or _is_weights_table(raw_csv)

    if from_weights:
        if args.verbose:
            print("Detected weights table. Raw head:")
            try:
                print(raw_csv.head().to_string(index=False))
            except Exception:
                print(list(raw_csv.columns))
            print("Columns:", list(raw_csv.columns))

        weights_df = _read_weights_df(raw_csv)
        if weights_df.empty:
            print("Weights table is empty or invalid; nothing to do.")
            return

        if args.verbose:
            print("Weights→Orders sizing (sym, tw, target$, cur$, last, action):")

        orders_df = _weights_to_orders(
            broker=broker,
            weights_df=weights_df,
            equity_source=args.equity_source,
            cash_buffer_pct=float(args.cash_buffer_pct or 0.0),
            min_qty=int(args.min_qty or 1),
            verbose=args.verbose
        )
        if orders_df.empty:
            print("No actionable orders from weights sizing.")
            return
    else:
        if args.verbose:
            print("Detected orders table. Raw head:")
            try:
                print(raw_csv.head().to_string(index=False))
            except Exception:
                print(list(raw_csv.columns))
            print("Columns:", list(raw_csv.columns))
        orders_df = _canonicalize_columns(raw_csv)
        if orders_df.empty:
            print("No actionable rows after normalization.")
            if args.verbose:
                print("Raw head:")
                try:
                    print(raw_csv.head().to_string(index=False))
                except Exception:
                    print(list(raw_csv.columns))
                print("Columns:", list(raw_csv.columns))
            return

    # --- Apply symbol filters (exclude / only) ---------------------------------------
    user_exclude = _parse_sym_list(args.exclude)
    only_set = _parse_sym_list(args.only)
    full_exclude = set(s.upper() for s in DEFAULT_TICKERS_EXCLUDE) | user_exclude

    # Positions map (for reconcile and availability)
    pos_map: Dict[str, int] = _load_positions_map(broker) if args.reconcile and args.reconcile_mode != "none" else _load_positions_map(broker)

    # Pull open orders once & index
    try:
        open_orders = broker.list_orders(status="open", limit=200)
    except AlpacaError:
        open_orders = []
    dup_open = _build_open_duplicates_set(open_orders)

    open_by_symbol = defaultdict(lambda: {"buy": [], "sell": []})
    open_sell_qty = defaultdict(float)
    for o in open_orders:
        sym = (o.get("symbol") or "").upper()
        side_o = (o.get("side") or "").lower()
        try:
            q = float(o.get("qty", 0) or 0)
        except Exception:
            q = 0.0
        open_by_symbol[sym][side_o].append(o)
        if side_o == "sell":
            open_sell_qty[sym] += q

    submitted = 0
    processed = 0
    used_notional = _today_submitted_notional()

    for _, row in orders_df.iterrows():
        symbol = str(row["symbol"]).upper().strip()
        side = str(row["side"]).lower().strip()
        qty = int(row["qty"])
        input_type = str(row.get("order_type", "market")).lower().strip()

        # Symbol allow/deny
        if symbol in full_exclude:
            if args.verbose:
                print(f"{symbol} is in exclude list; skipping.")
            _append_log([_now_iso(), session, "exclude-skip", symbol, side, qty, input_type, row.get("limit_price","") or "",
                         "", "SKIP", 0, "", "", "", ""])
            continue
        if only_set and symbol not in only_set:
            if args.verbose:
                print(f"{symbol} not in --only list; skipping.")
            _append_log([_now_iso(), session, "only-skip", symbol, side, qty, input_type, row.get("limit_price","") or "",
                         "", "SKIP", 0, "", "", "", ""])
            continue

        # Respect --max-orders (counts both live and dry-run entries we handle)
        if args.max_orders and processed >= args.max_orders:
            if args.verbose:
                print(f"Reached --max-orders limit ({args.max_orders}); stopping loop.")
            break

        processed += 1

        # ---- Wash-trade prevention: cancel opposite opens if requested
        opp_side = "sell" if side == "buy" else "buy"
        opp_open = open_by_symbol[symbol][opp_side]
        if opp_open:
            if args.cancel_opposites:
                if args.verbose:
                    print(f"Canceling {len(opp_open)} open {opp_side.upper()} orders for {symbol} to avoid wash-trade...")
                _cancel_orders(broker, opp_open)
                time.sleep(0.5)
                try:
                    refreshed = broker.list_orders(status="open", limit=200)
                except AlpacaError:
                    refreshed = open_orders
                open_by_symbol[symbol] = {"buy": [], "sell": []}
                open_sell_qty[symbol] = 0.0
                dup_open = _build_open_duplicates_set(refreshed)
                for o in refreshed:
                    s2 = (o.get("symbol") or "").upper()
                    sd = (o.get("side") or "").lower()
                    try:
                        q2 = float(o.get("qty", 0) or 0)
                    except Exception:
                        q2 = 0.0
                    if s2 == symbol:
                        open_by_symbol[s2][sd].append(o)
                        if sd == "sell":
                            open_sell_qty[s2] += q2
            else:
                if args.verbose:
                    print(f"Opposite-side open exists for {symbol} ({opp_side}); skipping. Use --cancel-opposites to auto-cancel.")
                _append_log([_now_iso(), session, "wash-skip", symbol, side, qty, input_type, row.get("limit_price","") or "",
                             "", "SKIP", 0, "", "", "", ""])
                continue

        # ---- SELL availability reduction (effective flag)
        if side == "sell" and reduce_sells:
            cur_pos_qty = int(pos_map.get(symbol, 0)) if symbol in pos_map else 0
            held_sell = int(open_sell_qty.get(symbol, 0.0))
            avail = max(0, cur_pos_qty - held_sell)
            if qty > avail:
                if args.verbose:
                    print(f"{symbol} SELL qty {qty} > available {avail} (pos={cur_pos_qty}, held_sells={held_sell}); reducing.")
                qty = avail
            if qty <= 0:
                if args.verbose:
                    print(f"No available shares to sell for {symbol} after holds; skipping.")
                _append_log([_now_iso(), session, "avail-skip", symbol, side, 0, input_type, row.get("limit_price","") or "",
                             "", "SKIP", 0, "", "", "", ""])
                continue

        # ---- SELL per-run cap (max-sell-fraction)
        if side == "sell" and args.max_sell_fraction < 1.0:
            cur_pos_qty = int(pos_map.get(symbol, 0))
            cap = max(0, int(cur_pos_qty * max(0.0, args.max_sell_fraction)))
            if qty > cap:
                if args.verbose:
                    print(f"{symbol} SELL qty {qty} > cap {cap} (max-sell-fraction={args.max_sell_fraction:.3f}); reducing.")
                qty = cap
            if qty <= 0:
                if args.verbose:
                    print(f"{symbol} SELL capped to 0 by --max-sell-fraction; skipping.")
                _append_log([_now_iso(), session, "sell-cap-skip", symbol, side, 0, input_type, row.get("limit_price","") or "",
                             "", "SKIP", 0, "", "", "", ""])
                continue

        # latest/tick
        tick_size = float(ticks_map.get(symbol, args.tick_size))

        # baseline price (prefer provided limit; else fetch)
        limit_price_input = row.get("limit_price", None)
        limit_price_input = None if (limit_price_input is None or pd.isna(limit_price_input)) else float(limit_price_input)
        ref_price = limit_price_input
        if ref_price is None:
            lp = _latest_price(broker, symbol)
            ref_price = lp if lp else None
        est_price = ref_price or 0.0
        est_notional = est_price * qty

        # --- min notional filter ---
        if est_notional < float(args.min_notional or 0):
            if args.verbose:
                print(f"Min-notional skip: {symbol} {side.upper()} {qty} est_notional={est_notional:.2f} < {args.min_notional:.2f}")
            _append_log([_now_iso(), session, "min-skip", symbol, side, qty, input_type, limit_price_input or "",
                         "", "SKIP", 0, "", "", "", ""])
            continue

        # Reconcile vs positions (optional)
        if symbol in pos_map:
            current = pos_map[symbol]
            if args.reconcile and args.reconcile_mode in ("all", "buys_only") and side == "buy" and current > 0:
                qty = max(qty - current, 0)
            if args.reconcile and args.reconcile_mode in ("all", "sells_only") and side == "sell" and current > 0:
                qty = min(qty, current)
            if qty == 0:
                if args.verbose:
                    print(f"Reconciled qty is 0 for {symbol}; skipping.")
                continue
            est_notional = est_price * qty

        # --- BUY/SELL order-type policy ---
        otype = input_type
        q_limit: Optional[float] = limit_price_input

        if side == "sell":
            otype = "market"
            q_limit = None
        else:
            if not args.respect_input_types:
                last = est_price
                if last and last > 0:
                    is_large = last >= float(args.large_cap_price_threshold)
                    offset = float(args.buy_limit_offset_large) if is_large else float(args.buy_limit_offset_small)
                    target = last * (1.0 - offset)
                    q_limit = _quantize(target, tick_size)
                    otype = "limit"
                else:
                    otype = "market"
                    q_limit = None
            else:
                if otype == "limit" and q_limit is not None:
                    q_limit = _quantize(q_limit, tick_size)

        # Recompute est_notional with limit when BUY
        est_price_for_checks = q_limit if (side == "buy" and otype == "limit" and q_limit) else est_price
        est_notional = (est_price_for_checks or 0.0) * qty

        # --- Projected position cap with auto-shrink (per-name % of equity) ---
        if side == "buy" and account_equity > 0:
            cap_dollars = account_equity * float(risk_cfg.get("max_position_pct", 0.10))
            cur_pos_qty = int(pos_map.get(symbol, 0))
            current_val = abs(cur_pos_qty) * (est_price_for_checks or est_price or 0.0)
            max_add_notional = max(0.0, cap_dollars - current_val)
            if est_notional > max_add_notional:
                # shrink qty to fit under the cap
                px = (est_price_for_checks or est_price or 0.0) or 1e18
                max_qty = int(max_add_notional // px)
                if max_qty <= 0:
                    print(f"Risk block: {symbol} projected position would exceed {float(risk_cfg.get('max_position_pct',0.10))*100:.1f}% of equity.")
                    notify("error", f"Risk block: {symbol} projected > cap")
                    _append_log([_now_iso(), session, "risk-cap-skip", symbol, side, qty, otype, q_limit or "",
                                 "", "SKIP", 0, "", "", "", ""])
                    continue
                if max_qty < qty:
                    if args.verbose:
                        print(f"{symbol} BUY qty {qty} > cap-fit {max_qty}; reducing.")
                    qty = max_qty
                    est_notional = (est_price_for_checks or 0.0) * qty
                    # If now below min-notional, skip
                    if est_notional < float(args.min_notional or 0):
                        if args.verbose:
                            print(f"Post-risk min-notional skip: {symbol} {side.upper()} {qty} est_notional={est_notional:.2f} < {args.min_notional:.2f}")
                        _append_log([_now_iso(), session, "min-skip", symbol, side, qty, otype, q_limit or "",
                                     "", "SKIP", 0, "", "", "", ""])
                        continue

        # --- Buying Power Check (for BUYs) ---
        if not args.no_bp_check and side == "buy":
            if bp_remaining <= 0.0 or est_notional > bp_remaining:
                msg = (f"BP block: {symbol} BUY {qty} est_notional={est_notional:.2f} "
                       f"> remaining_bp={bp_remaining:.2f} (reserve={bp_effective_reserve:.2f}, "
                       f"reported_bp={buying_power:.2f}).")
                print(msg)
                _append_log([_now_iso(), session, "bp-block", symbol, side, qty, otype, q_limit or "",
                             "", "SKIP", 0, "", "", "", ""])
                notify("error", msg)
                continue

        # --- Risk: daily notional cap (BUY-only) ---
        if side == "buy" and (used_notional + est_notional > float(risk_cfg.get("max_daily_notional", 0.0))):
            print(f"Risk block: daily notional cap reached ({used_notional:.2f} + {est_notional:.2f} > "
                  f"{float(risk_cfg.get('max_daily_notional',0.0)):.2f}).")
            notify("error", f"Risk block daily cap: {used_notional:.2f} + {est_notional:.2f} > "
                            f"{float(risk_cfg.get('max_daily_notional',0.0)):.2f}")
            continue

        # Quantize limit (again)
        if otype == "limit" and q_limit is not None:
            q_limit = _quantize(q_limit, tick_size)

        # Duplicate checks
        dup_key = (symbol, side, qty, otype, q_limit if otype == "limit" else None)
        if dup_key in dup_open:
            msg = (f"Duplicate open order exists for {symbol} {side.upper()} {qty} {otype.upper()} "
                   f"{('at %.2f' % q_limit) if (q_limit is not None) else ''}; skipping.")
            print(msg)
            _append_log([_now_iso(), session, "dupe-skip", symbol, side, qty, otype, q_limit,
                         "", "SKIPPED", 0, "", "", "", ""])
            notify("error", msg)
            continue

        if not args.allow_duplicates and _has_same_order_today(broker, symbol, side, qty, otype, q_limit):
            msg = (f"Duplicate guard: {symbol} {side.upper()} {qty} {otype.upper()} "
                   f"{('at %.2f' % q_limit) if (q_limit is not None) else 'MKT'} already submitted today; skipping.")
            print(msg)
            _append_log([_now_iso(), session, "skip-dup", symbol, side, qty, otype, q_limit,
                         "", "SKIP", 0, "", "", "", ""])
            notify("error", msg)
            continue

        # Dry run?
        if not args.live:
            print(f"DRY RUN: {symbol} {side.upper()} {qty} {otype.upper()} "
                  f"{('at %.2f' % q_limit) if (q_limit is not None) else ''}")
            _append_log([_now_iso(), session, "dry-run", symbol, side, qty, otype, q_limit,
                         "", "DRY", 0, "", "", "", ""])
            continue

        # Unique client_order_id
        uid_tail = uuid.uuid4().hex[:6]
        time_tail = datetime.now(timezone.utc).strftime("%H%M%S")
        client_id = f"{session}-{symbol}-{side}-{qty}-{time_tail}-{uid_tail}"

        # Brackets: BUY-only by default (unless --bracket-on-sell set)
        order_class = None
        take_profit = None
        stop_loss = None
        allow_bracket = (side == "buy") or bool(args.bracket_on_sell)

        if args.use_bracket and allow_bracket and est_price > 0:
            if side == "buy":
                tp_raw = est_price * (1.0 + args.tp_pct)
                sl_raw = est_price * (1.0 - args.sl_pct)
            else:
                tp_raw = est_price * (1.0 - args.tp_pct)
                sl_raw = est_price * (1.0 + args.sl_pct)
            tp = _quantize(tp_raw, tick_size)
            sl = _quantize(sl_raw, tick_size)
            order_class = "bracket"
            take_profit = {"limit_price": tp}
            stop_loss = {"stop_price": sl}

        try:
            resp = broker.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=otype,
                time_in_force=args.tif,
                limit_price=q_limit if otype == "limit" else None,
                client_order_id=client_id,
                order_class=order_class,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
            order_id = resp.get("id", "")
            status = resp.get("status", "")
            filled_qty = int(float(resp.get("filled_qty", 0) or 0))
            filled_avg_price = resp.get("filled_avg_price", "")

            _append_log([
                _now_iso(), session, "submit", symbol, side, qty, otype, q_limit,
                order_id, status, filled_qty, filled_avg_price, client_id,
                take_profit["limit_price"] if take_profit else "",
                stop_loss["stop_price"] if stop_loss else "",
            ])
            print(f"Submitted {symbol} {side.upper()} {qty} {otype.upper()}. status={status} order_id={order_id}")
            notify(
                "submit",
                f"{symbol} {side.upper()} {qty} {otype.upper()} "
                f"{('at %.2f' % q_limit) if q_limit else ''} status={status} id={order_id}"
            )
            submitted += 1
            if side == "buy":
                used_notional += est_notional
            if not args.no_bp_check and side == "buy":
                bp_remaining = max(0.0, bp_remaining - est_notional)
            dup_open.add(dup_key)

        except AlpacaError as e:
            print(f"Submit failed for {symbol}: {e}")
            _append_log([_now_iso(), session, "error", symbol, side, qty, otype, q_limit,
                         "", "ERROR", 0, "", client_id, "", ""])
            notify("error", f"Submit failed for {symbol}: {e}")

    print(f"Placement complete. Submitted: {submitted} (processed: {processed})")

if __name__ == "__main__":
    main()
