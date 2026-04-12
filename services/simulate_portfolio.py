# services/simulate_portfolio.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Defaults / Paths
# ─────────────────────────────────────────────────────────────

DEFAULT_INITIAL_BALANCE = 100_000.0
DEFAULT_POSITION_SIZE = 0.10  # 10% of CASH per new BUY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
PREDICTIONS_DIR = DATA_DIR / "predictions"

DEFAULT_SIGNAL_CANDIDATES: List[Path] = [
    RESULTS_DIR / "signals_with_rationale.csv",
    RESULTS_DIR / "signals.csv",
    PREDICTIONS_DIR / "signals.csv",
]

# Snapshot outputs (SAFE defaults)
DEFAULT_PORTFOLIO_SNAPSHOT_FILE = RESULTS_DIR / "portfolio_snapshot.csv"
DEFAULT_TRADE_LOG_SNAPSHOT_FILE = RESULTS_DIR / "trade_log_snapshot.csv"
DEFAULT_POSITIONS_SNAPSHOT_FILE = RESULTS_DIR / "positions_snapshot.csv"

# Legacy/history filenames (owned by backtest_signals.py)
LEGACY_PORTFOLIO_HISTORY_FILE = RESULTS_DIR / "portfolio_history.csv"
LEGACY_TRADE_LOG_FILE = RESULTS_DIR / "trade_log.csv"

PH_COLS = ["date", "cash", "market_value", "total_value"]
TL_COLS = ["date", "action", "ticker", "price", "quantity", "cash_after", "total_value"]
PS_COLS = ["date", "ticker", "shares", "avg_price", "last_price", "market_value", "weight_pct"]


# ─────────────────────────────────────────────────────────────
# IO / Normalization
# ─────────────────────────────────────────────────────────────


def _exists(p: Path | str) -> bool:
    p = Path(p)
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _normalize_signals_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has: date (naive datetime), ticker (UPPER), signal (UPPER), price (float).
    Accepts:
      - 'close' instead of 'price'
      - 'timestamp' instead of 'date'
    Drops rows missing essentials.
    """
    df = df.copy()

    # Date
    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(
                None
            )
        else:
            df["date"] = pd.Timestamp.today().normalize()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)

    # Ticker
    if "ticker" not in df.columns:
        raise ValueError("signals file has no 'ticker' column")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Signal
    if "signal" not in df.columns:
        df["signal"] = "HOLD"
    df["signal"] = df["signal"].astype(str).str.upper().str.strip()

    # Price
    if "price" not in df.columns:
        if "close" in df.columns:
            df = df.rename(columns={"close": "price"})
        else:
            df["price"] = np.nan
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df[["date", "ticker", "signal", "price"]]
    df = (
        df.dropna(subset=["date", "ticker", "signal"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    return df


def load_signals(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load signals from an explicit path or from default candidates."""
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend(DEFAULT_SIGNAL_CANDIDATES)

    last_err: Optional[Exception] = None
    for p in candidates:
        if _exists(p):
            try:
                df = pd.read_csv(p)
                df = _normalize_signals_schema(df)
                if not df.empty:
                    return df
            except Exception as e:
                last_err = e

    tried = ", ".join(str(c) for c in candidates)
    extra = f" (last error: {last_err})" if last_err else ""
    raise FileNotFoundError(f"No readable signals CSV found. Tried: {tried}{extra}")


# ─────────────────────────────────────────────────────────────
# Portfolio math
# ─────────────────────────────────────────────────────────────


def _mark_to_market(positions: Dict[str, Dict[str, float]], prices: Dict[str, float]) -> float:
    mv = 0.0
    for tkr, pos in positions.items():
        px = prices.get(tkr)
        if px is not None and np.isfinite(px):
            mv += float(pos.get("shares", 0.0)) * float(px)
    return float(mv)


def _positions_snapshot_df(
    positions: Dict[str, Dict[str, float]],
    last_prices: Dict[str, float],
    cash: float,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Build positions snapshot table.

    - Adds 'date' for integrity validation; same 'as_of' for all rows (incl. CASH).
    - Eliminates rounding drift by rounding non-cash weights to 4dp and forcing CASH as remainder,
      so sum(weight_pct) == 100.0000 (subject to normal floating point printing).
    """
    rows: List[Dict] = []
    mv = _mark_to_market(positions, last_prices)
    tv = float(cash) + float(mv)
    denom = tv if tv != 0 else 1.0

    as_of_ts = pd.to_datetime(as_of, errors="coerce") if as_of is not None else pd.NaT

    # Positions first (keep raw weights)
    for tkr, pos in sorted(positions.items()):
        shares = float(pos.get("shares", 0.0))
        avg_px = float(pos.get("avg_price", np.nan))
        last_px = float(last_prices.get(tkr, np.nan))

        mkt_val = shares * last_px if np.isfinite(last_px) else np.nan
        weight_raw = (mkt_val / denom) * 100.0 if np.isfinite(mkt_val) else np.nan

        rows.append(
            {
                "date": as_of_ts,
                "ticker": tkr,
                "shares": int(shares),
                "avg_price": round(avg_px, 6) if np.isfinite(avg_px) else np.nan,
                "last_price": round(last_px, 6) if np.isfinite(last_px) else np.nan,
                "market_value": round(mkt_val, 2) if np.isfinite(mkt_val) else np.nan,
                "weight_pct": weight_raw,
            }
        )

    # CASH row
    cash_weight_raw = (float(cash) / denom) * 100.0
    rows.append(
        {
            "date": as_of_ts,
            "ticker": "CASH",
            "shares": 0,
            "avg_price": np.nan,
            "last_price": np.nan,
            "market_value": round(float(cash), 2),
            "weight_pct": cash_weight_raw,
        }
    )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=PS_COLS)

    out["weight_pct"] = pd.to_numeric(out["weight_pct"], errors="coerce")

    non_cash_mask = out["ticker"] != "CASH"
    out.loc[non_cash_mask, "weight_pct"] = out.loc[non_cash_mask, "weight_pct"].round(4)

    non_cash_sum = float(out.loc[non_cash_mask, "weight_pct"].sum(skipna=True))
    cash_remainder = round(100.0 - non_cash_sum, 4)

    cash_idx = out.index[out["ticker"] == "CASH"]
    if len(cash_idx) == 1:
        out.loc[cash_idx[0], "weight_pct"] = cash_remainder
    else:
        out["weight_pct"] = out["weight_pct"].round(4)

    out = out[[c for c in PS_COLS if c in out.columns]]
    return out


def simulate_portfolio(
    trades_df: pd.DataFrame,
    starting_cash: float = DEFAULT_INITIAL_BALANCE,
    position_size: float = DEFAULT_POSITION_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate a simple portfolio curve from daily BUY/SELL signals.

    Returns:
      ph: [date, cash, market_value, total_value]
      tl: [date, action, ticker, price, quantity, cash_after, total_value]
      ps: [date, ticker, shares, avg_price, last_price, market_value, weight_pct]
    """
    if trades_df is None or trades_df.empty:
        return (
            pd.DataFrame(columns=PH_COLS),
            pd.DataFrame(columns=TL_COLS),
            pd.DataFrame(columns=PS_COLS),
        )

    df = trades_df.copy()

    for col in ("date", "ticker", "signal", "price"):
        if col not in df.columns:
            raise ValueError(
                f"simulate_portfolio: required column '{col}' missing after normalization"
            )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["signal"] = df["signal"].astype(str).str.upper().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = (
        df.dropna(subset=["date", "ticker", "signal"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    if df.empty:
        return (
            pd.DataFrame(columns=PH_COLS),
            pd.DataFrame(columns=TL_COLS),
            pd.DataFrame(columns=PS_COLS),
        )

    cash = float(starting_cash)
    positions: Dict[str, Dict[str, float]] = {}
    portfolio_history_rows: List[Dict] = []
    trade_log_rows: List[Dict] = []

    last_prices_for_day: Dict[str, float] = {}

    for day, day_df in df.groupby(df["date"].dt.normalize(), sort=True):
        day = pd.Timestamp(day)

        # last price map for the day
        last_prices_for_day = {}
        for r in day_df.itertuples(index=False):
            if np.isfinite(r.price):
                last_prices_for_day[r.ticker] = float(r.price)

        # SELL first
        for r in day_df[day_df["signal"] == "SELL"].itertuples(index=False):
            tkr = r.ticker
            px = float(r.price) if np.isfinite(r.price) else last_prices_for_day.get(tkr)
            if px is None or not np.isfinite(px) or px <= 0:
                continue

            pos = positions.get(tkr)
            if pos and float(pos.get("shares", 0.0)) > 0:
                qty = int(pos["shares"])
                cash += qty * px
                positions.pop(tkr, None)

                total_value = cash + _mark_to_market(positions, last_prices_for_day)
                trade_log_rows.append(
                    {
                        "date": day,
                        "action": "SELL",
                        "ticker": tkr,
                        "price": round(px, 6),
                        "quantity": -qty,
                        "cash_after": round(cash, 2),
                        "total_value": round(total_value, 2),
                    }
                )

        # BUY
        for r in day_df[day_df["signal"] == "BUY"].itertuples(index=False):
            tkr = r.ticker
            if tkr in positions and float(positions[tkr].get("shares", 0.0)) > 0:
                continue

            px = float(r.price) if np.isfinite(r.price) else last_prices_for_day.get(tkr)
            if px is None or not np.isfinite(px) or px <= 0:
                continue

            # clamp position size
            ps = float(position_size)
            if not np.isfinite(ps) or ps <= 0:
                continue
            ps = min(ps, 1.0)

            budget = cash * ps
            qty = int(budget // px)
            if qty <= 0:
                continue

            cost = qty * px
            if cost > cash:
                continue

            cash -= cost
            positions[tkr] = {"shares": qty, "avg_price": px}

            total_value = cash + _mark_to_market(positions, last_prices_for_day)
            trade_log_rows.append(
                {
                    "date": day,
                    "action": "BUY",
                    "ticker": tkr,
                    "price": round(px, 6),
                    "quantity": qty,
                    "cash_after": round(cash, 2),
                    "total_value": round(total_value, 2),
                }
            )

        mv = _mark_to_market(positions, last_prices_for_day)
        tv = cash + mv
        portfolio_history_rows.append(
            {
                "date": day,
                "cash": round(cash, 2),
                "market_value": round(mv, 2),
                "total_value": round(tv, 2),
            }
        )

    ph = pd.DataFrame(portfolio_history_rows)
    ph = (
        ph.sort_values("date").reset_index(drop=True)
        if not ph.empty
        else pd.DataFrame(columns=PH_COLS)
    )
    ph = ph[PH_COLS] if not ph.empty else pd.DataFrame(columns=PH_COLS)

    tl = pd.DataFrame(trade_log_rows)
    tl = (
        tl.sort_values("date").reset_index(drop=True)
        if not tl.empty
        else pd.DataFrame(columns=TL_COLS)
    )
    tl = tl[TL_COLS] if not tl.empty else pd.DataFrame(columns=TL_COLS)

    as_of_date = pd.to_datetime(ph["date"].max(), errors="coerce") if not ph.empty else None
    ps = _positions_snapshot_df(
        positions=positions, last_prices=last_prices_for_day, cash=cash, as_of=as_of_date
    )
    ps = ps[PS_COLS] if not ps.empty else pd.DataFrame(columns=PS_COLS)

    return ph, tl, ps


# ─────────────────────────────────────────────────────────────
# Carry-forward helper
# ─────────────────────────────────────────────────────────────


def _carry_forward_to_today(
    portfolio_history: pd.DataFrame, carry_time: str = "17:12:00"
) -> pd.DataFrame:
    """
    If last date < today, append a single "today" row at carry_time using last known totals.
    """
    if portfolio_history is None or portfolio_history.empty:
        return portfolio_history

    last_dt = pd.to_datetime(portfolio_history["date"].iloc[-1], errors="coerce")
    if pd.isna(last_dt):
        return portfolio_history

    today = pd.Timestamp.today().normalize()

    try:
        hh, mm, ss = [int(x) for x in carry_time.split(":")]
    except Exception:
        hh, mm, ss = 17, 12, 0

    today_dt = today + pd.Timedelta(hours=hh, minutes=mm, seconds=ss)

    if last_dt.normalize() >= today:
        return portfolio_history

    extra = pd.DataFrame(
        [
            {
                "date": today_dt,
                "cash": float(portfolio_history["cash"].iloc[-1]),
                "market_value": float(portfolio_history["market_value"].iloc[-1]),
                "total_value": float(portfolio_history["total_value"].iloc[-1]),
            }
        ]
    )

    out = (
        pd.concat([portfolio_history, extra], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out[PH_COLS]


# ─────────────────────────────────────────────────────────────
# Safety guard for legacy files
# ─────────────────────────────────────────────────────────────


def _guard_legacy_overwrite(out_hist: Path, out_trades: Path, allow_legacy: bool) -> None:
    """Refuse to overwrite legacy backtest-owned files unless explicitly allowed."""
    legacy_hist = LEGACY_PORTFOLIO_HISTORY_FILE.resolve()
    legacy_tr = LEGACY_TRADE_LOG_FILE.resolve()

    if allow_legacy:
        return

    if out_hist.resolve() == legacy_hist:
        raise SystemExit(
            f"Refusing to write to legacy {legacy_hist}. "
            f"Use --write-legacy-history-files if you REALLY want to overwrite backtest outputs."
        )

    if out_trades.resolve() == legacy_tr:
        raise SystemExit(
            f"Refusing to write to legacy {legacy_tr}. "
            f"Use --write-legacy-history-files if you REALLY want to overwrite backtest outputs."
        )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate a portfolio snapshot/curve from signals.")
    ap.add_argument(
        "--signals", type=str, default=None, help="Path to signals CSV. If omitted, tries defaults."
    )
    ap.add_argument(
        "--initial-cash", type=float, default=DEFAULT_INITIAL_BALANCE, help="Starting cash balance."
    )
    ap.add_argument(
        "--position-size",
        type=float,
        default=DEFAULT_POSITION_SIZE,
        help="Fraction of cash per new BUY (0..1).",
    )

    # SAFE defaults
    ap.add_argument(
        "--out-history",
        type=str,
        default=str(DEFAULT_PORTFOLIO_SNAPSHOT_FILE),
        help="Where to write portfolio snapshot CSV.",
    )
    ap.add_argument(
        "--out-trades",
        type=str,
        default=str(DEFAULT_TRADE_LOG_SNAPSHOT_FILE),
        help="Where to write trade log snapshot CSV.",
    )
    ap.add_argument(
        "--out-positions",
        type=str,
        default=str(DEFAULT_POSITIONS_SNAPSHOT_FILE),
        help="Where to write positions_snapshot.csv",
    )

    ap.add_argument(
        "--no-write", action="store_true", help="If set, prints tail instead of writing files."
    )
    ap.add_argument(
        "--carry-forward-to-today",
        action="store_true",
        help="Append a 'today' snapshot row using last known totals (no trading).",
    )
    ap.add_argument(
        "--carry-forward-time",
        type=str,
        default="17:12:00",
        help="Time to use for carry-forward row (HH:MM:SS).",
    )

    # Only if you *intentionally* want to overwrite dashboard/backtest history files
    ap.add_argument(
        "--write-legacy-history-files",
        action="store_true",
        help="Allow writing to data/results/portfolio_history.csv and data/results/trade_log.csv.",
    )

    args = ap.parse_args()

    signals_df = load_signals(args.signals)

    ph, tl, ps = simulate_portfolio(
        signals_df,
        starting_cash=args.initial_cash,
        position_size=args.position_size,
    )

    if args.carry_forward_to_today and ph is not None and not ph.empty:
        ph = _carry_forward_to_today(ph, carry_time=args.carry_forward_time)

    if args.no_write:
        print("---- portfolio_history (tail) ----")
        print(ph.tail(10).to_string(index=False))
        print("\n---- trade_log (tail) ----")
        print(tl.tail(10).to_string(index=False))
        print("\n---- positions_snapshot ----")
        print(ps.to_string(index=False))
        return

    out_hist = Path(args.out_history)
    out_tr = Path(args.out_trades)
    out_pos = Path(args.out_positions)

    _guard_legacy_overwrite(out_hist, out_tr, allow_legacy=args.write_legacy_history_files)

    out_hist.parent.mkdir(parents=True, exist_ok=True)
    out_tr.parent.mkdir(parents=True, exist_ok=True)
    out_pos.parent.mkdir(parents=True, exist_ok=True)

    # enforce exact schemas before writing
    ph = ph[PH_COLS] if not ph.empty else pd.DataFrame(columns=PH_COLS)
    tl = tl[TL_COLS] if not tl.empty else pd.DataFrame(columns=TL_COLS)
    ps = ps[PS_COLS] if not ps.empty else pd.DataFrame(columns=PS_COLS)

    ph.to_csv(out_hist, index=False)
    tl.to_csv(out_tr, index=False)
    ps.to_csv(out_pos, index=False)

    print(f"✅ Wrote {out_hist} ({len(ph)} rows)")
    print(f"✅ Wrote {out_tr} ({len(tl)} rows)")
    print(f"✅ Wrote {out_pos} ({len(ps)} rows)")

    # Optional legacy write (explicit)
    if args.write_legacy_history_files:
        legacy_hist = LEGACY_PORTFOLIO_HISTORY_FILE
        legacy_tr = LEGACY_TRADE_LOG_FILE

        ph.to_csv(legacy_hist, index=False)
        tl.to_csv(legacy_tr, index=False)

        print(f"✅ Also wrote legacy {legacy_hist} ({len(ph)} rows)")
        print(f"✅ Also wrote legacy {legacy_tr} ({len(tl)} rows)")


if __name__ == "__main__":
    main()
