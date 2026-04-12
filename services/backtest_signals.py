# services/backtest_signals.py
"""
Signal-based backtest with REAL SL/TP exits (priority over SELL signals).

Outputs (data/results):
  - backtest_trade_log.csv
  - backtest_portfolio_history.csv
  - backtest_summary.csv
  - strategy_vs_market.csv
  - trade_log.csv               (dashboard)
  - portfolio_history.csv       (dashboard)  ✅ 4 cols only: date,cash,market_value,total_value

Key fix (2026-01-22):
- Clamp ALL backtest outputs to the SAME effective AS_OF date used by signals.
  This prevents AS_OF contract failures when a subset of tickers (e.g., ^VIX)
  contains today's partial bar and pushes portfolio_history date_max forward.

Notes:
- Runnable: `python -m services.backtest_signals --verbose --threshold 0.01 --max-positions 10`
- Includes project-root bootstrap so `from services...` imports work on Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date as Date
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────
# Path bootstrap (REQUIRED when running as a script)
# ─────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ✅ Heartbeat writer (Phase 1.5)
from services.artifacts_writer import write_heartbeat  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join("data", "results")

SIGNALS_FILE = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
ASOF_DIAG_FILE = os.path.join(RESULTS_DIR, "signals_asof_diagnostics.json")

DASH_PORTFOLIO_HISTORY = os.path.join(RESULTS_DIR, "portfolio_history.csv")
DASH_TRADE_LOG = os.path.join(RESULTS_DIR, "trade_log.csv")

BT_TRADE_LOG = os.path.join(RESULTS_DIR, "backtest_trade_log.csv")
BT_PORTFOLIO_HISTORY = os.path.join(RESULTS_DIR, "backtest_portfolio_history.csv")
BT_SUMMARY = os.path.join(RESULTS_DIR, "backtest_summary.csv")
STRAT_VS_MARKET = os.path.join(RESULTS_DIR, "strategy_vs_market.csv")

# Dashboard portfolio_history contract
PH_COLS = ["date", "cash", "market_value", "total_value"]


# ─────────────────────────────────────────────────────────────
# Helpers: AS_OF policy
# ─────────────────────────────────────────────────────────────
def _parse_date_str(s: str) -> Optional[pd.Timestamp]:
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts).normalize()
    except Exception:
        return None


def _get_effective_asof(verbose: bool = False) -> Optional[pd.Timestamp]:
    """
    Prefer the effective_as_of_date produced by generate_signals.
    Falls back to market calendar if available; otherwise None.
    """
    # 1) Prefer signals_asof_diagnostics.json (source of truth)
    if os.path.exists(ASOF_DIAG_FILE) and os.path.getsize(ASOF_DIAG_FILE) > 0:
        try:
            with open(ASOF_DIAG_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            eff = d.get("effective_as_of_date") or d.get("session_as_of_date")
            ts = _parse_date_str(str(eff)) if eff else None
            if ts is not None:
                if verbose:
                    print(f"[backtest] effective_asof(from diagnostics)={ts.date().isoformat()}")
                return ts
        except Exception as e:
            if verbose:
                print(f"[backtest] WARN: could not read {ASOF_DIAG_FILE}: {e}")

    # 2) Fallback: try to use the same NYSE session helper if present
    try:
        from services.market_calendar import last_completed_nyse_session  # type: ignore

        d: Date = last_completed_nyse_session()
        ts = pd.Timestamp(d).normalize()
        if verbose:
            print(f"[backtest] effective_asof(from market_calendar)={ts.date().isoformat()}")
        return ts
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Load / Normalize
# ─────────────────────────────────────────────────────────────
def _must_load_signals(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise SystemExit(f"signals file missing or empty: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"signals file is empty: {path}")

    required_cols = {"date", "ticker", "close", "signal"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"signals missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["signal"] = df["signal"].astype(str).str.upper().str.strip()

    df = df.dropna(subset=["date", "ticker", "close", "signal"]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _add_strategy_vs_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df.groupby("ticker")["close"].pct_change()

    signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    df["signal_numeric"] = df["signal"].map(signal_map).fillna(0)

    # Position is yesterday's signal direction (simple hold model)
    df["position"] = df.groupby("ticker")["signal_numeric"].shift(1).fillna(0)

    df["strategy_return"] = df["returns"] * df["position"]
    df["cumulative_market"] = df.groupby("ticker")["returns"].transform(lambda x: (1 + x).cumprod())
    df["cumulative_strategy"] = df.groupby("ticker")["strategy_return"].transform(
        lambda x: (1 + x).cumprod()
    )
    return df


# ─────────────────────────────────────────────────────────────
# Portfolio history aggregation (dashboard-safe)
# ─────────────────────────────────────────────────────────────
def _build_dashboard_portfolio_history(per_ticker_ph: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-ticker portfolio_history (contains 'ticker') into a dashboard-safe curve.

    Because this script backtests each ticker independently (each starts at initial_balance),
    we aggregate by *mean* across tickers per date to keep the scale comparable to 1 account.

    Output columns: date,cash,market_value,total_value
    """
    if per_ticker_ph is None or per_ticker_ph.empty:
        return pd.DataFrame(columns=PH_COLS)

    df = per_ticker_ph.copy()
    if "date" not in df.columns:
        return pd.DataFrame(columns=PH_COLS)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=PH_COLS)

    for c in ("cash", "market_value", "total_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = (
        df.groupby(df["date"].dt.normalize(), sort=True)[["cash", "market_value", "total_value"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    out = out[PH_COLS]

    out["cash"] = out["cash"].round(2)
    out["market_value"] = out["market_value"].round(2)
    out["total_value"] = out["total_value"].round(2)
    return out


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print diagnostics.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="(Optional) unused placeholder for compatibility.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help="(Optional) unused placeholder for compatibility.",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Config
    initial_balance = 100_000.0
    sl_pct = 0.05  # 5% stop loss
    tp_pct = 0.05  # 5% take profit

    print("Running signal-based strategy backtest with REAL SL/TP exits...")

    # Load signals
    df = _must_load_signals(SIGNALS_FILE)

    # ✅ Clamp to effective_as_of_date to avoid AS_OF contract failures
    effective_asof = _get_effective_asof(verbose=args.verbose)
    if effective_asof is not None:
        df = df[df["date"].dt.normalize() <= effective_asof].copy()

    if df.empty:
        raise SystemExit("Backtest aborted: signals timeline empty after AS_OF clamp.")

    df = _add_strategy_vs_market_columns(df)

    if args.verbose:
        mn = df["date"].min()
        mx = df["date"].max()
        print(
            f"[backtest] asof={(effective_asof.date().isoformat() if effective_asof is not None else 'None')}"
        )
        print(f"[backtest] universe={df['ticker'].nunique()}")
        print(f"[backtest] date_range={mn.date().isoformat()} → {mx.date().isoformat()}")
        print(
            f"[backtest] dates={df['date'].dt.normalize().nunique()} tickers={df['ticker'].nunique()}"
        )

    trade_log: list[dict] = []
    portfolio_history: list[dict] = []
    summary: list[dict] = []

    tickers = df["ticker"].unique()

    for ticker in tickers:
        data = df[df["ticker"] == ticker].copy().reset_index(drop=True)
        if data.empty:
            continue

        balance = float(initial_balance)
        position = 0  # shares
        entry_price = 0.0
        entry_date: Optional[pd.Timestamp] = None

        trades = 0
        profit_total = 0.0

        for _, row in data.iterrows():
            dtm = row["date"]
            signal = str(row["signal"]).upper().strip()
            price = float(row["close"])

            if not (price > 0):
                continue

            # Active SL/TP only when in position
            stop_loss = (
                (entry_price * (1.0 - sl_pct)) if (position > 0 and entry_price > 0) else None
            )
            take_profit = (
                (entry_price * (1.0 + tp_pct)) if (position > 0 and entry_price > 0) else None
            )

            # 1) Priority: if in position, check SL/TP first
            if position > 0 and entry_price > 0:
                exit_reason = None
                if stop_loss is not None and price <= stop_loss:
                    exit_reason = "SL"
                elif take_profit is not None and price >= take_profit:
                    exit_reason = "TP"

                if exit_reason is not None:
                    exit_price = price
                    trade_profit = (exit_price - entry_price) * position
                    balance += position * exit_price
                    profit_total += trade_profit

                    holding_days = None
                    if entry_date is not None:
                        try:
                            holding_days = int((pd.Timestamp(dtm) - pd.Timestamp(entry_date)).days)
                        except Exception:
                            holding_days = None

                    trade_log.append(
                        {
                            "date": dtm,
                            "ticker": ticker,
                            "action": exit_reason,  # SL / TP
                            "price": exit_price,
                            "quantity": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "signal": signal,
                            "profit": trade_profit,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "exit_reason": exit_reason,
                            "holding_days": holding_days,
                        }
                    )

                    position = 0
                    entry_price = 0.0
                    entry_date = None

            # 2) If flat, allow BUY
            if signal == "BUY" and position == 0:
                quantity = int(balance // price)
                if quantity > 0:
                    position = quantity
                    entry_price = price
                    entry_date = dtm
                    balance -= quantity * price
                    trades += 1

                    trade_log.append(
                        {
                            "date": dtm,
                            "ticker": ticker,
                            "action": "BUY",
                            "price": price,
                            "quantity": quantity,
                            "entry_price": entry_price,
                            "exit_price": None,
                            "signal": "BUY",
                            "profit": None,
                            "stop_loss": entry_price * (1.0 - sl_pct),
                            "take_profit": entry_price * (1.0 + tp_pct),
                            "exit_reason": None,
                            "holding_days": 0,
                        }
                    )

            # 3) If still in position, allow SELL signal exit (non-SL/TP)
            elif signal == "SELL" and position > 0:
                exit_price = price
                trade_profit = (exit_price - entry_price) * position
                balance += position * exit_price
                profit_total += trade_profit

                holding_days = None
                if entry_date is not None:
                    try:
                        holding_days = int((pd.Timestamp(dtm) - pd.Timestamp(entry_date)).days)
                    except Exception:
                        holding_days = None

                trade_log.append(
                    {
                        "date": dtm,
                        "ticker": ticker,
                        "action": "SELL",
                        "price": exit_price,
                        "quantity": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "signal": "SELL",
                        "profit": trade_profit,
                        "stop_loss": entry_price * (1.0 - sl_pct),
                        "take_profit": entry_price * (1.0 + tp_pct),
                        "exit_reason": "SIGNAL",
                        "holding_days": holding_days,
                    }
                )

                position = 0
                entry_price = 0.0
                entry_date = None

            # Portfolio snapshot (every bar)
            market_value = position * price
            total_value = balance + market_value
            portfolio_history.append(
                {
                    "date": dtm,
                    "cash": balance,
                    "market_value": market_value,
                    "total_value": total_value,
                    "ticker": ticker,  # kept ONLY in backtest_portfolio_history.csv
                }
            )

        # Per-ticker summary
        final_price = float(data.iloc[-1]["close"])
        ending_value = balance + (position * final_price)
        total_return = (ending_value - initial_balance) / initial_balance * 100.0

        summary.append(
            {
                "ticker": ticker,
                "trades": trades,
                "profit": round(profit_total, 2),
                "final_value": round(ending_value, 2),
                "return_pct": round(total_return, 2),
            }
        )

    trade_log_df = pd.DataFrame(trade_log)
    per_ticker_ph_df = pd.DataFrame(portfolio_history)
    summary_df = pd.DataFrame(summary)

    # Normalize dates
    if not trade_log_df.empty and "date" in trade_log_df.columns:
        trade_log_df["date"] = pd.to_datetime(trade_log_df["date"], errors="coerce")
    if not per_ticker_ph_df.empty and "date" in per_ticker_ph_df.columns:
        per_ticker_ph_df["date"] = pd.to_datetime(per_ticker_ph_df["date"], errors="coerce")

    # Safety: clamp outputs again (in case anything slipped)
    if effective_asof is not None:
        if not trade_log_df.empty:
            trade_log_df = trade_log_df[
                trade_log_df["date"].dt.normalize() <= effective_asof
            ].copy()
        if not per_ticker_ph_df.empty:
            per_ticker_ph_df = per_ticker_ph_df[
                per_ticker_ph_df["date"].dt.normalize() <= effective_asof
            ].copy()

    # Write backtest artifacts
    trade_log_df.to_csv(BT_TRADE_LOG, index=False)
    per_ticker_ph_df.to_csv(BT_PORTFOLIO_HISTORY, index=False)
    summary_df.to_csv(BT_SUMMARY, index=False)
    df.to_csv(STRAT_VS_MARKET, index=False)

    # Dashboard outputs
    dash_ph_df = _build_dashboard_portfolio_history(per_ticker_ph_df)
    trade_log_df.to_csv(DASH_TRADE_LOG, index=False)
    dash_ph_df.to_csv(DASH_PORTFOLIO_HISTORY, index=False)

    if args.verbose:
        print(
            f"[backtest] trade_log_rows={len(trade_log_df)} portfolio_history_rows={len(dash_ph_df)}"
        )

    print("Backtest completed. Saved:")
    print("  - backtest_trade_log.csv")
    print("  - backtest_portfolio_history.csv")
    print("  - backtest_summary.csv")
    print("  - strategy_vs_market.csv")
    print("  - trade_log.csv (dashboard)")
    print("  - portfolio_history.csv (dashboard, 4 cols)")

    write_heartbeat(
        status="ok",
        stage="backtest",
        last_success_stage="backtest",
        message="Backtest complete (SL/TP exits active, AS_OF clamped).",
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as e:
        try:
            write_heartbeat(
                status="fail",
                stage="backtest",
                last_success_stage="backtest",
                message="Backtest failed.",
                error=str(e),
            )
        except Exception:
            pass
        raise
