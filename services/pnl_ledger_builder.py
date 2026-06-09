"""
TRITON Phase 164 — Profit & Loss Ledger Builder (read-only analytics).

Aggregates realized/unrealized P&L, cash flow, attribution, exit analysis, and
equity curves from local artifacts. Does NOT modify trading, execution, or
lifecycle logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
PROCESSED = ROOT / "data" / "processed"
LIVE = ROOT / "data" / "live"

# Sector / asset-class maps (analytics display only)
SECTOR_ETFS: Dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication",
}
BROAD_ETFS = {
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "VTI",
    "VOO",
    "ARKK",
}
COMMODITY_TICKERS = {"GLD", "SLV", "USO", "UNG", "DBA"}
CRYPTO_TICKERS = {"GBTC", "BITO"}
INDEX_PREFIX = "^"

_PRICE_CACHE: Dict[str, pd.DataFrame] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _to_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _parse_ts(x: Any) -> pd.Timestamp:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return pd.NaT
    return pd.to_datetime(s, errors="coerce", utc=True)


def _classify_sector(ticker: str) -> str:
    t = _norm_sym(ticker)
    if t in SECTOR_ETFS:
        return SECTOR_ETFS[t]
    if t.startswith(INDEX_PREFIX):
        return "Index"
    if t in COMMODITY_TICKERS:
        return "Commodities"
    if t in CRYPTO_TICKERS:
        return "Digital Assets"
    if t in BROAD_ETFS:
        return "Broad Market ETF"
    return "Equity"


def _classify_asset_class(ticker: str) -> str:
    t = _norm_sym(ticker)
    if t.startswith(INDEX_PREFIX):
        return "INDEX"
    if t in COMMODITY_TICKERS:
        return "COMMODITY"
    if t in CRYPTO_TICKERS:
        return "CRYPTO"
    if t in SECTOR_ETFS or t in BROAD_ETFS:
        return "ETF"
    return "EQUITY"


def _fingerprint(paths: List[Path]) -> str:
    parts: List[str] = []
    for p in paths:
        try:
            if p.is_file():
                st = p.stat()
                parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
            else:
                parts.append(f"{p.name}:missing")
        except OSError:
            parts.append(f"{p.name}:err")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _portfolio_history_paths() -> List[Path]:
    return [
        RESULTS / "portfolio_history.csv",
        RESULTS / "enhanced_portfolio_history.csv",
        RESULTS / "backtest_portfolio_history.csv",
    ]


def _input_paths() -> List[Path]:
    return [
        RESULTS / "trade_outcomes.csv",
        RESULTS / "trade_log.csv",
        RESULTS / "portfolio_history.csv",
        RESULTS / "enhanced_portfolio_history.csv",
        RESULTS / "backtest_portfolio_history.csv",
        RESULTS / "positions_snapshot.csv",
        RESULTS / "performance_intelligence.csv",
        RESULTS / "performance_intelligence_by_symbol.csv",
        RESULTS / "performance_intelligence_summary.json",
        RESULTS / "pnl_diagnostics.json",
        RESULTS / "pnl_diagnostics_by_symbol.csv",
        RESULTS / "trade_outcomes_by_symbol.csv",
        RESULTS / "live_orders_log.csv",
        RESULTS / "recent_orders.csv",
        RESULTS / "manage_positions_plan.csv",
        RESULTS / "signal_lifecycle_effective.csv",
        RESULTS / "capital_preservation_certification.json",
        RESULTS / "capital_preservation_scorecard.csv",
        LIVE / "manage_orders.csv",
        LIVE / "placement_session_intents.csv",
    ]


def _load_price_series(ticker: str) -> pd.DataFrame:
    sym = _norm_sym(ticker)
    if sym in _PRICE_CACHE:
        return _PRICE_CACHE[sym]
    path = PROCESSED / f"{sym}.parquet"
    if not path.is_file():
        _PRICE_CACHE[sym] = pd.DataFrame()
        return _PRICE_CACHE[sym]
    try:
        df = pd.read_parquet(path, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        _PRICE_CACHE[sym] = df
        return df
    except Exception:
        _PRICE_CACHE[sym] = pd.DataFrame()
        return _PRICE_CACHE[sym]


def _post_exit_return(
    ticker: str, exit_ts: pd.Timestamp, exit_price: float, days: int = 10
) -> float:
    if not np.isfinite(exit_price) or exit_price <= 0 or pd.isna(exit_ts):
        return np.nan
    prices = _load_price_series(ticker)
    if prices.empty:
        return np.nan
    exit_day = exit_ts.tz_localize(None) if exit_ts.tzinfo else exit_ts
    future = prices[prices["date"] > exit_day.normalize()].head(days)
    if future.empty:
        return np.nan
    px = float(future["close"].iloc[-1])
    return (px - exit_price) / exit_price


def _grade_exit(realized_pl: float, post_ret: float) -> str:
    if not np.isfinite(post_ret):
        return "C"
    if realized_pl >= 0:
        if post_ret <= -0.02:
            return "A"
        if post_ret >= 0.05:
            return "D"
        if abs(post_ret) <= 0.02:
            return "C"
        return "B"
    if post_ret <= -0.02:
        return "A"
    if post_ret >= 0.05:
        return "F"
    if abs(post_ret) <= 0.02:
        return "C"
    return "B"


def _holding_days(opened: pd.Timestamp, closed: pd.Timestamp) -> float:
    if pd.isna(opened) or pd.isna(closed):
        return np.nan
    o = opened.tz_localize(None) if opened.tzinfo else opened
    c = closed.tz_localize(None) if closed.tzinfo else closed
    return max((c - o).total_seconds() / 86400.0, 0.0)


def _build_exit_reason_map() -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    tl = _safe_read_csv(RESULTS / "trade_log.csv")
    if not tl.empty:
        tl.columns = [str(c).strip().lower() for c in tl.columns]
        for _, r in tl.iterrows():
            sym = _norm_sym(r.get("ticker") or r.get("symbol"))
            act = str(r.get("action", "")).upper()
            if act not in ("SELL", "EXIT", "CLOSE"):
                continue
            dt = _parse_ts(r.get("date"))
            reason = str(r.get("exit_reason") or r.get("reason") or "").strip()
            if sym and reason and not pd.isna(dt):
                out[(sym, dt.strftime("%Y-%m-%d"))] = reason
    mp = _safe_read_csv(RESULTS / "manage_positions_plan.csv")
    if not mp.empty:
        mp.columns = [str(c).strip().lower() for c in mp.columns]
        for _, r in mp.iterrows():
            sym = _norm_sym(r.get("symbol") or r.get("ticker"))
            action = str(
                r.get("final_action") or r.get("management_action_candidate") or ""
            ).strip()
            reason = str(r.get("reason_code") or r.get("reason_detail") or action).strip()
            if sym and reason:
                out.setdefault((sym, ""), reason)
    return out


def _build_session_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in (
        RESULTS / "live_orders_log.csv",
        RESULTS / "execution_intelligence.csv",
    ):
        df = _safe_read_csv(path)
        if df.empty:
            continue
        df.columns = [str(c).strip().lower() for c in df.columns]
        sym_col = (
            "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
        )
        if sym_col is None or "session" not in df.columns:
            continue
        for _, r in df.iterrows():
            sym = _norm_sym(r.get(sym_col))
            sess = str(r.get("session") or "").strip()
            if sym and sess:
                out[sym] = sess
    return out


def _build_lifecycle_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    lc = _safe_read_csv(RESULTS / "signal_lifecycle_effective.csv")
    if lc.empty:
        lc = _safe_read_csv(RESULTS / "signal_lifecycle.csv")
    if lc.empty:
        return out
    lc.columns = [str(c).strip().lower() for c in lc.columns]
    sym_col = "ticker" if "ticker" in lc.columns else ("symbol" if "symbol" in lc.columns else None)
    if sym_col is None:
        return out
    act_col = "lifecycle_action" if "lifecycle_action" in lc.columns else "stance"
    for _, r in lc.iterrows():
        sym = _norm_sym(r.get(sym_col))
        act = str(r.get(act_col) or "").strip()
        if sym and act:
            out[sym] = act
    return out


def _build_completed_ledger() -> pd.DataFrame:
    cols = [
        "trade_id",
        "ticker",
        "entry_date",
        "exit_date",
        "holding_days",
        "side",
        "quantity",
        "entry_price",
        "exit_price",
        "gross_pnl",
        "gross_pnl_pct",
        "fees",
        "net_pnl",
        "outcome",
        "exit_reason",
        "lifecycle_action",
        "execution_session",
    ]
    empty = pd.DataFrame(columns=cols)

    to = _safe_read_csv(RESULTS / "trade_outcomes.csv")
    exit_map = _build_exit_reason_map()
    session_map = _build_session_map()
    lifecycle_map = _build_lifecycle_map()

    rows: List[Dict[str, Any]] = []

    if not to.empty:
        to.columns = [str(c).strip().lower() for c in to.columns]
        realized = to[
            (to.get("outcome_type", pd.Series(dtype=str)).astype(str).str.upper() == "REALIZED")
            | (to.get("holding_status", pd.Series(dtype=str)).astype(str).str.upper() == "CLOSED")
        ].copy()
        realized = realized[pd.to_numeric(realized.get("realized_pl"), errors="coerce").notna()]
        for i, r in realized.iterrows():
            sym = _norm_sym(r.get("symbol"))
            opened = _parse_ts(r.get("opened_at"))
            closed = _parse_ts(r.get("closed_at"))
            qty = _to_float(r.get("qty"))
            entry_px = _to_float(r.get("entry_price"))
            exit_px = _to_float(r.get("exit_price"))
            gross = _to_float(r.get("realized_pl"))
            gross_pct = _to_float(r.get("return_pct"))
            fees = 0.0
            net = gross - fees if np.isfinite(gross) else np.nan
            exit_day = closed.strftime("%Y-%m-%d") if not pd.isna(closed) else ""
            exit_reason = exit_map.get((sym, exit_day), exit_map.get((sym, ""), ""))
            if not exit_reason:
                notes = str(r.get("notes") or "")
                if "exit" in notes.lower():
                    exit_reason = notes[:120]
            trade_id = hashlib.md5(f"{sym}|{closed}|{qty}|{i}".encode()).hexdigest()[:12].upper()
            rows.append(
                {
                    "trade_id": trade_id,
                    "ticker": sym,
                    "entry_date": opened,
                    "exit_date": closed,
                    "holding_days": _holding_days(opened, closed),
                    "side": "LONG",
                    "quantity": qty,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "gross_pnl": gross,
                    "gross_pnl_pct": gross_pct,
                    "fees": fees,
                    "net_pnl": net,
                    "outcome": ("WIN" if gross > 0 else ("LOSS" if gross < 0 else "FLAT")),
                    "exit_reason": exit_reason or "—",
                    "lifecycle_action": lifecycle_map.get(sym, "—"),
                    "execution_session": session_map.get(sym, "—"),
                }
            )

    # Supplement from trade_log exits when trade_outcomes is sparse
    if len(rows) < 5:
        tl = _safe_read_csv(RESULTS / "trade_log.csv")
        if not tl.empty:
            tl.columns = [str(c).strip().lower() for c in tl.columns]
            pnl_col = (
                "pnl" if "pnl" in tl.columns else ("profit" if "profit" in tl.columns else None)
            )
            if pnl_col:
                exits = tl[tl[pnl_col].notna()].copy()
                for i, r in exits.iterrows():
                    sym = _norm_sym(r.get("ticker") or r.get("symbol"))
                    gross = _to_float(r.get(pnl_col))
                    if not sym or not np.isfinite(gross):
                        continue
                    closed = _parse_ts(r.get("date"))
                    entry_px = _to_float(r.get("entry_price") or r.get("price"))
                    exit_px = _to_float(r.get("exit_price") or r.get("price"))
                    qty = _to_float(r.get("quantity") or r.get("qty"))
                    hd = _to_float(r.get("holding_days"))
                    if not np.isfinite(hd):
                        hd = np.nan
                    trade_id = (
                        hashlib.md5(f"tl|{sym}|{closed}|{i}".encode()).hexdigest()[:12].upper()
                    )
                    rows.append(
                        {
                            "trade_id": trade_id,
                            "ticker": sym,
                            "entry_date": pd.NaT,
                            "exit_date": closed,
                            "holding_days": hd,
                            "side": str(r.get("action") or "LONG").upper(),
                            "quantity": qty,
                            "entry_price": entry_px,
                            "exit_price": exit_px,
                            "gross_pnl": gross,
                            "gross_pnl_pct": (
                                (gross / (entry_px * qty) * 100.0)
                                if np.isfinite(entry_px) and np.isfinite(qty) and entry_px * qty
                                else np.nan
                            ),
                            "fees": 0.0,
                            "net_pnl": gross,
                            "outcome": ("WIN" if gross > 0 else ("LOSS" if gross < 0 else "FLAT")),
                            "exit_reason": str(r.get("exit_reason") or "—"),
                            "lifecycle_action": lifecycle_map.get(sym, "—"),
                            "execution_session": session_map.get(sym, "—"),
                        }
                    )

    if not rows:
        return empty

    df = pd.DataFrame(rows)
    df = df.sort_values("exit_date", na_position="last").reset_index(drop=True)
    return df


def _ledger_metrics(ledger: pd.DataFrame) -> Dict[str, Any]:
    if ledger.empty:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": np.nan,
            "avg_winner": np.nan,
            "avg_loser": np.nan,
            "profit_factor": np.nan,
            "largest_winner": np.nan,
            "largest_loser": np.nan,
        }
    pnl = pd.to_numeric(ledger["net_pnl"], errors="coerce")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_sum = float(wins.sum()) if len(wins) else 0.0
    loss_sum = abs(float(losses.sum())) if len(losses) else 0.0
    pf = win_sum / loss_sum if loss_sum > 0 else np.nan
    total = int(pnl.notna().sum())
    return {
        "total_trades": total,
        "winning_trades": int((pnl > 0).sum()),
        "losing_trades": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0) if total else np.nan,
        "avg_winner": float(wins.mean()) if len(wins) else np.nan,
        "avg_loser": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": pf,
        "largest_winner": float(wins.max()) if len(wins) else np.nan,
        "largest_loser": float(losses.min()) if len(losses) else np.nan,
    }


def _realized_performance(
    ledger: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    summary = {
        "total_realized_pnl": 0.0,
        "daily_realized_pnl": 0.0,
        "weekly_realized_pnl": 0.0,
        "monthly_realized_pnl": 0.0,
        "ytd_realized_pnl": 0.0,
    }
    daily_cols = ["date", "daily_pnl", "cumulative_pnl", "trade_count"]
    if ledger.empty:
        return summary, pd.DataFrame(columns=daily_cols)

    df = ledger.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce", utc=True)
    df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce")
    df = df.dropna(subset=["exit_date", "net_pnl"])
    if df.empty:
        return summary, pd.DataFrame(columns=daily_cols)

    df["date"] = df["exit_date"].dt.tz_convert(None).dt.normalize()
    daily = df.groupby("date", as_index=False).agg(
        daily_pnl=("net_pnl", "sum"), trade_count=("net_pnl", "count")
    )
    daily = daily.sort_values("date")
    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()

    now = pd.Timestamp.now(tz=None).normalize()
    summary["total_realized_pnl"] = float(daily["daily_pnl"].sum())
    today = daily[daily["date"] == now]
    summary["daily_realized_pnl"] = float(today["daily_pnl"].sum()) if not today.empty else 0.0
    week_start = now - pd.Timedelta(days=now.weekday())
    summary["weekly_realized_pnl"] = float(
        daily[(daily["date"] >= week_start) & (daily["date"] <= now)]["daily_pnl"].sum()
    )
    month_start = now.replace(day=1)
    summary["monthly_realized_pnl"] = float(
        daily[(daily["date"] >= month_start) & (daily["date"] <= now)]["daily_pnl"].sum()
    )
    ytd_start = now.replace(month=1, day=1)
    summary["ytd_realized_pnl"] = float(
        daily[(daily["date"] >= ytd_start) & (daily["date"] <= now)]["daily_pnl"].sum()
    )
    return summary, daily


def _open_positions() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = [
        "ticker",
        "quantity",
        "market_value",
        "cost_basis",
        "current_price",
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "position_weight",
        "days_held",
    ]
    empty = pd.DataFrame(columns=cols)
    metrics = {
        "total_unrealized_pnl": 0.0,
        "largest_winner": np.nan,
        "largest_loser": np.nan,
    }

    snap = _safe_read_csv(RESULTS / "positions_snapshot.csv")
    rows: List[Dict[str, Any]] = []

    if not snap.empty:
        snap.columns = [str(c).strip().lower() for c in snap.columns]
        sym_col = "ticker" if "ticker" in snap.columns else "symbol"
        for _, r in snap.iterrows():
            sym = _norm_sym(r.get(sym_col))
            qty = _to_float(r.get("qty") or r.get("qty_available"))
            if not sym or not np.isfinite(qty) or abs(qty) < 1e-9:
                continue
            mv = _to_float(r.get("market_value") or r.get("value"))
            cb = _to_float(r.get("cost_basis"))
            if not np.isfinite(cb) and np.isfinite(_to_float(r.get("avg_entry_price"))):
                cb = _to_float(r.get("avg_entry_price")) * abs(qty)
            px = _to_float(r.get("current_price") or r.get("lastday_price"))
            upl = _to_float(r.get("unrealized_pl") or r.get("unrealized_pnl"))
            upl_pct = _to_float(r.get("unrealized_plpc"))
            if not np.isfinite(upl_pct) and np.isfinite(upl) and np.isfinite(cb) and cb:
                upl_pct = upl / cb
            opened = _parse_ts(r.get("snapshot_ts") or r.get("date"))
            days = np.nan
            if not pd.isna(opened):
                days = (pd.Timestamp.now(tz=timezone.utc) - opened).total_seconds() / 86400.0
            rows.append(
                {
                    "ticker": sym,
                    "quantity": qty,
                    "market_value": mv,
                    "cost_basis": cb,
                    "current_price": px,
                    "unrealized_pnl": upl,
                    "unrealized_pnl_pct": (
                        upl_pct * 100.0 if np.isfinite(upl_pct) and abs(upl_pct) < 5 else upl_pct
                    ),
                    "position_weight": np.nan,
                    "days_held": days,
                }
            )

    if not rows:
        to = _safe_read_csv(RESULTS / "trade_outcomes.csv")
        if not to.empty:
            to.columns = [str(c).strip().lower() for c in to.columns]
            open_rows = to[
                to.get("outcome_type", pd.Series(dtype=str)).astype(str).str.upper() == "OPEN"
            ]
            for _, r in open_rows.iterrows():
                sym = _norm_sym(r.get("symbol"))
                qty = _to_float(r.get("qty"))
                entry = _to_float(r.get("entry_price"))
                mkt = _to_float(r.get("exit_price"))
                upl = _to_float(r.get("unrealized_pl") or r.get("total_pl"))
                mv = mkt * qty if np.isfinite(mkt) and np.isfinite(qty) else np.nan
                cb = entry * qty if np.isfinite(entry) and np.isfinite(qty) else np.nan
                opened = _parse_ts(r.get("opened_at"))
                days = (
                    _holding_days(opened, pd.Timestamp.now(tz=timezone.utc))
                    if not pd.isna(opened)
                    else np.nan
                )
                rows.append(
                    {
                        "ticker": sym,
                        "quantity": qty,
                        "market_value": mv,
                        "cost_basis": cb,
                        "current_price": mkt,
                        "unrealized_pnl": upl,
                        "unrealized_pnl_pct": _to_float(r.get("return_pct")),
                        "position_weight": np.nan,
                        "days_held": days,
                    }
                )

    if not rows:
        return empty, metrics

    df = pd.DataFrame(rows)
    total_mv = float(pd.to_numeric(df["market_value"], errors="coerce").sum())
    if total_mv > 0:
        df["position_weight"] = (
            pd.to_numeric(df["market_value"], errors="coerce") / total_mv * 100.0
        )

    upl = pd.to_numeric(df["unrealized_pnl"], errors="coerce")
    metrics["total_unrealized_pnl"] = float(upl.sum()) if upl.notna().any() else 0.0
    if upl.notna().any():
        metrics["largest_winner"] = float(upl.max())
        metrics["largest_loser"] = float(upl.min())
    return df.sort_values("market_value", ascending=False, na_position="last"), metrics


def _load_portfolio_daily() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in _portfolio_history_paths():
        df = _safe_read_csv(path)
        if df.empty or "date" not in df.columns:
            continue
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        for c in ("cash", "market_value", "total_value"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "ticker" in df.columns:
            if "total_value" in df.columns:
                daily = df.groupby("date", as_index=False).agg(
                    total_value=("total_value", "max"),
                    cash=(("cash", "max") if "cash" in df.columns else ("total_value", "first")),
                    market_value=(
                        ("market_value", "sum")
                        if "market_value" in df.columns
                        else ("total_value", "first")
                    ),
                )
            elif "market_value" in df.columns:
                daily = df.groupby("date", as_index=False).agg(
                    market_value=("market_value", "sum"),
                    cash=(("cash", "max") if "cash" in df.columns else ("market_value", "first")),
                )
                daily["total_value"] = daily["market_value"] + daily.get("cash", 0)
            else:
                continue
        else:
            daily = df[
                ["date"] + [c for c in ("cash", "market_value", "total_value") if c in df.columns]
            ].copy()
        frames.append(daily)
    if not frames:
        return pd.DataFrame(columns=["date", "cash", "market_value", "total_value"])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def _cash_flow(
    ledger: pd.DataFrame, portfolio: pd.DataFrame
) -> Tuple[Dict[str, float], pd.DataFrame]:
    summary = {
        "opening_cash": np.nan,
        "capital_deployed": 0.0,
        "capital_returned": 0.0,
        "realized_gains": 0.0,
        "realized_losses": 0.0,
        "cash_available": np.nan,
        "current_equity": np.nan,
        "net_account_change": np.nan,
    }
    series_cols = [
        "date",
        "cash",
        "capital_deployed_cum",
        "capital_returned_cum",
        "equity",
    ]
    if portfolio.empty and ledger.empty:
        return summary, pd.DataFrame(columns=series_cols)

    tl = _safe_read_csv(RESULTS / "trade_log.csv")
    deployed = 0.0
    returned = 0.0
    if not tl.empty:
        tl.columns = [str(c).strip().lower() for c in tl.columns]
        for _, r in tl.iterrows():
            act = str(r.get("action", "")).upper()
            px = _to_float(r.get("price"))
            qty = _to_float(r.get("quantity") or r.get("qty"))
            notional = px * qty if np.isfinite(px) and np.isfinite(qty) else np.nan
            if act == "BUY" and np.isfinite(notional):
                deployed += notional
            elif act in ("SELL", "EXIT") and np.isfinite(notional):
                returned += notional

    if not ledger.empty:
        pnl = pd.to_numeric(ledger["net_pnl"], errors="coerce")
        summary["realized_gains"] = float(pnl[pnl > 0].sum())
        summary["realized_losses"] = float(pnl[pnl < 0].sum())

    summary["capital_deployed"] = deployed
    summary["capital_returned"] = returned

    if not portfolio.empty:
        first = portfolio.iloc[0]
        last = portfolio.iloc[-1]
        summary["opening_cash"] = _to_float(first.get("cash"))
        summary["cash_available"] = _to_float(last.get("cash"))
        summary["current_equity"] = _to_float(last.get("total_value"))
        if np.isfinite(summary["opening_cash"]) and np.isfinite(summary["current_equity"]):
            summary["net_account_change"] = summary["current_equity"] - _to_float(
                first.get("total_value", summary["opening_cash"])
            )

    cf = portfolio.copy() if not portfolio.empty else pd.DataFrame({"date": [pd.Timestamp.today()]})
    if "cash" not in cf.columns:
        cf["cash"] = np.nan
    cf["equity"] = pd.to_numeric(cf.get("total_value"), errors="coerce")
    cf["capital_deployed_cum"] = deployed
    cf["capital_returned_cum"] = returned
    return summary, cf[series_cols] if all(c in cf.columns for c in series_cols) else cf


def _attribution(ledger: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "top_winners": pd.DataFrame(),
        "top_losers": pd.DataFrame(),
        "best_sector": "—",
        "worst_sector": "—",
        "best_asset_class": "—",
        "worst_asset_class": "—",
        "by_symbol": pd.DataFrame(),
        "by_strategy": pd.DataFrame(),
        "by_exit_reason": pd.DataFrame(),
    }
    if ledger.empty:
        return out

    df = ledger.copy()
    df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce")
    df = df.dropna(subset=["net_pnl"])
    if df.empty:
        return out

    by_sym = df.groupby("ticker", as_index=False).agg(
        net_pnl=("net_pnl", "sum"), trades=("net_pnl", "count")
    )
    by_sym = by_sym.sort_values("net_pnl", ascending=False)
    out["by_symbol"] = by_sym
    out["top_winners"] = by_sym.head(10)
    out["top_losers"] = by_sym.sort_values("net_pnl").head(10)

    df["sector"] = df["ticker"].map(_classify_sector)
    df["asset_class"] = df["ticker"].map(_classify_asset_class)
    sec = df.groupby("sector", as_index=False).agg(net_pnl=("net_pnl", "sum"))
    if not sec.empty:
        out["best_sector"] = str(sec.sort_values("net_pnl", ascending=False).iloc[0]["sector"])
        out["worst_sector"] = str(sec.sort_values("net_pnl").iloc[0]["sector"])
    ac = df.groupby("asset_class", as_index=False).agg(net_pnl=("net_pnl", "sum"))
    if not ac.empty:
        out["best_asset_class"] = str(
            ac.sort_values("net_pnl", ascending=False).iloc[0]["asset_class"]
        )
        out["worst_asset_class"] = str(ac.sort_values("net_pnl").iloc[0]["asset_class"])

    strat_col = "lifecycle_action" if "lifecycle_action" in df.columns else None
    if strat_col:
        out["by_strategy"] = (
            df.groupby(strat_col, as_index=False)
            .agg(net_pnl=("net_pnl", "sum"), trades=("net_pnl", "count"))
            .sort_values("net_pnl", ascending=False)
        )
    if "exit_reason" in df.columns:
        out["by_exit_reason"] = (
            df.groupby("exit_reason", as_index=False)
            .agg(net_pnl=("net_pnl", "sum"), trades=("net_pnl", "count"))
            .sort_values("net_pnl", ascending=False)
        )
    return out


def _exit_analysis(ledger: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = [
        "ticker",
        "exit_date",
        "exit_reason",
        "realized_pnl",
        "post_exit_performance_pct",
        "would_holding_be_better",
        "decision_grade",
    ]
    metrics = {
        "average_exit_grade": "—",
        "exit_accuracy_pct": np.nan,
        "capital_saved": 0.0,
        "capital_lost_premature": 0.0,
    }
    if ledger.empty:
        return pd.DataFrame(columns=cols), metrics

    grade_score = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    rows: List[Dict[str, Any]] = []
    sample = ledger.tail(500) if len(ledger) > 500 else ledger
    for _, r in sample.iterrows():
        sym = r["ticker"]
        closed = _parse_ts(r["exit_date"])
        exit_px = _to_float(r["exit_price"])
        realized = _to_float(r["net_pnl"])
        post = _post_exit_return(sym, closed, exit_px, days=10)
        post_pct = post * 100.0 if np.isfinite(post) else np.nan
        grade = _grade_exit(realized, post if np.isfinite(post) else np.nan)
        better = (
            "Yes"
            if (
                np.isfinite(post)
                and ((realized >= 0 and post > 0.02) or (realized < 0 and post > 0.05))
            )
            else (
                "No"
                if (
                    np.isfinite(post)
                    and ((realized >= 0 and post < -0.02) or (realized < 0 and post < -0.02))
                )
                else "Unclear"
            )
        )
        rows.append(
            {
                "ticker": sym,
                "exit_date": closed,
                "exit_reason": r.get("exit_reason", "—"),
                "realized_pnl": realized,
                "post_exit_performance_pct": post_pct,
                "would_holding_be_better": better,
                "decision_grade": grade,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, metrics

    grades = df["decision_grade"].astype(str)
    scores = grades.map(grade_score).dropna()
    if not scores.empty:
        avg = scores.mean()
        letter = "C"
        for g, s in sorted(grade_score.items(), key=lambda x: -x[1]):
            if avg >= s - 0.5:
                letter = g
                break
        metrics["average_exit_grade"] = letter
        metrics["exit_accuracy_pct"] = float((grades.isin(["A", "B"])).mean() * 100.0)

    for _, r in df.iterrows():
        g = r["decision_grade"]
        pnl = _to_float(r["realized_pnl"])
        post = _to_float(r["post_exit_performance_pct"]) / 100.0
        if g == "A" and np.isfinite(post) and post < 0:
            metrics["capital_saved"] += abs(pnl * post) if pnl < 0 else abs(pnl * min(post, 0.05))
        if g in ("D", "F") and np.isfinite(post) and post > 0:
            qty_px = abs(pnl) if np.isfinite(pnl) else 0.0
            metrics["capital_lost_premature"] += qty_px * post
    return df, metrics


def _equity_curve(
    portfolio: pd.DataFrame, daily_realized: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics = {
        "current_equity": np.nan,
        "peak_equity": np.nan,
        "current_drawdown_pct": np.nan,
        "max_drawdown_pct": np.nan,
        "recovery_factor": np.nan,
    }
    if portfolio.empty and daily_realized.empty:
        return (
            pd.DataFrame(columns=["date", "equity", "drawdown_pct", "rolling_return_30d"]),
            metrics,
        )

    if not portfolio.empty and "total_value" in portfolio.columns:
        eq = portfolio[["date", "total_value"]].copy()
        eq = eq.rename(columns={"total_value": "equity"})
    elif not daily_realized.empty:
        eq = daily_realized[["date", "cumulative_pnl"]].copy()
        eq = eq.rename(columns={"cumulative_pnl": "equity"})
    else:
        return (
            pd.DataFrame(columns=["date", "equity", "drawdown_pct", "rolling_return_30d"]),
            metrics,
        )

    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["equity"]).sort_values("date")
    if eq.empty:
        return eq, metrics

    peak = eq["equity"].cummax()
    eq["drawdown_pct"] = (eq["equity"] / peak - 1.0) * 100.0
    eq["rolling_return_30d"] = eq["equity"].pct_change(30) * 100.0

    cur = float(eq["equity"].iloc[-1])
    pk = float(peak.iloc[-1])
    max_dd = float(eq["drawdown_pct"].min())
    metrics["current_equity"] = cur
    metrics["peak_equity"] = pk
    metrics["current_drawdown_pct"] = float(eq["drawdown_pct"].iloc[-1])
    metrics["max_drawdown_pct"] = max_dd
    if max_dd < 0:
        total_ret = cur - float(eq["equity"].iloc[0])
        metrics["recovery_factor"] = total_ret / abs(max_dd / 100.0 * pk) if pk else np.nan
    return eq, metrics


def _operator_scorecard(
    ledger: pd.DataFrame,
    open_df: pd.DataFrame,
    ledger_metrics: Dict[str, Any],
    pi_summary: Dict[str, Any],
) -> Dict[str, Any]:
    realized = (
        float(pd.to_numeric(ledger.get("net_pnl"), errors="coerce").sum())
        if not ledger.empty
        else 0.0
    )
    unrealized = (
        float(pd.to_numeric(open_df.get("unrealized_pnl"), errors="coerce").sum())
        if not open_df.empty
        else 0.0
    )
    best_trade = worst_trade = most_prof = least_prof = "—"
    if not ledger.empty:
        pnl = pd.to_numeric(ledger["net_pnl"], errors="coerce")
        idx_max = pnl.idxmax() if pnl.notna().any() else None
        idx_min = pnl.idxmin() if pnl.notna().any() else None
        if idx_max is not None and np.isfinite(pnl.loc[idx_max]):
            r = ledger.loc[idx_max]
            best_trade = f"{r['ticker']} ${pnl.loc[idx_max]:,.2f}"
        if idx_min is not None and np.isfinite(pnl.loc[idx_min]):
            r = ledger.loc[idx_min]
            worst_trade = f"{r['ticker']} ${pnl.loc[idx_min]:,.2f}"
        by_sym = ledger.groupby("ticker")["net_pnl"].sum()
        if not by_sym.empty:
            most_prof = f"{by_sym.idxmax()} ${by_sym.max():,.2f}"
            least_prof = f"{by_sym.idxmin()} ${by_sym.min():,.2f}"

    cert = _safe_read_json(RESULTS / "capital_preservation_certification.json")
    cps = _to_float(cert.get("certification_score"), np.nan)

    deployed = _to_float(pi_summary.get("capital_deployed"), np.nan)
    combined = realized + unrealized
    cap_eff = combined / deployed if np.isfinite(deployed) and deployed > 0 else np.nan

    return {
        "total_realized_pnl": realized,
        "total_unrealized_pnl": unrealized,
        "combined_pnl": combined,
        "win_rate_pct": ledger_metrics.get("win_rate_pct"),
        "profit_factor": ledger_metrics.get("profit_factor"),
        "sharpe_ratio": _to_float(pi_summary.get("sharpe_ratio"), np.nan),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "most_profitable_symbol": most_prof,
        "least_profitable_symbol": least_prof,
        "capital_efficiency": cap_eff,
        "portfolio_health": str(
            pi_summary.get("portfolio_health") or cert.get("certification_status") or "—"
        ),
        "current_cps": cps,
    }


@dataclass
class PnlLedgerBundle:
    fingerprint: str = ""
    generated_at: str = ""
    ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    ledger_metrics: Dict[str, Any] = field(default_factory=dict)
    realized_summary: Dict[str, float] = field(default_factory=dict)
    daily_realized: pd.DataFrame = field(default_factory=pd.DataFrame)
    open_positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    open_metrics: Dict[str, Any] = field(default_factory=dict)
    cash_flow_summary: Dict[str, float] = field(default_factory=dict)
    cash_flow_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    attribution: Dict[str, Any] = field(default_factory=dict)
    exit_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)
    exit_metrics: Dict[str, Any] = field(default_factory=dict)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_metrics: Dict[str, Any] = field(default_factory=dict)
    scorecard: Dict[str, Any] = field(default_factory=dict)
    has_realized_trades: bool = False
    has_open_positions: bool = False
    has_pnl_history: bool = False


_BUILD_CACHE: Dict[str, PnlLedgerBundle] = {}


def build_pnl_ledger(*, results_dir: Optional[Path] = None, force: bool = False) -> PnlLedgerBundle:
    global RESULTS
    if results_dir is not None:
        RESULTS = Path(results_dir)
    fp = _fingerprint(_input_paths())
    if not force and fp in _BUILD_CACHE:
        return _BUILD_CACHE[fp]

    _PRICE_CACHE.clear()
    ledger = _build_completed_ledger()
    ledger_metrics = _ledger_metrics(ledger)
    realized_summary, daily_realized = _realized_performance(ledger)
    open_positions, open_metrics = _open_positions()
    portfolio = _load_portfolio_daily()
    cash_flow_summary, cash_flow_series = _cash_flow(ledger, portfolio)
    attribution = _attribution(ledger)
    exit_analysis, exit_metrics = _exit_analysis(ledger)
    equity_curve, equity_metrics = _equity_curve(portfolio, daily_realized)
    pi_summary = _safe_read_json(RESULTS / "performance_intelligence_summary.json")
    if not pi_summary:
        pi_csv = _safe_read_csv(RESULTS / "performance_intelligence.csv")
        if not pi_csv.empty:
            pi_summary = pi_csv.iloc[-1].to_dict()
    pnl_diag = _safe_read_json(RESULTS / "pnl_diagnostics.json")
    pi_summary = {**pnl_diag, **pi_summary}
    scorecard = _operator_scorecard(ledger, open_positions, ledger_metrics, pi_summary)

    bundle = PnlLedgerBundle(
        fingerprint=fp,
        generated_at=_now_iso(),
        ledger=ledger,
        ledger_metrics=ledger_metrics,
        realized_summary=realized_summary,
        daily_realized=daily_realized,
        open_positions=open_positions,
        open_metrics=open_metrics,
        cash_flow_summary=cash_flow_summary,
        cash_flow_series=cash_flow_series,
        attribution=attribution,
        exit_analysis=exit_analysis,
        exit_metrics=exit_metrics,
        equity_curve=equity_curve,
        equity_metrics=equity_metrics,
        scorecard=scorecard,
        has_realized_trades=not ledger.empty,
        has_open_positions=not open_positions.empty,
        has_pnl_history=not daily_realized.empty or not portfolio.empty,
    )
    _BUILD_CACHE[fp] = bundle
    return bundle


if __name__ == "__main__":
    b = build_pnl_ledger(force=True)
    print(f"Ledger trades: {len(b.ledger)}")
    print(f"Open positions: {len(b.open_positions)}")
    print(f"Realized P&L: {b.realized_summary.get('total_realized_pnl')}")
