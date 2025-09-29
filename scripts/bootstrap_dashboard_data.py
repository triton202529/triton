# scripts/bootstrap_dashboard_data.py
# Create missing CSVs/JSONL the Streamlit dashboard expects.
# It pulls from whatever you already have (trade_log, signals, fundamentals, portfolio_history)
# and fills the rest with lightweight, sensible placeholders.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


# ---------- paths ----------
THIS = Path(__file__).resolve()
CANDIDATES = [THIS.parent, THIS.parent.parent, Path.cwd()]
PROJECT_ROOT = None
for p in CANDIDATES:
    if (p / "data").exists():
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path.cwd()

DATA = PROJECT_ROOT / "data"
RESULTS = DATA / "results"
PRED = DATA / "predictions"
PROC = DATA / "processed"
ORDERS = DATA / "orders"
SERVICES_RESULTS = PROJECT_ROOT / "services" / "data" / "results"

for d in (RESULTS, PRED, PROC, ORDERS):
    d.mkdir(parents=True, exist_ok=True)


# ---------- io helpers ----------
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def to_datetime_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    s = pd.to_datetime(df[col], errors="coerce")
    return s


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ---------- load base data you already have ----------
def load_trade_log() -> pd.DataFrame:
    # You already have these
    for p in [RESULTS / "trade_log.csv", RESULTS / "executed_trades.csv"]:
        df = read_csv_safe(p)
        if not df.empty:
            # normalize column names
            rename = {}
            if "side" in df.columns and "action" not in df.columns:
                rename["side"] = "action"
            df = df.rename(columns=rename)
            return df
    return pd.DataFrame()


def load_signals() -> pd.DataFrame:
    # Prefer signals_with_rationale.csv, fallback to signals.csv (in results or predictions)
    for p in [
        RESULTS / "signals_with_rationale.csv",
        RESULTS / "signals.csv",
        PRED / "signals.csv",
        PROJECT_ROOT / "predictions" / "signals.csv",
    ]:
        df = read_csv_safe(p)
        if not df.empty:
            break
    else:
        return pd.DataFrame()

    # unify expected cols
    df = df.copy()
    # pick a date-like column
    if "date" not in df.columns:
        for alt in ["timestamp", "asof_date", "as_of", "dt"]:
            if alt in df.columns:
                df["date"] = df[alt]
                break
    df["date"] = to_datetime_col(df, "date")
    df = df[~df["date"].isna()]
    df["date"] = df["date"].dt.normalize()

    # standard optional columns
    must = ["ticker", "date"]
    nice = ["close", "predicted_close", "signal", "confidence", "rationale", "total_score",
            "rsi14", "sma20", "sma50", "atr14", "sentiment", "pe_ratio", "dividend_yield"]
    df = ensure_cols(df, must + nice)
    # numeric coercions
    for c in ["close", "predicted_close", "confidence", "total_score",
              "rsi14", "sma20", "sma50", "atr14", "sentiment", "pe_ratio", "dividend_yield"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # add edge if possible
    if {"close", "predicted_close"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["edge_pct"] = (df["predicted_close"] - df["close"]) / df["close"]
    else:
        df["edge_pct"] = np.nan

    # fill default signal/confidence if missing
    if df["signal"].isna().all():
        df["signal"] = "HOLD"
    if df["confidence"].isna().all():
        df["confidence"] = 0.5
    return df


def load_fundamentals() -> pd.DataFrame:
    for p in [RESULTS / "fundamentals.csv", PROC / "fundamentals.csv"]:
        df = read_csv_safe(p)
        if not df.empty:
            return df
    return pd.DataFrame()


def copy_portfolio_history_if_needed(force: bool = False):
    dst = RESULTS / "portfolio_history.csv"
    if dst.exists() and not force:
        return
    src = SERVICES_RESULTS / "portfolio_history.csv"
    if src.exists():
        try:
            dst.write_bytes(src.read_bytes())
        except Exception:
            pass


# ---------- creators (each returns the path it touched or None) ----------
def make_strategy_vs_market(signals: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = RESULTS / "strategy_vs_market.csv"
    if out.exists() and not force:
        return None
    if signals.empty:
        # minimal placeholder
        df = pd.DataFrame({
            "date": [pd.Timestamp.today().normalize() - pd.Timedelta(days=i) for i in range(10)][::-1],
            "ticker": ["SPY"] * 10,
            "cumulative_strategy": np.linspace(1.00, 1.05, 10),
            "cumulative_market":   np.linspace(1.00, 1.03, 10),
        })
        df.to_csv(out, index=False)
        return out

    rows = []
    for tkr, g in signals.groupby("ticker"):
        g = g.sort_values("date")
        # crude "strategy" = cumprod(1+edge) where edge exists, otherwise 1.0
        edge = g["edge_pct"].fillna(0.0).clip(-0.1, 0.1)
        strat = (1 + edge).cumprod()
        # simple market baseline = a flat 1.0 (you can replace with SPY later)
        market = pd.Series(1.0, index=g.index)
        rows.append(pd.DataFrame({
            "date": g["date"].values,
            "ticker": tkr,
            "cumulative_strategy": strat.values,
            "cumulative_market": market.values
        }))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out, index=False)
    return out


def make_backtest_summary(trades: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = RESULTS / "backtest_summary.csv"
    if out.exists() and not force:
        return None
    if trades.empty:
        # small placeholder
        df = pd.DataFrame([{
            "ticker": "SPY", "trades": 0, "wins": 0, "win_rate": 0.0,
            "total_profit": 0.0, "avg_profit": 0.0, "last_trade": ""
        }])
        df.to_csv(out, index=False)
        return out

    df = trades.copy()
    if "profit" not in df.columns:
        df["profit"] = np.nan
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

    agg = df.groupby("ticker").agg(
        trades=("ticker", "size"),
        wins=("profit", lambda s: int((s.fillna(0) > 0).sum())),
        total_profit=("profit", lambda s: float(s.fillna(0).sum())),
        avg_profit=("profit", lambda s: float(s.fillna(0).mean()))
    ).reset_index()
    agg["win_rate"] = agg.apply(lambda r: (r["wins"] / r["trades"]) if r["trades"] else 0.0, axis=1)
    if "date" in df.columns:
        dts = to_datetime_col(df, "date")
        last = df.assign(_d=dts).sort_values("_d").groupby("ticker").tail(1)[["ticker", "_d"]]
        last = last.rename(columns={"_d": "last_trade"})
        agg = agg.merge(last, on="ticker", how="left")
    agg.to_csv(out, index=False)
    return out


def make_stock_scores(fund: pd.DataFrame, signals: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = RESULTS / "stock_scores.csv"
    if out.exists() and not force:
        return None

    if fund.empty and signals.empty:
        df = pd.DataFrame([{"ticker": "SPY", "total_score": 0.0}])
        df.to_csv(out, index=False)
        return out

    rows = []
    if not fund.empty and "ticker" in fund.columns:
        f = fund.copy()
        # try to use common fundamentals if present
        # lower PE better, higher dividend_yield better
        for c in f.columns:
            if c != "ticker":
                f[c] = pd.to_numeric(f[c], errors="coerce")
        pe = f.filter(regex=r"^pe(_ratio)?$", axis=1)
        dy = f.filter(regex=r"^dividend(_yield)?$", axis=1)
        score = pd.Series(0.0, index=f.index, dtype=float)
        if not pe.empty:
            pev = pd.to_numeric(pe.iloc[:, 0], errors="coerce")
            score = score + (-((pev - pev.mean()) / (pev.std() + 1e-9))).fillna(0)  # lower PE -> higher score
        if not dy.empty:
            dyv = pd.to_numeric(dy.iloc[:, 0], errors="coerce")
            score = score + ((dyv - dyv.mean()) / (dyv.std() + 1e-9)).fillna(0)
        rows.append(pd.DataFrame({"ticker": f["ticker"], "total_score": score}))

    if not signals.empty and "ticker" in signals.columns:
        s = signals.groupby("ticker")["confidence"].mean().rename("sig_conf").reset_index()
        s["sig_conf"] = pd.to_numeric(s["sig_conf"], errors="coerce").fillna(0)
        rows.append(pd.DataFrame({"ticker": s["ticker"], "total_score": s["sig_conf"]}))

    df = pd.concat(rows, ignore_index=True)
    df = df.groupby("ticker", as_index=False)["total_score"].mean()
    df = df.sort_values("total_score", ascending=False)
    df.to_csv(out, index=False)
    return out


def make_news_sentiment(signals: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = RESULTS / "news_sentiment.csv"
    if out.exists() and not force:
        return None
    tickers = (signals["ticker"].dropna().unique().tolist() if not signals.empty else ["SPY"])
    today = pd.Timestamp.today().normalize()
    rows = []
    for t in tickers[:8]:
        rows.append({
            "date": today.strftime("%Y-%m-%d"),
            "ticker": t,
            "sentiment": 0.0,
            "title": f"{t} placeholder headline",
            "url": "https://example.com/",
            "description": f"Auto-generated placeholder news for {t}."
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def make_alerts(signals: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = RESULTS / "alerts.csv"
    if out.exists() and not force:
        return None
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    rows = []
    if not signals.empty:
        s = signals.copy()
        s = s.sort_values("edge_pct", ascending=False)
        for _, r in s.head(10).iterrows():
            rows.append({
                "date": today,
                "ticker": r.get("ticker"),
                "type": "EDGE_SPIKE",
                "priority": "HIGH",
                "score": float((r.get("edge_pct") or 0) * 100),
                "title": f"{r.get('ticker')} elevated model edge",
                "url": "https://example.com/",
                "message": f"Model edge {float((r.get('edge_pct') or 0)*100):.2f}%"
            })
    else:
        rows = [{"date": today, "ticker": "SPY", "type": "INFO", "priority": "LOW",
                 "score": 0, "title": "Placeholder alert", "url": "", "message": "Generated placeholder"}]
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def make_economic_calendar(force: bool = False) -> Optional[Path]:
    out = RESULTS / "economic_calendar.csv"
    if out.exists() and not force:
        return None
    base = pd.Timestamp.today().normalize()
    rows = []
    items = [
        ("CPI YoY", "High"),
        ("PPI MoM", "Medium"),
        ("Initial Jobless Claims", "Medium"),
        ("FOMC Minutes", "High"),
        ("Nonfarm Payrolls", "High"),
    ]
    for i, (event, imp) in enumerate(items):
        rows.append({
            "date": (base + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d"),
            "time": "08:30",
            "event": event,
            "period": "",
            "actual": "",
            "forecast": "",
            "previous": "",
            "importance": imp
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def make_model_comparison(signals: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = RESULTS / "model_comparison.csv"
    if out.exists() and not force:
        return None
    if signals.empty:
        # tiny placeholder
        today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        df = pd.DataFrame([
            {"ticker": "SPY", "date": today, "model": "Naive", "close": 100.0, "predicted_close": 100.0},
            {"ticker": "SPY", "date": today, "model": "ML", "close": 100.0, "predicted_close": 101.0},
        ])
        df.to_csv(out, index=False)
        return out

    s = signals.copy().sort_values(["ticker", "date"])
    s = ensure_cols(s, ["close", "predicted_close"])
    s["close"] = pd.to_numeric(s["close"], errors="coerce")
    s["predicted_close"] = pd.to_numeric(s["predicted_close"], errors="coerce")

    # Two models:
    #  - ML = predicted_close
    #  - Naive = yesterday's close (or today's close if not available)
    rows = []
    for tkr, g in s.groupby("ticker"):
        g = g.dropna(subset=["date"]).sort_values("date")
        close = g["close"].fillna(method="ffill").fillna(method="bfill")
        naive_pred = close.shift(1).fillna(close)
        ml_pred = g["predicted_close"].fillna(close)

        base = g[["date", "ticker"]].copy()
        m1 = base.copy(); m1["model"] = "Naive"; m1["close"] = close.values; m1["predicted_close"] = naive_pred.values
        m2 = base.copy(); m2["model"] = "ML";    m2["close"] = close.values; m2["predicted_close"] = ml_pred.values
        rows.append(pd.concat([m1, m2], ignore_index=True))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out, index=False)
    return out


def make_feature_importance(signals: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = RESULTS / "feature_importance.csv"
    if out.exists() and not force:
        return None
    if signals.empty:
        pd.DataFrame([{"ticker": "SPY", "feature": "confidence", "importance": 1.0}]).to_csv(out, index=False)
        return out

    feats = [c for c in ["rsi14", "sma20", "sma50", "atr14", "sentiment", "confidence"] if c in signals.columns]
    if not feats:
        pd.DataFrame([{"ticker": "SPY", "feature": "confidence", "importance": 1.0}]).to_csv(out, index=False)
        return out

    rows = []
    for tkr, g in signals.groupby("ticker"):
        g = g.dropna(subset=["edge_pct"]).copy()
        if g.empty:
            continue
        imps = []
        y = pd.to_numeric(g["edge_pct"], errors="coerce")
        for f in feats:
            x = pd.to_numeric(g[f], errors="coerce")
            # correlation as a quick proxy
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                corr = 0.0
            else:
                corr = float(pd.Series(x[mask]).corr(pd.Series(y[mask])))
            imps.append({"feature": f, "importance": abs(corr)})
        imp_df = pd.DataFrame(imps)
        if imp_df["importance"].sum() > 0:
            imp_df["importance"] = imp_df["importance"] / imp_df["importance"].sum()
        imp_df["ticker"] = tkr
        rows.append(imp_df)
    if rows:
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.DataFrame([{"ticker": "SPY", "feature": "confidence", "importance": 1.0}])
    df.to_csv(out, index=False)
    return out


def make_buffett_orders(scores: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = ORDERS / "buffett_orders.csv"
    if out.exists() and not force:
        return None
    ORDERS.mkdir(parents=True, exist_ok=True)

    if scores.empty or "ticker" not in scores.columns:
        rows = [{"ticker": "SPY", "action": "HOLD", "target_weight": 0.0, "current_weight": 0.0,
                 "current_value": 0.0, "target_value": 0.0, "delta_notional": 0.0, "buffett_score": 0.0}]
        pd.DataFrame(rows).to_csv(out, index=False)
        return out

    base_value = 100_000.0
    top = scores.sort_values("total_score", ascending=False).head(10).copy()
    # weight proportional to positive score
    s = top["total_score"].clip(lower=0)
    if s.sum() == 0:
        w = np.repeat(1 / len(top), len(top))
    else:
        w = s / s.sum()
    top["target_weight"] = w
    top["current_weight"] = 0.0
    top["current_value"] = 0.0
    top["target_value"] = top["target_weight"] * base_value
    top["delta_notional"] = top["target_value"] - top["current_value"]
    top["action"] = np.where(top["delta_notional"] > 0, "BUY", "SELL")
    top["buffett_score"] = top["total_score"]
    cols = ["ticker", "action", "target_weight", "current_weight", "current_value",
            "target_value", "delta_notional", "buffett_score"]
    top[cols].to_csv(out, index=False)
    return out


def make_orders_today(scores: pd.DataFrame, signals: pd.DataFrame, force: bool = False) -> Optional[Path]:
    out = ORDERS / "orders_today.csv"
    if out.exists() and not force:
        return None
    ORDERS.mkdir(parents=True, exist_ok=True)

    rows = []
    # blend: top 5 by score → BUY; bottom 5 → SELL
    if not scores.empty and "total_score" in scores.columns:
        srt = scores.sort_values("total_score", ascending=False)
        for _, r in srt.head(5).iterrows():
            rows.append({"ticker": r["ticker"], "action": "BUY", "target_weight": 0.03})
        for _, r in srt.tail(5).iterrows():
            rows.append({"ticker": r["ticker"], "action": "SELL", "target_weight": 0.00})

    # add any high-confidence signals (if present)
    if not signals.empty:
        ss = signals.copy()
        if "confidence" in ss.columns and "signal" in ss.columns:
            hi = ss[(pd.to_numeric(ss["confidence"], errors="coerce") >= 0.8)]
            hi = hi.sort_values("confidence", ascending=False).drop_duplicates("ticker")
            for _, r in hi.head(5).iterrows():
                rows.append({"ticker": r["ticker"], "action": str(r.get("signal", "HOLD")).upper(), "target_weight": 0.02})

    if not rows:
        rows = [{"ticker": "SPY", "action": "HOLD", "target_weight": 0.0}]
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def make_ai_feedback(scores: pd.DataFrame, orders_today: Path, force: bool = False) -> Optional[Path]:
    out = RESULTS / "ai_feedback.jsonl"
    if out.exists() and not force:
        return None

    uni = int(scores["ticker"].nunique()) if not scores.empty and "ticker" in scores.columns else 0
    # summarize orders
    odf = read_csv_safe(orders_today) if orders_today.exists() else pd.DataFrame()
    total_buy = float((odf[odf.get("action", "").astype(str).str.upper() == "BUY"]["target_weight"].fillna(0).sum()) * 100000.0) if not odf.empty else 0.0
    total_sell = float((odf[odf.get("action", "").astype(str).str.upper() == "SELL"]["target_weight"].fillna(0).sum()) * 100000.0) if not odf.empty else 0.0

    rec = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "universe": {"count": uni},
        "orders": {"total_buy_notional": total_buy, "total_sell_notional": total_sell},
        "notes": "Auto-generated placeholder so the dashboard has something to show."
    }
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Bootstrap missing dashboard data files.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = ap.parse_args()
    force = args.force

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Writing into: {RESULTS} and {ORDERS}")

    # baseline copies
    copy_portfolio_history_if_needed(force=force)

    trades = load_trade_log()
    signals = load_signals()
    fundamentals = load_fundamentals()

    # Create results/*
    made = []
    for fn in [
        make_strategy_vs_market(signals, force),
        make_backtest_summary(trades, force),
        make_stock_scores(fundamentals, signals, force),
        make_news_sentiment(signals, force),
        make_alerts(signals, force),
        make_economic_calendar(force),
        make_model_comparison(signals, force),
        make_feature_importance(signals, force),
    ]:
        if fn is not None:
            made.append(fn)

    # Create orders/*
    scores = read_csv_safe(RESULTS / "stock_scores.csv")
    bo = make_buffett_orders(scores, force)
    if bo is not None:
        made.append(bo)
    ot = make_orders_today(scores, signals, force)
    if ot is not None:
        made.append(ot)

    # Create ai_feedback.jsonl
    af = make_ai_feedback(scores, ORDERS / "orders_today.csv", force)
    if af is not None:
        made.append(af)

    if made:
        print("\n✅ Created/updated:")
        for p in made:
            print(" -", p)
    else:
        print("\nNothing to do (everything exists). Use --force to regenerate.")


if __name__ == "__main__":
    main()
