#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRITON — Baseline Analyzer / Stress Diagnostics (Phase 3 → Step 1)
------------------------------------------------------------------
Scans the latest portfolio, signals, trades, and (optionally) weights to
produce a compact baseline summary JSON and a detailed CSV diagnostic table.

Inputs (default: data/results/):
  - portfolio_history.csv   (columns: date[, equity|portfolio_value|cash+market_value], ... )
  - trade_log.csv           (optional; columns: date,ticker,qty,price[,pnl],side,signal,action/status,...)
  - signals_with_rationale.csv (optional; columns: date,ticker,signal,confidence/score,...)
  - weights.csv             (optional; columns: ticker,target_weight[,...])

Outputs (default: data/results/baseline/):
  - baseline_summary.json
  - baseline_report.csv

Exit code:
  0 = PASS or WARN
  2 = FAIL (one or more critical diagnostics failed)

Author: TRITON
Updated: 2025-10-19
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Configurable thresholds
# ---------------------------

@dataclass
class Thresholds:
    max_drawdown_warn: float = -0.15     # warn if DD <= -15%
    max_drawdown_fail: float = -0.25     # fail if DD <= -25%

    vol_warn: float = 0.02               # daily vol > 2% -> warn
    vol_fail: float = 0.035              # daily vol > 3.5% -> fail

    sharpe_warn: float = 0.6             # recent Sharpe < 0.6 -> warn
    sharpe_fail: float = 0.3             # recent Sharpe < 0.3 -> fail

    buy_sell_imbalance_warn: float = 0.75  # if one side > 75% of signals -> warn
    buy_sell_imbalance_fail: float = 0.9   # if one side > 90% -> fail

    concentration_warn: float = 0.25     # top position weight > 25% -> warn
    concentration_fail: float = 0.35     # top position weight > 35% -> fail

    winrate_warn: float = 0.45           # win rate < 45% -> warn (context-dependent)
    winrate_fail: float = 0.35           # win rate < 35% -> fail

    # minimum rows to compute stable stats
    min_days_for_stats: int = 30
    min_trades_for_stats: int = 20
    min_signals_for_stats: int = 30


# ---------------------------
# Helpers
# ---------------------------

def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def max_drawdown_from_equity(eq: pd.Series) -> float:
    """Return max drawdown as a negative fraction (e.g., -0.22 for -22%)."""
    if eq is None or len(eq) < 2:
        return float("nan")
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    return float(dd.min())


def recent_window(df: pd.DataFrame, date_col: str, days: int) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return df
    end = df[date_col].max()
    start = end - pd.Timedelta(days=days)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


def coalesce_date_column(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Rename any of ['date','timestamp','dt','time'] to 'date' and convert to datetime."""
    if df is None or df.empty:
        return df
    for cand in ["date", "timestamp", "dt", "time"]:
        if cand in df.columns:
            if cand != "date":
                df = df.rename(columns={cand: "date"})
            break
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df


def get_equity_series(ph: pd.DataFrame) -> pd.Series:
    """
    Return an equity-like series from portfolio_history with tolerant column mapping.
    Preference order:
      equity | portfolio_value | portfolioValue | value | nav | equity_value |
      total_equity | total_value | market_value/marketValue (optionally + cash)
      or derived: cash + market_value/marketValue
    """
    if ph is None or ph.empty or "date" not in ph.columns:
        return pd.Series(dtype=float)

    ph = ph.copy()
    # normalize numeric columns where present
    num_cols = [
        "equity", "portfolio_value", "portfolioValue", "value", "nav",
        "equity_value", "market_value", "marketValue", "cash",
        "total_equity", "total_value"
    ]
    for c in num_cols:
        if c in ph.columns:
            ph[c] = pd.to_numeric(ph[c], errors="coerce")

    candidates = [
        "equity",
        "portfolio_value", "portfolioValue",
        "value", "nav", "equity_value",
        "total_equity", "total_value",
        "market_value", "marketValue",
    ]
    col = next((c for c in candidates if c in ph.columns), None)

    if col is None:
        # Try to derive from cash + market_value
        if "cash" in ph.columns and ("market_value" in ph.columns or "marketValue" in ph.columns):
            mv_col = "market_value" if "market_value" in ph.columns else "marketValue"
            s = (ph["cash"].fillna(0.0) + ph[mv_col].fillna(0.0))
        else:
            return pd.Series(dtype=float)
    else:
        s = ph[col]
        # If chosen column is market_value, prefer adding cash if available
        if col in ("market_value", "marketValue") and "cash" in ph.columns:
            s = s.fillna(0.0) + ph["cash"].fillna(0.0)

    out = pd.Series(s.values, index=pd.to_datetime(ph["date"]), dtype="float64").sort_index()
    return out


def side_fraction_counts(signal_series: pd.Series) -> Dict[str, float]:
    if signal_series is None or signal_series.empty:
        return {}
    counts = signal_series.fillna("UNKNOWN").astype(str).str.upper().value_counts()
    total = counts.sum()
    fracs = {k: (v / total) for k, v in counts.items()}
    return fracs


def classify_status(value: float, warn: float, fail: float, higher_is_better: bool) -> str:
    """
    Return PASS/WARN/FAIL depending on thresholds.
    If higher_is_better is True: value < fail -> FAIL, < warn -> WARN, else PASS
    If False (lower better): value > fail -> FAIL, > warn -> WARN, else PASS
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "UNKNOWN"

    if higher_is_better:
        if value < fail:
            return "FAIL"
        if value < warn:
            return "WARN"
        return "PASS"
    else:
        if value > fail:
            return "FAIL"
        if value > warn:
            return "WARN"
        return "PASS"


# ---------------------------
# Analyzer
# ---------------------------

@dataclass
class DiagnosticRow:
    metric: str
    value: float | str
    window_days: int
    status: str
    notes: str = ""


class BaselineAnalyzer:
    def __init__(self,
                 results_dir: str = "data/results",
                 outdir: str = "data/results/baseline",
                 days: int = 90,
                 shocks: Tuple[float, ...] = (-0.05, -0.10, -0.20),
                 thresholds: Thresholds = Thresholds(),
                 quiet: bool = False):
        self.results_dir = results_dir
        self.outdir = outdir
        self.days = days
        self.shocks = shocks
        self.th = thresholds
        self.quiet = quiet

        ensure_outdir(self.outdir)

        # Inputs
        self.path_ph = os.path.join(results_dir, "portfolio_history.csv")
        self.path_trades = os.path.join(results_dir, "trade_log.csv")
        self.path_signals = os.path.join(results_dir, "signals_with_rationale.csv")
        self.path_weights = os.path.join(results_dir, "weights.csv")

        # Outputs
        self.path_summary = os.path.join(outdir, "baseline_summary.json")
        self.path_report = os.path.join(outdir, "baseline_report.csv")

        # Data holders
        self.df_ph = None
        self.df_tr = None
        self.df_sig = None
        self.df_w = None

        # Diagnostics
        self.rows: List[DiagnosticRow] = []

    # ---------- Loading ----------
    def load(self) -> None:
        self.df_ph = read_csv_safe(self.path_ph)
        self.df_tr = read_csv_safe(self.path_trades)
        self.df_sig = read_csv_safe(self.path_signals)
        self.df_w = read_csv_safe(self.path_weights)

        if self.df_ph is None:
            raise FileNotFoundError(f"Missing portfolio history: {self.path_ph}")

        # Normalize date columns
        self.df_ph = coalesce_date_column(self.df_ph)
        self.df_tr = coalesce_date_column(self.df_tr)
        self.df_sig = coalesce_date_column(self.df_sig)

        # Window by days for consistency
        self.df_ph = recent_window(self.df_ph, "date", self.days)
        if self.df_tr is not None and "date" in (self.df_tr.columns if self.df_tr is not None else []):
            self.df_tr = recent_window(self.df_tr, "date", self.days)
        if self.df_sig is not None and "date" in (self.df_sig.columns if self.df_sig is not None else []):
            self.df_sig = recent_window(self.df_sig, "date", self.days)

    # ---------- Equity & Risk ----------
    def compute_equity_and_risk(self) -> Dict[str, float]:
        # Build equity series from the (already windowed) df
        eq_series = get_equity_series(self.df_ph)
        # If window too sparse, try full history as a fallback to compute metrics
        if len(eq_series) < 2:
            full_ph = coalesce_date_column(read_csv_safe(self.path_ph))
            if full_ph is not None:
                eq_full = get_equity_series(full_ph.sort_values("date"))
                if len(eq_full) >= 2:
                    eq_series = eq_full

        last_equity = float(eq_series.iloc[-1]) if len(eq_series) else float("nan")
        rets = eq_series.pct_change().dropna()
        n = int(len(rets))

        vol = float("nan")
        mdd = float("nan")
        sharpe = float("nan")

        if len(eq_series) >= 2 and n >= 1:
            # compute even for small samples; we’ll mark sample-size as a separate diagnostic
            vol = float(rets.std()) if n >= 2 else 0.0  # std undefined for 1 return; treat as 0.0
            mdd = max_drawdown_from_equity(eq_series)
            mu = float(rets.mean())
            sharpe = (mu / vol) * math.sqrt(252) if vol and vol > 0 else float("nan")

        # Sample-size diagnostic
        if len(eq_series) >= 2:
            size_status = "PASS" if n >= self.th.min_days_for_stats else "WARN"
            self.rows.append(DiagnosticRow(
                metric="equity_sample_size",
                value=n,
                window_days=self.days,
                status=size_status,
                notes=f"Number of daily returns used. Target >= {self.th.min_days_for_stats} for stable stats."
            ))
        else:
            self.rows.append(DiagnosticRow(
                metric="equity_sample_size",
                value=0,
                window_days=self.days,
                status="UNKNOWN",
                notes="Too few equity points to compute returns (even after full-history fallback)."
            ))

        # Status rows
        self.rows.append(DiagnosticRow(
            metric="daily_volatility",
            value=vol,
            window_days=self.days,
            status=classify_status(vol, self.th.vol_warn, self.th.vol_fail, higher_is_better=False),
            notes="Daily return std. Lower is better."
        ))
        self.rows.append(DiagnosticRow(
            metric="max_drawdown",
            value=mdd,
            window_days=self.days,
            status=classify_status(mdd, self.th.max_drawdown_warn, self.th.max_drawdown_fail, higher_is_better=True),
            notes="Max drawdown (negative). Less negative is better."
        ))
        self.rows.append(DiagnosticRow(
            metric="recent_sharpe",
            value=sharpe,
            window_days=self.days,
            status=classify_status(sharpe, self.th.sharpe_warn, self.th.sharpe_fail, higher_is_better=True),
            notes="Approx. annualized; simple mean/vol * sqrt(252)."
        ))

        # Stress shocks
        if not (last_equity is None or np.isnan(last_equity)):
            for shock in self.shocks:
                est = last_equity * (1.0 + shock)
                self.rows.append(DiagnosticRow(
                    metric=f"shock_equity_{int(shock*100)}pct",
                    value=est,
                    window_days=self.days,
                    status="INFO",
                    notes=f"Estimated equity after {int(shock*100)}% market shock."
                ))

        return {
            "daily_volatility": vol,
            "max_drawdown": mdd,
            "recent_sharpe": sharpe,
            "last_equity": None if np.isnan(last_equity) else last_equity
        }

    # ---------- Signals ----------
    def compute_signals_health(self) -> Dict[str, float | Dict]:
        if self.df_sig is None or self.df_sig.empty:
            self.rows.append(DiagnosticRow(
                metric="signals_presence",
                value="absent",
                window_days=self.days,
                status="WARN",
                notes="signals_with_rationale.csv not found or empty in window."
            ))
            return {}

        sig_col = "signal" if "signal" in self.df_sig.columns else None
        conf_col = "confidence" if "confidence" in self.df_sig.columns else ("score" if "score" in self.df_sig.columns else None)

        counts = {}
        if sig_col:
            fracs = side_fraction_counts(self.df_sig[sig_col])
            counts = {k: int(v * len(self.df_sig)) for k, v in fracs.items()}
            buys = fracs.get("BUY", 0.0)
            sells = fracs.get("SELL", 0.0)
            major_side = max(buys, sells)
            status = "PASS"
            if major_side >= self.th.buy_sell_imbalance_fail:
                status = "FAIL"
            elif major_side >= self.th.buy_sell_imbalance_warn:
                status = "WARN"
            self.rows.append(DiagnosticRow(
                metric="signal_buy_sell_imbalance",
                value=major_side,
                window_days=self.days,
                status=status,
                notes=f"Max(BUY%, SELL%)={major_side:.2%}. "
                      f"Thresholds warn>{self.th.buy_sell_imbalance_warn:.0%}, fail>{self.th.buy_sell_imbalance_fail:.0%}."
            ))

        if conf_col:
            conf_vals = pd.to_numeric(self.df_sig[conf_col], errors="coerce").dropna()
            mean_conf = float(conf_vals.mean()) if len(conf_vals) else float("nan")
            med_conf = float(conf_vals.median()) if len(conf_vals) else float("nan")
        else:
            mean_conf = float("nan")
            med_conf = float("nan")

        # per-ticker concentration
        skew_stat = float("nan")
        if "ticker" in self.df_sig.columns and len(self.df_sig) >= self.th.min_signals_for_stats:
            per_ticker = self.df_sig["ticker"].value_counts(normalize=True)
            skew_stat = float(per_ticker.max()) if len(per_ticker) else float("nan")
            status = "PASS" if (np.isnan(skew_stat) or skew_stat <= 0.40) else "WARN"
            self.rows.append(DiagnosticRow(
                metric="signal_ticker_concentration",
                value=skew_stat,
                window_days=self.days,
                status=status,
                notes="Max fraction of signals in any single ticker."
            ))

        # Presence
        self.rows.append(DiagnosticRow(
            metric="signals_presence",
            value="present",
            window_days=self.days,
            status="PASS",
            notes=f"Signals rows={len(self.df_sig)} in last {self.days}d."
        ))

        return {
            "counts": counts,
            "mean_confidence": mean_conf,
            "median_confidence": med_conf,
            "ticker_concentration": skew_stat
        }

    # ---------- Trades / PnL ----------
    def compute_trade_stats(self) -> Dict[str, float]:
        if self.df_tr is None or self.df_tr.empty:
            self.rows.append(DiagnosticRow(
                metric="trades_presence",
                value="absent",
                window_days=self.days,
                status="WARN",
                notes="trade_log.csv not found or empty in window."
            ))
            return {}

        df = self.df_tr.copy()

        # Normalize action/status text if present
        for col in ("action", "status"):
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).str.lower()

        # Primary: keep action == fill if present
        if "action" in df.columns and (df["action"] == "fill").any():
            df = df[df["action"] == "fill"]
        # Fallback: status values
        elif "status" in df.columns and (df["status"].isin(["filled", "done"])).any():
            df = df[df["status"].isin(["filled", "done"])]
        # Last resort: any numeric PnL
        elif "pnl" in df.columns:
            df = df[pd.to_numeric(df["pnl"], errors="coerce").notna()]

        pnl_col = "pnl" if "pnl" in df.columns else None

        trade_count = len(df)
        winrate = float("nan")
        avg_win = float("nan")
        avg_loss = float("nan")

        if pnl_col:
            pnl_vals = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
            wins = pnl_vals[pnl_vals > 0]
            losses = pnl_vals[pnl_vals < 0]
            if len(pnl_vals) >= max(10, self.th.min_trades_for_stats // 2):
                winrate = float((len(wins) / len(pnl_vals))) if len(pnl_vals) else float("nan")
                avg_win = float(wins.mean()) if len(wins) else float("nan")
                avg_loss = float(losses.mean()) if len(losses) else float("nan")

        if not np.isnan(winrate):
            self.rows.append(DiagnosticRow(
                metric="win_rate",
                value=winrate,
                window_days=self.days,
                status=classify_status(winrate, self.th.winrate_warn, self.th.winrate_fail, higher_is_better=True),
                notes="Share of profitable trades in window."
            ))

        self.rows.append(DiagnosticRow(
            metric="trades_presence",
            value=f"present ({trade_count})",
            window_days=self.days,
            status="PASS" if trade_count > 0 else "WARN",
            notes=f"{trade_count} trades in last {self.days}d."
        ))

        return {
            "trade_count": trade_count,
            "win_rate": winrate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    # ---------- Weights / Exposure ----------
    def compute_exposure(self) -> Dict[str, float | Dict]:
        if self.df_w is None or self.df_w.empty:
            self.rows.append(DiagnosticRow(
                metric="weights_presence",
                value="absent",
                window_days=self.days,
                status="WARN",
                notes="weights.csv not found (exposure checks skipped)."
            ))
            return {}

        # Flexible weight column
        weight_col = None
        for c in ["target_weight", "weight", "w", "target"]:
            if c in self.df_w.columns:
                weight_col = c
                break
        if weight_col is None:
            self.rows.append(DiagnosticRow(
                metric="weights_presence",
                value="invalid",
                window_days=self.days,
                status="WARN",
                notes="weights.csv present but no weight column."
            ))
            return {}

        w = pd.to_numeric(self.df_w[weight_col], errors="coerce").fillna(0.0)
        gross = float(np.abs(w).sum())
        net = float(w.sum())
        top = float(w.abs().max()) if len(w) else float("nan")

        status = classify_status(top, self.th.concentration_warn, self.th.concentration_fail, higher_is_better=False)
        self.rows.append(DiagnosticRow(
            metric="top_weight_concentration",
            value=top,
            window_days=self.days,
            status=status,
            notes=f"Max(|weight|). Gross={gross:.2f}, Net={net:.2f}."
        ))

        return {
            "gross_exposure": gross,
            "net_exposure": net,
            "top_weight": top
        }

    # ---------- Correlation Snapshot (optional) ----------
    def compute_correlation_snapshot(self) -> Optional[float]:
        """
        Rough correlation snapshot using trade-level daily PnL by ticker.
        Returns mean off-diagonal correlation as a scalar if computable.
        """
        if self.df_tr is None or self.df_tr.empty or "date" not in self.df_tr.columns or "ticker" not in self.df_tr.columns:
            return None
        if "pnl" not in self.df_tr.columns:
            return None

        df = self.df_tr.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        daily = (
            df.groupby(["ticker", df["date"].dt.date])["pnl"]
              .sum()
              .unstack("ticker")
              .fillna(0.0)
        )
        if daily.shape[1] < 2 or daily.shape[0] < 5:
            return None

        standardized = (daily - daily.mean()) / (daily.std().replace(0, np.nan))
        corr = standardized.corr().values
        n = corr.shape[0]
        if n < 2:
            return None
        off_diag = corr[np.triu_indices(n, k=1)]
        mean_corr = float(np.nanmean(off_diag))

        status = "PASS"
        if not np.isnan(mean_corr):
            if mean_corr > 0.8:
                status = "FAIL"
            elif mean_corr > 0.6:
                status = "WARN"

        self.rows.append(DiagnosticRow(
            metric="mean_offdiag_correlation",
            value=mean_corr,
            window_days=self.days,
            status=status,
            notes="Higher correlation = lower diversification (proxy from trade PnL)."
        ))
        return mean_corr

    # ---------- Aggregate & Save ----------
    def aggregate_overall_status(self) -> str:
        statuses = [r.status for r in self.rows if r.status in ("PASS", "WARN", "FAIL")]
        if "FAIL" in statuses:
            return "FAIL"
        if "WARN" in statuses:
            return "WARN"
        return "PASS" if statuses else "UNKNOWN"

    def save_outputs(self,
                     equity_risk: Dict[str, float],
                     signal_health: Dict,
                     trade_stats: Dict[str, float],
                     exposure: Dict[str, float],
                     mean_corr: Optional[float]) -> str:
        report_df = pd.DataFrame([asdict(r) for r in self.rows])
        report_df.to_csv(self.path_report, index=False)

        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "window_days": self.days,
            "results_dir": self.results_dir,
            "equity_risk": equity_risk,
            "signals": signal_health,
            "trades": trade_stats,
            "exposure": exposure,
            "mean_offdiag_correlation": mean_corr,
            "shocks": list(self.shocks),
            "diagnostics": [asdict(r) for r in self.rows],
            "overall_status": self.aggregate_overall_status(),
        }
        with open(self.path_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary["overall_status"]

    # ---------- Run ----------
    def run(self) -> str:
        self.load()
        equity_risk = self.compute_equity_and_risk()
        signal_health = self.compute_signals_health()
        trade_stats = self.compute_trade_stats()
        exposure = self.compute_exposure()
        mean_corr = self.compute_correlation_snapshot()
        status = self.save_outputs(equity_risk, signal_health, trade_stats, exposure, mean_corr)

        if not self.quiet:
            print("\n=== TRITON Baseline Analyzer ===")
            print(f"Window: {self.days}d | Results: {self.results_dir} | Out: {self.outdir}")
            print(f"Overall status: {status}")
            print("\nKey metrics:")
            for r in self.rows:
                if r.status in ("FAIL", "WARN", "PASS"):
                    if isinstance(r.value, (int, float)) and not np.isnan(r.value):
                        val = f"{r.value:.4f}"
                    else:
                        val = str(r.value)
                    print(f" - {r.metric:28s} {val:>12s}  [{r.status}]  {r.notes}")

            print(f"\nWrote: {self.path_summary}")
            print(f"Wrote: {self.path_report}\n")

        return status


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRITON Baseline Analyzer / Stress Diagnostics")
    p.add_argument("--results-dir", default="data/results", help="Folder with Triton result CSVs")
    p.add_argument("--outdir", default="data/results/baseline", help="Output folder for summary/report")
    p.add_argument("--days", type=int, default=90, help="Lookback window in days")
    p.add_argument(
        "--shocks",
        type=float,
        nargs="+",
        default=[-0.05, -0.10, -0.20],
        help="Shock percentages (negative numbers, e.g. -0.05 -0.10)"
    )
    p.add_argument("--min-days-stats", type=int, default=30,
                   help="Target number of daily returns for stable stats (warn if fewer).")
    p.add_argument("--quiet", action="store_true", help="Less console output")
    return p.parse_args()


def main():
    args = parse_args()
    analyzer = BaselineAnalyzer(
        results_dir=args.results_dir,
        outdir=args.outdir,
        days=args.days,
        shocks=tuple(args.shocks),
        quiet=args.quiet
    )
    # Let CLI override the stability target without code edits
    analyzer.th.min_days_for_stats = args.min_days_stats

    status = analyzer.run()
    if status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
