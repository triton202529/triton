# services/trade_rationale.py
# ------------------------------------------------------------
# TRITON — Trade Rationale Generator
# Phase 2 / Step 1
#
# Inputs:
#   - data/predictions/*_predictions.parquet
#   - data/results/feature_importance.csv (optional)
#   - data/results/risk_snapshot.json (optional)
#
# Outputs (SNAPSHOT, does NOT overwrite timeseries):
#   - data/results/signals_snapshot.csv
#   - data/results/signals_snapshot.json
#   - data/results/signals_with_rationale.csv  (same rows; consumed by apply_signal_lifecycle / UI)
#
# Zero impact on execution or backtest pipelines.
# Capital Preservation Doctrine respected.
# ------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
PRED_DIR = DATA_ROOT / "predictions"
RESULTS_DIR = DATA_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# New snapshot outputs (do NOT collide with generate_signals output)
SNAPSHOT_CSV = RESULTS_DIR / "signals_snapshot.csv"
SNAPSHOT_JSON = RESULTS_DIR / "signals_snapshot.json"
SIGNALS_WITH_RATIONALE_CSV = RESULTS_DIR / "signals_with_rationale.csv"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _utc_now_z() -> str:
    # Stable Zulu format (matches your other artifacts)
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_feature_importance() -> pd.DataFrame:
    fp = RESULTS_DIR / "feature_importance.csv"
    if not fp.exists() or fp.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def load_risk_snapshot() -> dict:
    fp = RESULTS_DIR / "risk_snapshot.json"
    if not fp.exists() or fp.stat().st_size == 0:
        return {}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def infer_regime(row: pd.Series) -> str:
    """
    Lightweight regime inference.
    No model changes. Pure context layer.
    """
    vol = row.get("volatility", np.nan)
    trend = row.get("trend_strength", np.nan)

    try:
        if pd.notna(vol) and float(vol) > 0.03:
            return "High Volatility"
    except Exception:
        pass

    try:
        if pd.notna(trend) and float(trend) > 0:
            return "Trend Expansion"
    except Exception:
        pass

    return "Neutral / Range"


def confidence_score(row: pd.Series) -> float:
    """
    Conservative confidence estimate.
    Bounded, explainable, non-overfitting.
    """
    base = 0.5
    sig = str(row.get("signal", "HOLD")).upper().strip()

    if sig == "BUY":
        base += 0.15
    elif sig == "SELL":
        base += 0.10

    try:
        if float(row.get("trend_strength", 0) or 0) > 0:
            base += 0.10
    except Exception:
        pass

    try:
        if float(row.get("volatility", 0) or 0) > 0.035:
            base -= 0.10
    except Exception:
        pass

    return round(float(np.clip(base, 0.10, 0.90)), 2)


def build_rationale(row: pd.Series, feat_imp: pd.DataFrame) -> str:
    bullets = []
    sig = str(row.get("signal", "HOLD")).upper().strip() or "HOLD"
    bullets.append(f"Signal: {sig}")

    if "returns" in row.index:
        try:
            bullets.append(f"Recent return: {float(row['returns']):.2%}")
        except Exception:
            pass

    if "ma_diff" in row.index:
        try:
            bullets.append(f"MA spread: {float(row['ma_diff']):.4f}")
        except Exception:
            pass

    top_feats: list[str] = []
    ticker = row.get("ticker")
    if ticker and not feat_imp.empty and "ticker" in feat_imp.columns:
        sub = feat_imp[feat_imp["ticker"] == ticker]
        if not sub.empty and {"feature", "importance"}.issubset(sub.columns):
            top_feats = (
                sub.sort_values("importance", ascending=False)
                .head(3)["feature"]
                .astype(str)
                .tolist()
            )

    if top_feats:
        bullets.append(f"Key drivers: {', '.join(top_feats)}")

    return " • ".join(bullets)


def _safe_ticker_from_fp(fp: Path) -> str:
    # fp like: AAPL_predictions.parquet
    t = fp.stem.replace("_predictions", "").strip().upper()
    return t


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run() -> None:
    records: list[pd.Series] = []
    feat_imp = load_feature_importance()
    risk = load_risk_snapshot()

    if not PRED_DIR.exists():
        print(f"No predictions directory: {PRED_DIR}")
        return

    for fp in sorted(PRED_DIR.glob("*_predictions.parquet")):
        ticker = _safe_ticker_from_fp(fp)

        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"[SKIP] {ticker}: failed to read parquet ({e})")
            continue

        if df is None or df.empty:
            continue

        # We only snapshot latest row that already has a computed signal
        if "signal" not in df.columns:
            continue

        last = df.iloc[-1].copy()
        last["ticker"] = ticker

        # Add context fields
        last["regime"] = infer_regime(last)
        last["confidence"] = confidence_score(last)
        last["rationale"] = build_rationale(last, feat_imp)

        # Attach global risk context (non-blocking)
        last["portfolio_drawdown"] = risk.get("drawdown")
        last["capital_mode"] = risk.get("capital_mode", "NORMAL")

        records.append(last)

    if not records:
        print("No signals found. Nothing written.")
        return

    out_df = pd.DataFrame(records)

    # generated_at stamps (keep UTC as primary truth)
    out_df["generated_at_utc"] = _utc_now_z()
    try:
        out_df["generated_at_local"] = datetime.now().isoformat(timespec="seconds")
    except Exception:
        out_df["generated_at_local"] = ""

    # Optional: keep output tidy and deterministic
    if "date" in out_df.columns:
        out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Write snapshot outputs
    out_df.to_csv(SNAPSHOT_CSV, index=False)
    out_df.to_json(SNAPSHOT_JSON, orient="records", indent=2)
    out_df.to_csv(SIGNALS_WITH_RATIONALE_CSV, index=False)

    print(f"✔ Snapshot written: {SNAPSHOT_CSV}")
    print(f"✔ Snapshot written: {SNAPSHOT_JSON}")
    print(f"✔ Written: {SIGNALS_WITH_RATIONALE_CSV}")
    print(f"🕒 generated_at_utc: {out_df['generated_at_utc'].iloc[0]}")


if __name__ == "__main__":
    run()
