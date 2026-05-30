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
#
# ──────────────────────────────────────────────────────────────
# SCORE DISTRIBUTION FIX (see services/signal_distribution.py)
#
# Historical bug (root-cause of repeated 0.65 / 0.60 confidence values across
# downstream artifacts — signal_lifecycle*.csv and trade_opportunities.csv):
#
#   The previous ``confidence_score`` implementation evaluated a minimal
#   rule set (``+0.15`` for BUY, ``+0.10`` for SELL plus conditional
#   ``trend_strength`` / ``volatility`` adjustments). The prediction
#   parquets only carry ``[date, close, predicted_close, signal, ticker]``
#   — ``trend_strength`` / ``volatility`` are never present — so every row
#   collapsed to either 0.50 / 0.60 / 0.65. That flattened value was then
#   written to ``signals_with_rationale.csv`` (which
#   ``apply_signal_lifecycle.py`` prefers over ``signals.csv``), and the
#   collapse propagated through the lifecycle, edge_ranking, and into
#   ``trade_opportunities.csv``.
#
# Fix strategy (this module):
#   * Compute deterministic per-ticker features from the prediction parquet
#     history the pipeline already produces (no new inputs, no randomness).
#   * Blend them into a bounded [0, 1] composite confidence that degrades
#     gracefully when components are missing (skewing conservative, never
#     collapsing to a shared mid-band default).
#   * Emit interpretable component columns so downstream code / dashboards
#     can see *why* a score is high or low, and so edge_ranking can blend
#     them into ``edge_score`` (giving both ``confidence`` and ``edge_score``
#     real per-ticker dispersion).
#
# Execution / broker / lifecycle gate / apply-layer / adaptation behavior
# is NOT changed by this fix.
# ------------------------------------------------------------

from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

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
# Conservative default when no discriminating features are
# available. Deliberately BELOW the historical mid-band (0.50/0.60/0.65)
# so missing-data rows are visibly weak rather than silently accepted.
# ─────────────────────────────────────────────────────────────
MISSING_FEATURE_FLOOR = 0.40


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if not math.isfinite(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _clamp01(v: float) -> float:
    if not math.isfinite(v):
        return float("nan")
    return max(0.0, min(1.0, v))


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


# ─────────────────────────────────────────────────────────────
# Feature extraction (deterministic, from real pipeline data)
# ─────────────────────────────────────────────────────────────
def compute_signal_features(df_hist: pd.DataFrame) -> Dict[str, float]:
    """
    Compute deterministic per-ticker features from the latest row of the
    ticker's prediction history. All features are derived from data the
    pipeline already produces (close, predicted_close). No randomness, no
    jitter. Any missing value returns NaN so callers can degrade gracefully.

    Keys returned:
      delta_pct          : (predicted_close - close) / close for the last row
      return_3d          : close pct-change over last ~3 sessions
      return_7d          : close pct-change over last ~7 sessions
      realized_vol_14d   : std-dev of daily returns over last 14 sessions
      trend_strength     : (ma7 - ma21) / ma21
      delta_rank_pctile  : where the latest |delta| sits in the last 20-row
                           |delta| distribution for this ticker, in [0, 1]
    """
    out: Dict[str, float] = {
        "delta_pct": float("nan"),
        "return_3d": float("nan"),
        "return_7d": float("nan"),
        "realized_vol_14d": float("nan"),
        "trend_strength": float("nan"),
        "delta_rank_pctile": float("nan"),
    }

    if df_hist is None or df_hist.empty:
        return out

    close = pd.to_numeric(df_hist.get("close"), errors="coerce")
    pred = pd.to_numeric(df_hist.get("predicted_close"), errors="coerce")
    if close is None or close.empty or pred is None or pred.empty:
        return out

    last_close = _safe_float(close.iloc[-1])
    last_pred = _safe_float(pred.iloc[-1])

    if math.isfinite(last_close) and last_close > 0 and math.isfinite(last_pred):
        out["delta_pct"] = (last_pred - last_close) / last_close

    if len(close) >= 4:
        c_prev_3 = _safe_float(close.iloc[-4])
        if math.isfinite(c_prev_3) and c_prev_3 > 0 and math.isfinite(last_close):
            out["return_3d"] = last_close / c_prev_3 - 1.0
    if len(close) >= 8:
        c_prev_7 = _safe_float(close.iloc[-8])
        if math.isfinite(c_prev_7) and c_prev_7 > 0 and math.isfinite(last_close):
            out["return_7d"] = last_close / c_prev_7 - 1.0

    if len(close) >= 15:
        rets = close.pct_change().dropna()
        tail = rets.iloc[-14:]
        if len(tail) >= 5:
            v = _safe_float(tail.std())
            if math.isfinite(v):
                out["realized_vol_14d"] = v

    if len(close) >= 22:
        ma7 = _safe_float(close.iloc[-7:].mean())
        ma21 = _safe_float(close.iloc[-21:].mean())
        if math.isfinite(ma7) and math.isfinite(ma21) and ma21 != 0:
            out["trend_strength"] = (ma7 - ma21) / ma21

    try:
        safe_close = close.where(close > 0)
        hist_dp = (pred - close) / safe_close
        mag = hist_dp.abs().iloc[-20:]
        today_mag = abs(out["delta_pct"]) if math.isfinite(out["delta_pct"]) else float("nan")
        if mag.notna().sum() >= 5 and math.isfinite(today_mag):
            denom = max(1, int(mag.notna().sum()))
            out["delta_rank_pctile"] = float((mag < today_mag).sum()) / float(denom)
    except Exception:
        pass

    return out


# ─────────────────────────────────────────────────────────────
# Composite scoring
# ─────────────────────────────────────────────────────────────
def _directional_component(value: float, signal: str, scale: float) -> float:
    """
    Map a signed feature into [0, 1] relative to the signal direction.
    BUY  : positive value → high; negative → low
    SELL : negative value → high (confirming the short); positive → low
    HOLD : small |value| → high (regime is calm); large |value| → low
    """
    if not math.isfinite(value):
        return float("nan")
    scale = max(scale, 1e-9)
    sig = (signal or "").upper().strip()
    if sig == "BUY":
        return 0.5 + 0.5 * math.tanh(value / scale)
    if sig == "SELL":
        return 0.5 + 0.5 * math.tanh(-value / scale)
    # HOLD / unknown — favour quiet, penalise large moves
    return 0.5 - 0.5 * math.tanh(abs(value) / scale)


def build_signal_components(signal: str, features: Dict[str, float]) -> Dict[str, float]:
    """
    Translate raw features into component sub-scores in [0, 1] and emit the
    final composite confidence. Missing components are skipped and the
    remaining weights re-normalise. If *every* discriminating component is
    missing we fall back to MISSING_FEATURE_FLOOR (NOT to the mid-band) so
    we never silently collapse onto a shared default.

    Returns a dict with:
      - confidence               (final, [0, 1])
      - score                    (pre-penalty composite, [0, 1])
      - momentum_score           (return_3d component, [0, 1])
      - trend_score              (trend_strength component, [0, 1])
      - breakout_score           (delta_rank_pctile component, [0, 1])
      - volatility_score         (quality proxy, [0, 1])
      - delta_conviction         (delta_pct component, [0, 1])
      - score_model_component    (delta_pct-driven sub-score)
      - score_rank_component     (delta_rank_pctile sub-score)
      - score_quality_component  (realized-vol quality proxy)
      - score_penalty_component  (penalty magnitude, >= 0)
      - score_final_preclip      (composite before clipping)
    """
    sig = (signal or "").upper().strip()

    delta = _directional_component(features.get("delta_pct", float("nan")), sig, scale=0.010)
    mom_3 = _directional_component(features.get("return_3d", float("nan")), sig, scale=0.020)
    mom_7 = _directional_component(features.get("return_7d", float("nan")), sig, scale=0.040)
    trend = _directional_component(features.get("trend_strength", float("nan")), sig, scale=0.030)

    # Breakout / rank: higher percentile of today's |delta| vs recent |delta|
    # means today's prediction is unusually strong for this ticker. Flat HOLD
    # signals prefer low percentiles.
    rank_raw = features.get("delta_rank_pctile", float("nan"))
    if math.isfinite(rank_raw):
        if sig in ("BUY", "SELL"):
            breakout = _clamp01(rank_raw)
        else:
            breakout = _clamp01(1.0 - rank_raw)
    else:
        breakout = float("nan")

    # Quality proxy: lower realized vol → cleaner signal → higher component.
    # 1% daily vol → ~0.88; 3% → ~0.19. Caps at [0, 1].
    vol = features.get("realized_vol_14d", float("nan"))
    if math.isfinite(vol):
        quality = _clamp01(1.0 - math.tanh(max(0.0, vol - 0.005) / 0.015))
    else:
        quality = float("nan")

    # Momentum component blends 3d and 7d when both present; either one is OK.
    if math.isfinite(mom_3) and math.isfinite(mom_7):
        momentum = 0.6 * mom_3 + 0.4 * mom_7
    elif math.isfinite(mom_3):
        momentum = mom_3
    elif math.isfinite(mom_7):
        momentum = mom_7
    else:
        momentum = float("nan")

    # Re-normalising composite across present components.
    weights = {
        "delta": 0.35,
        "momentum": 0.25,
        "trend": 0.20,
        "breakout": 0.10,
        "quality": 0.10,
    }
    values = {
        "delta": delta,
        "momentum": momentum,
        "trend": trend,
        "breakout": breakout,
        "quality": quality,
    }

    wsum = 0.0
    wtot = 0.0
    for k, w in weights.items():
        v = values[k]
        if math.isfinite(v):
            wsum += w * v
            wtot += w

    if wtot <= 0.0:
        score_final_preclip = MISSING_FEATURE_FLOOR
        composite = MISSING_FEATURE_FLOOR
    else:
        composite = wsum / wtot
        score_final_preclip = composite

    # Penalty: high realized vol drags confidence down even after blending.
    # 2% daily vol → 0.0 pen; 4% → 0.10; 6%+ → capped at 0.15.
    penalty = 0.0
    if math.isfinite(vol):
        penalty = min(0.15, max(0.0, (vol - 0.02) * 5.0))

    # Short-history / unknown-signal rows should skew conservative. We do not
    # apply extra penalty when HOLD comes with strong calm evidence; for
    # BUY/SELL with no conviction signal at all the composite already lands
    # near MISSING_FEATURE_FLOOR.
    final = _clamp01(composite - penalty)

    # When fewer than 2 components contributed we cap the upside so a single
    # noisy component cannot inflate confidence. This keeps low-data rows
    # from masquerading as "strong".
    components_present = sum(1 for v in values.values() if math.isfinite(v))
    if components_present <= 1:
        final = min(final, 0.55)

    return {
        "confidence": _clamp01(final),
        "score": _clamp01(composite),
        "delta_conviction": _clamp01(delta) if math.isfinite(delta) else float("nan"),
        "momentum_score": _clamp01(momentum) if math.isfinite(momentum) else float("nan"),
        "trend_score": _clamp01(trend) if math.isfinite(trend) else float("nan"),
        "breakout_score": _clamp01(breakout) if math.isfinite(breakout) else float("nan"),
        "volatility_score": _clamp01(quality) if math.isfinite(quality) else float("nan"),
        "score_model_component": _clamp01(delta) if math.isfinite(delta) else float("nan"),
        "score_rank_component": _clamp01(breakout) if math.isfinite(breakout) else float("nan"),
        "score_quality_component": _clamp01(quality) if math.isfinite(quality) else float("nan"),
        "score_penalty_component": float(penalty),
        "score_final_preclip": float(score_final_preclip),
        "components_present": int(components_present),
    }


def confidence_score(row: pd.Series) -> float:
    """
    Backwards-compatible single-row confidence calculator.

    Prefers an already-computed ``confidence`` on the row (set by ``run()``
    via ``build_signal_components``). Falls back to a deterministic
    feature-driven estimate; no longer collapses to the old 0.50/0.60/0.65
    triad. Returns a value in [0.10, 0.90] so we keep the historical
    public API of a rounded, bounded scalar.
    """
    existing = _safe_float(row.get("confidence"))
    if math.isfinite(existing):
        return round(max(0.10, min(0.90, existing)), 4)

    feats = {
        "delta_pct": float("nan"),
        "return_3d": float("nan"),
        "return_7d": float("nan"),
        "realized_vol_14d": float("nan"),
        "trend_strength": _safe_float(row.get("trend_strength")),
        "delta_rank_pctile": float("nan"),
    }

    close = _safe_float(row.get("close"))
    pred = _safe_float(row.get("predicted_close"))
    if math.isfinite(close) and close > 0 and math.isfinite(pred):
        feats["delta_pct"] = (pred - close) / close

    returns = _safe_float(row.get("returns"))
    if math.isfinite(returns):
        feats["return_3d"] = returns

    vol = _safe_float(row.get("volatility"))
    if math.isfinite(vol):
        feats["realized_vol_14d"] = vol

    comp = build_signal_components(str(row.get("signal", "HOLD")), feats)
    return round(max(0.10, min(0.90, comp["confidence"])), 4)


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
    t = fp.stem.replace("_predictions", "").strip().upper()
    return t


# Component columns we emit onto signals_with_rationale.csv so they flow
# through the lifecycle writer and downstream edge_ranking.
_COMPONENT_COLUMNS = [
    "score",
    "delta_conviction",
    "momentum_score",
    "trend_score",
    "breakout_score",
    "volatility_score",
    "score_model_component",
    "score_rank_component",
    "score_quality_component",
    "score_penalty_component",
    "score_final_preclip",
    "components_present",
    "delta_pct_snapshot",
    "return_3d_snapshot",
    "return_7d_snapshot",
    "realized_vol_14d_snapshot",
    "trend_strength_snapshot",
    "delta_rank_pctile_snapshot",
]


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

        if "signal" not in df.columns:
            continue

        last = df.iloc[-1].copy()
        last["ticker"] = ticker

        # Per-ticker deterministic features from the full history.
        feats = compute_signal_features(df)

        sig = str(last.get("signal", "HOLD")).upper().strip() or "HOLD"
        comps = build_signal_components(sig, feats)

        last["regime"] = infer_regime(last)
        last["confidence"] = round(float(comps["confidence"]), 6)
        last["rationale"] = build_rationale(last, feat_imp)

        for key in (
            "score",
            "delta_conviction",
            "momentum_score",
            "trend_score",
            "breakout_score",
            "volatility_score",
            "score_model_component",
            "score_rank_component",
            "score_quality_component",
            "score_penalty_component",
            "score_final_preclip",
        ):
            v = comps.get(key)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                last[key] = np.nan
            else:
                last[key] = round(float(v), 6)

        last["components_present"] = int(comps.get("components_present", 0))

        for fkey, outcol in (
            ("delta_pct", "delta_pct_snapshot"),
            ("return_3d", "return_3d_snapshot"),
            ("return_7d", "return_7d_snapshot"),
            ("realized_vol_14d", "realized_vol_14d_snapshot"),
            ("trend_strength", "trend_strength_snapshot"),
            ("delta_rank_pctile", "delta_rank_pctile_snapshot"),
        ):
            fv = feats.get(fkey, float("nan"))
            last[outcol] = round(float(fv), 6) if math.isfinite(fv) else np.nan

        last["portfolio_drawdown"] = risk.get("drawdown")
        last["capital_mode"] = risk.get("capital_mode", "NORMAL")

        records.append(last)

    if not records:
        print("No signals found. Nothing written.")
        return

    out_df = pd.DataFrame(records)

    out_df["generated_at_utc"] = _utc_now_z()
    try:
        out_df["generated_at_local"] = datetime.now().isoformat(timespec="seconds")
    except Exception:
        out_df["generated_at_local"] = ""

    if "date" in out_df.columns:
        out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out_df.to_csv(SNAPSHOT_CSV, index=False)
    out_df.to_json(SNAPSHOT_JSON, orient="records", indent=2)
    out_df.to_csv(SIGNALS_WITH_RATIONALE_CSV, index=False)

    try:
        conf_series = pd.to_numeric(out_df["confidence"], errors="coerce").dropna()
        if not conf_series.empty:
            print(
                "[trade_rationale] confidence distribution: "
                f"n={len(conf_series)} unique={conf_series.nunique()} "
                f"min={conf_series.min():.3f} max={conf_series.max():.3f} "
                f"mean={conf_series.mean():.3f} median={conf_series.median():.3f}"
            )
    except Exception:
        pass

    print(f"✔ Snapshot written: {SNAPSHOT_CSV}")
    print(f"✔ Snapshot written: {SNAPSHOT_JSON}")
    print(f"✔ Written: {SIGNALS_WITH_RATIONALE_CSV}")
    print(f"🕒 generated_at_utc: {out_df['generated_at_utc'].iloc[0]}")


if __name__ == "__main__":
    run()
