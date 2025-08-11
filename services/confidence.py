# services/confidence.py
import numpy as np
import math

def _nz(x, default=0.0):
    try:
        return float(x) if x == x and np.isfinite(x) else float(default)
    except Exception:
        return float(default)

def compute_confidence(row):
    """
    Returns confidence in [0,1] based on:
      • Model edge (predicted_close vs close)
      • Trend alignment (SMA20 vs SMA50, optional RSI band)
      • Volatility regime (ATR/Price)
      • Fundamental/score + news sentiment (if present)
    """
    close = _nz(row.get("close"))
    pred  = _nz(row.get("predicted_close"), close)
    edge  = (pred - close) / close if close > 0 else 0.0                  # +/- %
    atrp  = _nz(row.get("atr14")) / close if close > 0 else 0.0           # ATR as % price
    rsi   = _nz(row.get("rsi14"))
    sma20 = _nz(row.get("sma20"))
    sma50 = _nz(row.get("sma50"))
    sent  = _nz(row.get("sentiment"))   # -1..+1 expected
    score = _nz(row.get("total_score")) # 0..100 expected (or 0..1)

    # 1) Model edge quality (bigger absolute edge -> higher confidence)
    edge_abs = abs(edge)
    edge_norm = np.tanh(edge_abs / 0.02)  # ~2% edge maps near 0.76, 5% -> ~0.96
    edge_sign_ok = 1.0 if (edge >= 0 and row.get("signal") == "BUY") or (edge < 0 and row.get("signal") == "SELL") else 0.3
    w_edge = edge_norm * edge_sign_ok

    # 2) Trend alignment: SMA20 vs SMA50
    trend = 1.0 if (row.get("signal") == "BUY" and sma20 > sma50) or (row.get("signal") == "SELL" and sma20 < sma50) else 0.35

    # Optional RSI confirmation (BUY prefers oversold <40, SELL prefers overbought >60)
    rsi_conf = 0.7 if (row.get("signal") == "BUY" and rsi < 40) or (row.get("signal") == "SELL" and rsi > 60) else 0.4

    # 3) Volatility regime (too high vol reduces confidence; too low also meh)
    # atr% sweet spot around 1–3%
    atrp_clamped = min(max(atrp, 0.0), 0.10)
    if atrp_clamped <= 0.01:
        vol_factor = 0.6
    elif atrp_clamped <= 0.03:
        vol_factor = 1.0
    elif atrp_clamped <= 0.06:
        vol_factor = 0.8
    else:
        vol_factor = 0.55

    # 4) Sentiment + fundamentals (optional, neutral if missing)
    sent_norm  = (sent + 1.0) / 2.0   # map -1..+1 → 0..1
    score_norm = score / 100.0 if score > 1.0 else score
    if row.get("signal") == "BUY":
        qual = 0.5 * sent_norm + 0.5 * score_norm if (sent or score) else 0.5
    else:  # SELL
        qual = 0.5 * (1 - sent_norm) + 0.5 * (1 - score_norm) if (sent or score) else 0.5

    # Weighted blend → [0,1]
    # Heavier weight on model edge + trend; others modulate
    conf_raw = (0.45 * w_edge + 0.30 * trend + 0.10 * rsi_conf + 0.10 * qual) * vol_factor

    # Smooth & clamp
    confidence = float(np.clip(conf_raw, 0.0, 1.0))

    # Suggested position sizing (Kelly-lite on edge/vol; capped 0–5%)
    k_edge = np.clip(edge, -0.15, 0.15)     # cap crazy edges
    risk = max(atrp_clamped, 0.005)         # avoid divide by tiny
    kelly_like = np.clip((k_edge / (2 * risk)), -0.10, 0.10)   # -10%..+10%
    pos_size = float(np.clip(confidence * max(kelly_like, 0.0) * 0.5, 0.0, 0.05))  # cap 5%

    return confidence, pos_size, edge
