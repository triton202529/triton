# services/generate_signals.py
"""
Generate signals + rationale from prediction files.

- Reads:  data/predictions/*_predictions.(parquet|csv)
- Writes: data/results/signals_with_rationale.csv
          data/results/signals.csv (back-compat)

Phase 1.5:
- Heartbeat ok/fail
- Windows import bootstrap

AS_OF Discipline (clean + robust):
- session_as_of_date = NYSE last completed session date (ET, calendar-aware via pandas_market_calendars)
- effective_as_of_date:
    * Default: session_as_of_date
    * If predictions coverage for session_as_of_date is low (provider lag), downgrade to a data-supported date
      so pipeline can complete while logging the issue loudly.
- Hard fail if predictions are truly stale (more than MAX_LAG_SESSIONS behind).

Env controls:
- TRITON_STRICT_ASOF=1/0             (default 1) : if 1, still enforces freshness via MAX_LAG_SESSIONS
- TRITON_ASOF_COVERAGE=0.90          (default 0.90) : fraction of tickers that must have session bar to keep session_as_of
- TRITON_ASOF_MAX_LAG_SESSIONS=1     (default 1) : fail if effective_as_of is behind session_as_of by more than this many sessions
"""

from __future__ import annotations

import os
import sys
import glob
import math
import re
import json
from datetime import datetime, timezone, timedelta, date as date_type
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------
# Path bootstrap (fixes `No module named 'services'`)
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.artifacts_writer import write_heartbeat  # noqa: E402

PREDICTIONS_DIR = "data/predictions"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_WITH_RATIONALE = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
OUT_SIGNALS = os.path.join(RESULTS_DIR, "signals.csv")  # backward compatibility
ASOF_DIAG_PATH = os.path.join(RESULTS_DIR, "signals_asof_diagnostics.json")

# Legacy thresholds (still used as a fallback when feature history is too short).
BUY_DELTA = 0.002  # +0.20%
SELL_DELTA = -0.002  # -0.20%

# Feature-based signal thresholds on the composite bullishness score [0, 1].
SCORE_BUY_THRESHOLD = 0.65
SCORE_SELL_THRESHOLD = 0.35

# Composite weights (must match spec; kept here for introspection/tuning).
W_MOMENTUM = 0.35
W_TREND = 0.30
W_BREAKOUT = 0.20
W_VOLATILITY = 0.15

# Minimum number of historical bars required to compute features reliably.
MIN_FEATURE_BARS = 22  # need ma21 + 1

STRICT_ASOF = os.getenv("TRITON_STRICT_ASOF", "1").strip().lower() not in ("0", "false", "no", "")
ASOF_COVERAGE = float(os.getenv("TRITON_ASOF_COVERAGE", "0.90"))
MAX_LAG_SESSIONS = int(os.getenv("TRITON_ASOF_MAX_LAG_SESSIONS", "1"))


# -----------------------------
# Helpers
# -----------------------------
def utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now_utc_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _nyse_last_completed_session_date(now_utc: Optional[pd.Timestamp] = None) -> date_type:
    """
    Calendar-aware NYSE last completed session date using pandas_market_calendars.
    Uses market_close <= now_utc as "completed".
    """
    import pandas_market_calendars as mcal

    now_utc = now_utc or _now_utc_ts()
    nyse = mcal.get_calendar("NYSE")

    # Wide enough window to include holidays/weekends.
    start = (now_utc - pd.Timedelta(days=15)).date()
    end = (now_utc + pd.Timedelta(days=3)).date()

    sched = nyse.schedule(start_date=start, end_date=end)
    if sched is None or sched.empty:
        # Fallback: previous weekday
        d = now_utc.date()
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d

    closes = pd.to_datetime(sched["market_close"], utc=True)
    completed = sched[closes <= now_utc]
    if completed.empty:
        # before first close in window -> fallback
        d = now_utc.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d

    last = completed.iloc[-1]
    return last.name.date()


def _next_session_date(d: date_type) -> date_type:
    """
    Next NYSE session date (calendar-aware).
    """
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=d, end_date=d + timedelta(days=10))
    # first index is d if it's a session; next is what we want
    idx = list(sched.index.date)
    if not idx:
        return d + timedelta(days=1)
    if idx[0] == d and len(idx) >= 2:
        return idx[1]
    # if d wasn't a session, idx[0] is the next session
    return idx[0]


def _add_n_sessions(d: date_type, n: int) -> date_type:
    """
    Add n NYSE sessions to date d.
    """
    out = d
    for _ in range(n):
        out = _next_session_date(out)
    return out


def normalize_date(s: pd.Series) -> pd.Series:
    """
    Convert any mix of tz-aware/naive to UTC then drop tz (naive),
    then normalize to midnight (date-grain).
    """
    s = pd.to_datetime(s, errors="coerce", utc=True)
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.normalize()


def build_rationale(delta_pct: float, buy_thr: float, sell_thr: float) -> str:
    """Legacy prediction-only rationale; kept for fallback when feature history is insufficient."""
    pct = f"{delta_pct*100:.2f}%"
    if delta_pct >= buy_thr:
        return f"Predicted up {pct} vs close (>{buy_thr*100:.2f}%). Upside expected; BUY bias."
    if delta_pct <= sell_thr:
        return f"Predicted down {pct} vs close (<{sell_thr*100:.2f}%). Downside risk; SELL bias."
    return f"Predicted {pct} vs close within band; momentum unclear; HOLD."


# -----------------------------
# Feature engineering + scoring
# -----------------------------
def _load_processed_history(ticker: str) -> pd.DataFrame:
    """
    Load OHLCV history for a ticker from data/processed/{ticker}.parquet.
    Returns an empty DataFrame if the file is missing or unreadable.
    """
    path = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        # try csv
        alt = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        if not os.path.exists(alt):
            return pd.DataFrame()
        try:
            return pd.read_csv(alt)
        except Exception:
            return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _build_history_frame(pred_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Build a daily history frame (columns: date, close, [high]) for feature computation.
    Prefers processed OHLCV history; falls back to prediction-file close series.
    """
    proc = _load_processed_history(ticker)
    if not proc.empty:
        cols = {c.lower(): c for c in proc.columns}
        date_col = cols.get("date") or cols.get("datetime")
        close_col = cols.get("close")
        high_col = cols.get("high")
        if date_col and close_col:
            out = pd.DataFrame(
                {
                    "date": normalize_date(proc[date_col]),
                    "close": pd.to_numeric(proc[close_col], errors="coerce"),
                }
            )
            if high_col:
                out["high"] = pd.to_numeric(proc[high_col], errors="coerce")
            else:
                out["high"] = out["close"]
            out = out.dropna(subset=["date", "close"]).sort_values("date")
            if not out.empty:
                return out.reset_index(drop=True)

    # Fallback to predictions close history
    if "close" in pred_df.columns and "date" in pred_df.columns:
        out = pd.DataFrame(
            {
                "date": normalize_date(pred_df["date"]),
                "close": pd.to_numeric(pred_df["close"], errors="coerce"),
            }
        )
        out["high"] = out["close"]
        out = out.dropna(subset=["date", "close"]).sort_values("date")
        return out.reset_index(drop=True)

    return pd.DataFrame(columns=["date", "close", "high"])


def compute_feature_signals(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum, trend, volatility, and breakout features per row.
    Assumes `hist` is sorted ascending by date and has columns: date, close, high.
    """
    df = hist.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close

    # Momentum (pct change)
    df["returns_1"] = close.pct_change(1)
    df["returns_3"] = close.pct_change(3)
    df["returns_7"] = close.pct_change(7)

    # Trend (moving averages)
    df["ma7"] = close.rolling(window=7, min_periods=3).mean()
    df["ma21"] = close.rolling(window=21, min_periods=7).mean()
    df["trend_strength"] = (df["ma7"] - df["ma21"]) / df["ma21"].replace(0, pd.NA)

    # Volatility (rolling std of daily returns, 14-day window)
    df["volatility_14"] = df["returns_1"].rolling(window=14, min_periods=7).std()

    # Breakout (close above prior 10-day high)
    recent_high = high.rolling(window=10, min_periods=5).max().shift(1)
    df["recent_high_10"] = recent_high
    df["breakout"] = (close > recent_high).fillna(False)
    # Magnitude of breakout (0 if none, else pct above prior high)
    df["breakout_strength"] = ((close - recent_high) / recent_high.replace(0, pd.NA)).clip(lower=0)

    return df


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if pd.isna(v) or not (v == v):  # NaN check
            return default
        return v
    except Exception:
        return default


def score_components(feat_row: pd.Series) -> Dict[str, float]:
    """
    Convert the last feature row into normalized component scores in [0, 1],
    where 0.5 is neutral, >0.5 is bullish, <0.5 is bearish.
    """
    r3 = _safe_float(feat_row.get("returns_3"))
    r7 = _safe_float(feat_row.get("returns_7"))
    trend = _safe_float(feat_row.get("trend_strength"))
    vol = _safe_float(feat_row.get("volatility_14"))
    breakout = bool(feat_row.get("breakout", False))
    breakout_mag = _safe_float(feat_row.get("breakout_strength"))

    # --- Momentum score: blend 3d and 7d returns, squashed via tanh ---
    # A ~5% 3-day move or ~10% 7-day move saturates toward 1.0.
    mom_raw = 0.6 * (r3 / 0.05) + 0.4 * (r7 / 0.10)
    momentum_score = 0.5 + 0.5 * math.tanh(mom_raw)

    # --- Trend score: ma7 vs ma21 gap, saturated at ~3% divergence ---
    trend_score = 0.5 + 0.5 * math.tanh(trend / 0.03)

    # --- Volatility score: prefer moderate vol, penalize extremes ---
    # Target daily vol ~1.2% (annualized ~19%). Score peaks at target, decays outward.
    # Scoring: bell-shaped around target.
    target_vol = 0.012
    if vol <= 0:
        vol_score = 0.5  # unknown/flat -> neutral
    else:
        # Gaussian-like kernel; width 0.015
        diff = (vol - target_vol) / 0.015
        vol_score = math.exp(-0.5 * diff * diff)
        # Clamp lower bound to keep signal alive for quiet/very noisy tickers
        vol_score = max(0.2, min(1.0, vol_score))

    # --- Breakout score: strong positive when breaking out, neutral otherwise ---
    if breakout:
        # Scale breakout magnitude (2% above prior high -> saturated)
        breakout_score = 0.65 + 0.35 * math.tanh(breakout_mag / 0.02)
    else:
        # Mild penalty only if we're clearly below recent highs AND trend is weak.
        breakout_score = 0.5

    return {
        "momentum_score": float(max(0.0, min(1.0, momentum_score))),
        "trend_score": float(max(0.0, min(1.0, trend_score))),
        "volatility_score": float(max(0.0, min(1.0, vol_score))),
        "breakout_score": float(max(0.0, min(1.0, breakout_score))),
    }


def composite_score(components: Dict[str, float]) -> float:
    """Weighted blend of the four component scores, clamped to [0, 1]."""
    s = (
        W_MOMENTUM * components["momentum_score"]
        + W_TREND * components["trend_score"]
        + W_BREAKOUT * components["breakout_score"]
        + W_VOLATILITY * components["volatility_score"]
    )
    return float(max(0.0, min(1.0, s)))


def decide_signal(score: float) -> str:
    if score > SCORE_BUY_THRESHOLD:
        return "BUY"
    if score < SCORE_SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def signal_confidence(score: float) -> float:
    """
    Directional conviction in the emitted signal, in [0.5, 1.0]:
    - BUY  : confidence = score
    - SELL : confidence = 1 - score
    - HOLD : confidence = max(score, 1-score) (naturally closer to 0.5)

    This keeps the spec's `confidence = score` intent while giving strong bearish
    signals high conviction too (so downstream min-confidence gates work both ways).
    """
    score = float(max(0.0, min(1.0, score)))
    return float(max(score, 1.0 - score))


# -----------------------------
# Rationale engine
# -----------------------------
def build_enhanced_rationale(
    signal: str,
    score: float,
    components: Dict[str, float],
    feat_row: pd.Series,
    delta_pct: float,
) -> str:
    """
    Human-readable rationale focused on 1-2 dominant drivers.
    """
    mom = components["momentum_score"]
    trend = components["trend_score"]
    brk = components["breakout_score"]
    vol = components["volatility_score"]

    r7 = _safe_float(feat_row.get("returns_7"))
    trend_strength = _safe_float(feat_row.get("trend_strength"))
    volatility = _safe_float(feat_row.get("volatility_14"))
    is_breakout = bool(feat_row.get("breakout", False))

    pct_pred = f"{delta_pct * 100:+.2f}%"
    pct_r7 = f"{r7 * 100:+.2f}%"
    pct_trend = f"{trend_strength * 100:+.2f}%"

    # Pick top drivers by distance from 0.5 (conviction per component).
    drivers = sorted(
        [
            ("momentum", mom),
            ("trend", trend),
            ("breakout", brk),
            ("volatility", vol),
        ],
        key=lambda kv: abs(kv[1] - 0.5),
        reverse=True,
    )
    top_name, _ = drivers[0]
    second_name, _ = drivers[1]

    if signal == "BUY":
        # Prefer narrative combinations for strong BUYs.
        if is_breakout and mom > 0.6:
            return (
                f"Price breakout above recent highs with {pct_r7} 7d momentum; "
                f"bullish setup (model {pct_pred})."
            )
        if mom > 0.6 and trend > 0.6:
            return (
                f"Strong upward momentum ({pct_r7} over 7d) with bullish trend alignment "
                f"(MA7 {pct_trend} vs MA21). Model {pct_pred}."
            )
        if is_breakout:
            return f"Breakout above 10-day high with supportive structure; model {pct_pred}."
        if trend > 0.6:
            return (
                f"Bullish trend alignment (MA7 {pct_trend} above MA21) with constructive momentum; "
                f"model {pct_pred}."
            )
        if mom > 0.6:
            return f"Positive momentum ({pct_r7} over 7d) leading a BUY bias; model {pct_pred}."
        return (
            f"Composite score {score:.2f} above BUY threshold; "
            f"led by {top_name} and {second_name}. Model {pct_pred}."
        )

    if signal == "SELL":
        if mom < 0.4 and trend < 0.4:
            return (
                f"Downward momentum ({pct_r7} over 7d) with bearish trend (MA7 {pct_trend} "
                f"below MA21). Model {pct_pred}."
            )
        if trend < 0.4:
            return (
                f"Bearish trend alignment (MA7 {pct_trend} below MA21) pressuring price; "
                f"model {pct_pred}."
            )
        if mom < 0.4:
            return (
                f"Negative momentum ({pct_r7} over 7d) with weakening structure; model {pct_pred}."
            )
        if vol < 0.35:
            return (
                f"Elevated volatility (daily sigma {volatility * 100:.2f}%) eroding signal quality; "
                f"de-risk bias. Model {pct_pred}."
            )
        return (
            f"Composite score {score:.2f} below SELL threshold; "
            f"led by {top_name} and {second_name}. Model {pct_pred}."
        )

    # HOLD
    if abs(score - 0.5) < 0.05:
        return (
            f"Low conviction signal with mixed trend and weak momentum "
            f"(score {score:.2f}); HOLD. Model {pct_pred}."
        )
    if mom > 0.55 and trend < 0.45:
        return (
            f"Momentum positive ({pct_r7}) but trend not yet aligned; wait for confirmation. "
            f"Model {pct_pred}."
        )
    if trend > 0.55 and mom < 0.45:
        return (
            f"Trend supportive (MA7 {pct_trend}) but momentum soft; HOLD pending follow-through. "
            f"Model {pct_pred}."
        )
    return (
        f"Mixed signals: {top_name} and {second_name} diverging; "
        f"composite {score:.2f} in HOLD band. Model {pct_pred}."
    )


# -----------------------------
# Ticker validation
# -----------------------------
TICKER_RE = re.compile(r"^[A-Z0-9.\-\^]{1,15}$")  # allow ^GSPC ^VIX etc
BANNED_TICKERS = {
    "STOCK",
    "DATA",
    "ALL",
    "COMBINED",
    "MERGED",
    "PORTFOLIO",
    "RESULTS",
    "PREDICTIONS",
}


def _u(x) -> str:
    return str(x).strip().upper()


def is_valid_ticker(t: str) -> bool:
    t = _u(t)
    if not t:
        return False
    if t in BANNED_TICKERS:
        return False
    if "_" in t:
        return False
    return bool(TICKER_RE.match(t))


def infer_ticker_from_filename(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith("_predictions.parquet"):
        t = base[: -len("_predictions.parquet")]
    elif base.endswith("_predictions.csv"):
        t = base[: -len("_predictions.csv")]
    else:
        t = ""
    return _u(t)


def load_pred_file(path: str) -> pd.DataFrame:
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:
        print(f"🔥 Error reading {path}: {e}")
        return pd.DataFrame()


def resolve_ticker(path: str, df: pd.DataFrame) -> str:
    """
    Prefer ticker inside the file; otherwise infer from filename.
    Blocks ambiguous/multi-ticker files.
    """
    if isinstance(df, pd.DataFrame) and not df.empty and "ticker" in df.columns:
        tickers = df["ticker"].dropna().astype(str).map(_u).unique().tolist()
        tickers = [t for t in tickers if t]
        if len(tickers) == 1 and is_valid_ticker(tickers[0]):
            return tickers[0]
        if len(tickers) != 1:
            return ""
        return ""  # single but invalid

    t = infer_ticker_from_filename(path)
    return t if is_valid_ticker(t) else ""


# -----------------------------
# AS_OF policy
# -----------------------------
def compute_asof_policy(
    latest_by_ticker: Dict[str, date_type], session_asof: date_type
) -> Tuple[date_type, Dict[str, Any]]:
    """
    Decide effective_as_of_date based on coverage and freshness.
    """
    n = len(latest_by_ticker)
    if n == 0:
        return session_asof, {"coverage": 0.0, "reason": "no_tickers"}

    # coverage: fraction of tickers that have >= session_asof (i.e. have session bar)
    have_session = sum(1 for d in latest_by_ticker.values() if d >= session_asof)
    coverage = have_session / float(n)

    # data-supported candidate: most recent date that a large fraction has
    # We use the max of all latest dates, but also compute a "median latest" for stability.
    all_dates = sorted(latest_by_ticker.values())
    max_date = all_dates[-1]
    med_date = all_dates[n // 2]

    diag: Dict[str, Any] = {
        "session_asof_date": str(session_asof),
        "tickers": n,
        "have_session": have_session,
        "coverage": round(coverage, 4),
        "max_prediction_date": str(max_date),
        "median_prediction_date": str(med_date),
        "asof_coverage_threshold": ASOF_COVERAGE,
        "strict_asof": STRICT_ASOF,
        "max_lag_sessions": MAX_LAG_SESSIONS,
    }

    # If coverage is good -> keep strict session_asof.
    if coverage >= ASOF_COVERAGE:
        diag["effective_reason"] = "coverage_ok_use_session"
        return session_asof, diag

    # Otherwise downgrade to what the data actually supports (max_date is safest for those that DO have newer)
    # but we enforce a lag limit vs session_asof.
    effective = max_date
    lag_sessions = 0
    # count lag in session steps (approx) by walking sessions from effective->session
    # (only needed when effective < session)
    if effective < session_asof:
        # move forward from effective until >= session_asof
        cur = effective
        while cur < session_asof and lag_sessions <= 10:
            cur = _next_session_date(cur)
            lag_sessions += 1

    diag["effective_reason"] = "coverage_low_use_data_supported"
    diag["effective_asof_date"] = str(effective)
    diag["lag_sessions_vs_session"] = lag_sessions

    if STRICT_ASOF and lag_sessions > MAX_LAG_SESSIONS:
        diag["fail"] = True
    else:
        diag["fail"] = False

    return effective, diag


def _emit_signal_universe_block(
    *,
    attempted_tickers: set,
    ticker_skip_reason: Dict[str, str],
    processed_tickers: set,
    prediction_files_found: int,
) -> None:
    """
    Print the canonical [SIGNAL_UNIVERSE] diagnostic block.

    Counts are by *unique ticker label*, not by skip-event count: a ticker
    that fails first-pass parsing is counted once even if downstream code
    would have rejected it for a second reason. The first reason wins so
    the operator sees the most upstream cause.
    """
    from collections import Counter

    configured_count = len(attempted_tickers)
    n_processed = len(processed_tickers)
    n_skipped = max(0, configured_count - n_processed)
    counter = Counter(ticker_skip_reason.values())
    skip_reasons_str = "{" + ",".join(f"{k}={v}" for k, v in sorted(counter.items())) + "}"
    print(
        f"[SIGNAL_UNIVERSE] configured_tickers={configured_count} "
        f"prediction_files_found={prediction_files_found} "
        f"tickers_processed={n_processed} "
        f"tickers_skipped={n_skipped} "
        f"skip_reasons={skip_reasons_str}",
        flush=True,
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("⚙️ Generating signals with rationale...")

    now_utc = _now_utc_ts()
    session_asof = _nyse_last_completed_session_date(now_utc=now_utc)
    print(f"📌 session_as_of_date (NYSE last completed): {session_asof}")

    pred_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*_predictions.parquet"))) + sorted(
        glob.glob(os.path.join(PREDICTIONS_DIR, "*_predictions.csv"))
    )
    if not pred_files:
        print(f"🚫 No predictions found in {PREDICTIONS_DIR}. Run train_model.py first.")
        raise SystemExit(0)

    generated_at = utc_now_z()

    # ──────────────────────────────────────────────────────────────────
    # Universe diagnostics: track every ticker we attempt to process and
    # the first reason any individual ticker is dropped, so the final
    # [SIGNAL_UNIVERSE] block accurately reflects what made it through
    # vs. what was silently dropped before this instrumentation existed.
    #
    # `attempted_tickers` accumulates a stable label per file we touched
    # (resolved ticker symbol when known, otherwise filename-derived
    # fallback) so two failure modes — "we couldn't even resolve the
    # ticker" and "we resolved it but downstream parsing failed" — are
    # both visible without double-counting.
    # ──────────────────────────────────────────────────────────────────
    attempted_tickers: set[str] = set()
    ticker_skip_reason: Dict[str, str] = {}

    def _skip(label: str, reason: str) -> None:
        if not label:
            label = "<unknown>"
        attempted_tickers.add(label)
        ticker_skip_reason.setdefault(label, reason)
        print(f"[SIGNAL_SKIP] ticker={label} reason={reason}", flush=True)

    # First pass: discover latest date per ticker
    latest_by_ticker: Dict[str, date_type] = {}
    parsed_files: List[Tuple[str, str]] = []

    for path in pred_files:
        df = load_pred_file(path)
        if df.empty:
            label = infer_ticker_from_filename(path) or os.path.basename(path)
            _skip(label, "load_failed_or_empty")
            continue

        ticker = resolve_ticker(path, df)
        if not ticker:
            label = infer_ticker_from_filename(path) or os.path.basename(path)
            # Disambiguate WHY resolve_ticker dropped it — multi-ticker file,
            # invalid in-file ticker, or unparseable filename — so the
            # operator can fix the upstream producer instead of guessing.
            reason = "ticker_unresolved"
            if isinstance(df, pd.DataFrame) and "ticker" in df.columns:
                tk_uniq = (
                    df["ticker"]
                    .dropna()
                    .astype(str)
                    .map(_u)
                    .replace({"": pd.NA})
                    .dropna()
                    .unique()
                    .tolist()
                )
                if len(tk_uniq) > 1:
                    reason = f"multiple_tickers_in_file(n={len(tk_uniq)})"
                elif len(tk_uniq) == 1 and not is_valid_ticker(tk_uniq[0]):
                    reason = f"invalid_ticker_in_file({tk_uniq[0]})"
            else:
                inferred = infer_ticker_from_filename(path)
                if inferred and not is_valid_ticker(inferred):
                    reason = f"invalid_ticker_from_filename({inferred})"
            _skip(label, reason)
            continue

        attempted_tickers.add(ticker)

        if "date" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

        if "date" not in df.columns:
            _skip(ticker, "missing_date_column")
            continue

        dser = normalize_date(df["date"]).dropna()
        if dser.empty:
            _skip(ticker, "no_parseable_dates")
            continue

        latest = pd.to_datetime(dser.max(), errors="coerce")
        if pd.isna(latest):
            _skip(ticker, "max_date_nat")
            continue

        latest_by_ticker[ticker] = latest.date()
        parsed_files.append((ticker, path))

    effective_asof, diag = compute_asof_policy(latest_by_ticker, session_asof)

    diag_payload = {
        "generated_at_utc": generated_at,
        "now_utc": str(now_utc),
        **diag,
    }

    # Write diagnostics JSON (always)
    try:
        with open(ASOF_DIAG_PATH, "w", encoding="utf-8") as f:
            json.dump(diag_payload, f, indent=2)
    except Exception:
        pass

    # If strict + truly stale -> fail here (clearer than later)
    if diag_payload.get("fail", False):
        # show top stale tickers
        stale = sorted(
            [(t, d) for t, d in latest_by_ticker.items() if d < session_asof], key=lambda x: x[0]
        )
        lines = "\n".join([f"- {t} latest={d} expected={session_asof}" for t, d in stale[:50]])
        raise RuntimeError(
            "AS_OF invariant violated: predictions are too far behind the last completed NYSE session.\n"
            f"session_as_of_date={session_asof}\n"
            f"effective_as_of_date={effective_asof}\n"
            f"coverage={diag_payload.get('coverage')}\n"
            f"Stale tickers (up to 50 shown):\n{lines}\n"
            f"See: {ASOF_DIAG_PATH}\n"
            "Fix upstream: ensure train_model fetch window includes the completed bar (end-date inclusivity)."
        )

    print(f"✅ effective_as_of_date: {effective_asof} (see {ASOF_DIAG_PATH})")

    # Second pass: produce one row per ticker using latest row with date <= effective_asof
    rows_out: List[Dict[str, Any]] = []

    for ticker, path in parsed_files:
        df = load_pred_file(path)
        if df.empty:
            _skip(ticker, "reread_failed_pass2")
            continue

        if "date" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

        required = {"date", "close", "predicted_close"}
        missing = required - set(df.columns)
        if missing:
            _skip(ticker, f"missing_required_cols({','.join(sorted(missing))})")
            continue

        df = df.copy()
        df["date"] = normalize_date(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["predicted_close"] = pd.to_numeric(df["predicted_close"], errors="coerce")
        df = df.dropna(subset=["date", "close", "predicted_close"])
        if df.empty:
            _skip(ticker, "all_rows_invalid_after_coerce")
            continue
        df = df.sort_values("date")

        eff_dt = pd.Timestamp(effective_asof).normalize()
        df_upto = df[df["date"] <= eff_dt]
        if df_upto.empty:
            _skip(ticker, "no_rows_on_or_before_asof")
            continue

        latest_row = df_upto.iloc[-1]
        row_date = pd.to_datetime(latest_row["date"], errors="coerce")
        if pd.isna(row_date):
            _skip(ticker, "row_date_nat")
            continue
        row_date = row_date.normalize()

        close = float(latest_row["close"])
        pred = float(latest_row["predicted_close"])
        delta_pct = (pred / close) - 1.0

        # --- Feature-based signal generation ---
        hist = _build_history_frame(df, ticker)
        # Only look at history up to and including the effective AS_OF row.
        if not hist.empty:
            hist = hist[hist["date"] <= eff_dt].reset_index(drop=True)

        use_features = len(hist) >= MIN_FEATURE_BARS
        components: Dict[str, float]
        score: float
        feat_row: pd.Series

        if use_features:
            feats = compute_feature_signals(hist)
            feat_row = feats.iloc[-1]
            components = score_components(feat_row)
            score = composite_score(components)
            sig = decide_signal(score)
            confidence = signal_confidence(score)
            rationale = build_enhanced_rationale(sig, score, components, feat_row, delta_pct)
        else:
            # Fallback: short history -> stick with prediction-delta signal.
            feat_row = pd.Series(dtype=float)
            components = {
                "momentum_score": 0.5,
                "trend_score": 0.5,
                "volatility_score": 0.5,
                "breakout_score": 0.5,
            }
            if delta_pct >= BUY_DELTA:
                sig = "BUY"
            elif delta_pct <= SELL_DELTA:
                sig = "SELL"
            else:
                sig = "HOLD"
            # Legacy fallback confidence derived from delta magnitude (floored for usability).
            conf_raw = min(1.0, abs(delta_pct) * 50.0)  # 2% delta -> 1.0
            confidence = (
                float(max(0.5, conf_raw))
                if sig != "HOLD"
                else float(max(0.5, 0.5 + abs(delta_pct) * 10))
            )
            score = 0.5 + (0.5 if sig == "BUY" else (-0.5 if sig == "SELL" else 0.0)) * min(
                1.0, abs(delta_pct) * 50.0
            )
            rationale = build_rationale(delta_pct, BUY_DELTA, SELL_DELTA)

        # Safety: never emit NaNs.
        if pd.isna(confidence):
            confidence = 0.5
        if pd.isna(score):
            score = 0.5

        rows_out.append(
            {
                "date": row_date,
                "as_of_date": eff_dt,
                "session_as_of_date": pd.Timestamp(session_asof),
                "ticker": ticker,
                "close": close,
                "predicted_close": pred,
                "delta_pct": float(delta_pct),
                "signal": sig,
                # Keep enough precision so downstream filters see real variance across tickers.
                "confidence": float(round(float(confidence), 6)),
                "score": float(round(float(score), 6)),
                "momentum_score": float(round(components["momentum_score"], 6)),
                "trend_score": float(round(components["trend_score"], 6)),
                "breakout_score": float(round(components["breakout_score"], 6)),
                "volatility_score": float(round(components["volatility_score"], 6)),
                "rationale": rationale,
                "generated_at_utc": generated_at,
            }
        )

    if not rows_out:
        # Even a fully-empty output deserves a summary so operators see
        # WHY (e.g. all 64 tickers got skipped for the same reason).
        _emit_signal_universe_block(
            attempted_tickers=attempted_tickers,
            ticker_skip_reason=ticker_skip_reason,
            processed_tickers=set(),
            prediction_files_found=len(pred_files),
        )
        print("🚫 No signals generated (no valid prediction rows).")
        raise SystemExit(0)

    signals = pd.DataFrame(rows_out)
    signals["ticker"] = signals["ticker"].astype(str).map(_u)

    # Ticker-validation filter: surface drops as a distinct skip reason so
    # operators can spot upstream label corruption (e.g. "AAPL " or "aapl").
    pre_validation = set(signals["ticker"].unique())
    signals = signals[signals["ticker"].map(is_valid_ticker)].copy()
    post_validation = set(signals["ticker"].unique())
    for t in sorted(pre_validation - post_validation):
        _skip(t, "ticker_failed_validation")

    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    signals["as_of_date"] = pd.to_datetime(signals["as_of_date"], errors="coerce")
    signals["session_as_of_date"] = pd.to_datetime(signals["session_as_of_date"], errors="coerce")

    pre_dropna = set(signals["ticker"].unique())
    signals = signals.dropna(subset=["date", "as_of_date"])
    post_dropna = set(signals["ticker"].unique())
    for t in sorted(pre_dropna - post_dropna):
        _skip(t, "date_or_asof_invalid_post_coerce")

    signals = signals.sort_values(["ticker", "date"], kind="mergesort")

    # Write outputs
    signals.to_csv(OUT_WITH_RATIONALE, index=False)

    signals_no_rat = signals.drop(columns=["rationale"]).rename(
        columns={"confidence": "confidence_score"}
    )
    signals_no_rat.to_csv(OUT_SIGNALS, index=False)

    print(f"✅ signals_with_rationale.csv → {OUT_WITH_RATIONALE}")
    print(f"✅ signals.csv               → {OUT_SIGNALS}")
    print(f"📌 session_as_of_date        → {session_asof}")
    print(f"📌 effective_as_of_date       → {effective_asof}")
    print(f"🕒 generated_at_utc          → {generated_at}")

    print("\n📊 Signal counts (overall):")
    print(signals_no_rat["signal"].value_counts())

    processed_tickers = set(signals_no_rat["ticker"].astype(str).map(_u).unique())
    _emit_signal_universe_block(
        attempted_tickers=attempted_tickers,
        ticker_skip_reason=ticker_skip_reason,
        processed_tickers=processed_tickers,
        prediction_files_found=len(pred_files),
    )

    try:
        from services.signal_pressure_diagnostics import refresh_signal_pressure_diagnostics

        refresh_signal_pressure_diagnostics()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
        write_heartbeat(
            status="ok",
            stage="signals",
            last_success_stage="signals",
            message="Signals generated successfully (session_as_of + effective_as_of policy applied).",
        )
    except BaseException as e:
        try:
            write_heartbeat(
                status="fail",
                stage="signals",
                last_success_stage="signals",
                message="Signals generation failed.",
                error=str(e),
            )
        except Exception:
            pass
        raise
