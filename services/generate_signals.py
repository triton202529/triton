# services/generate_signals.py
"""
Generate signals + rationale from prediction files.

- Reads:  data/predictions/*_predictions.(parquet|csv)
- Writes: data/results/signals_with_rationale.csv
          data/results/signals.csv (back-compat)

Phase 1.5:
- Heartbeat ok/fail
- Windows import bootstrap

AS_OF Discipline (fail-closed session contract):
- session_as_of_date = NYSE last completed session date (ET, calendar-aware via pandas_market_calendars)
- Premarket and open-session evaluation use the latest completed XNYS session.
- After regular or early close, that newly completed session is used only when the full daily bar exists.
- Holidays and weekends use the preceding completed XNYS session. No weekday fallback.
- effective_as_of_date is always the authoritative session date. Never a data-supported substitute.
- Tradable coverage uses tickers.txt equities/ETFs only (index ^ symbols are reported separately).
- If tradable coverage < TRITON_ASOF_COVERAGE, the batch is non-actionable: no lifecycle rows.
- A missing or older bar never emits BUY/ADD/HOLD/TRIM/EXIT/WAIT; it is an explicit skip.

Env controls:
- TRITON_STRICT_ASOF=1/0             (default 1) : if 1, raise after publishing empty rows when coverage fails
- TRITON_ASOF_COVERAGE=0.90          (default 0.90) : fraction of tradable tickers that must have the expected session bar
- TRITON_ASOF_MAX_LAG_SESSIONS=1     (default 1) : retained for diagnostics; no weekday/data-supported fallback
"""

from __future__ import annotations

import os
import sys
import glob
import math
import re
import json
import time
import hashlib
from datetime import datetime, timezone, timedelta, date as date_type
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------
# Path bootstrap (fixes `No module named 'services'`)
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Windows cp1252 consoles cannot encode emoji in status prints without UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from services.artifacts_writer import write_heartbeat  # noqa: E402

PREDICTIONS_DIR = "data/predictions"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "data/results"
TICKERS_PATH = os.path.join("data", "config", "tickers.txt")

OUT_WITH_RATIONALE = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
OUT_SIGNALS = os.path.join(RESULTS_DIR, "signals.csv")  # backward compatibility
ASOF_DIAG_PATH = os.path.join(RESULTS_DIR, "signals_asof_diagnostics.json")
INDEX_PREFIX = "^"
SKIP_MISSING_EXPECTED_SESSION_BAR = "MISSING_EXPECTED_SESSION_BAR"
SKIP_MISSING_PREDICTION_FILE = "MISSING_PREDICTION_FILE"
SKIP_EMPTY_PREDICTION_FILE = "EMPTY_PREDICTION_FILE"
SKIP_FUTURE_DATED_BAR = "FUTURE_DATED_BAR"
SKIP_DUPLICATE_DATE = "DUPLICATE_DATE"
SKIP_MALFORMED_DATE = "MALFORMED_DATE"
SKIP_INDEX_ONLY_EXCLUDED = "INDEX_ONLY_EXCLUDED"
SKIP_BATCH_COVERAGE_BELOW_THRESHOLD = "BATCH_COVERAGE_BELOW_THRESHOLD"
SKIP_INVALID_PROCESSED_SESSION_BAR = "INVALID_PROCESSED_SESSION_BAR"
_OPERATIONAL_REQUIRED_COLS = {
    "date",
    "as_of_date",
    "session_as_of_date",
    "ticker",
    "close",
    "predicted_close",
    "delta_pct",
    "signal",
    "confidence",
    "score",
    "rationale",
    "generated_at_utc",
}


def _unique_sibling_tmp(target: str) -> str:
    abs_target = os.path.abspath(target)
    parent = os.path.dirname(abs_target) or "."
    name = os.path.basename(abs_target)
    return os.path.join(parent, f".{name}.{os.getpid()}.{time.time_ns()}.tmp")


def atomic_publish_csv(df: pd.DataFrame, target: str) -> None:
    """Write CSV to a unique sibling tmp, flush, then os.replace onto target."""
    abs_target = os.path.abspath(target)
    parent = os.path.dirname(abs_target) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = _unique_sibling_tmp(abs_target)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, abs_target)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def publish_operational_signal_csvs(signals: pd.DataFrame) -> None:
    """Publish both operational CSVs only after the complete frame is valid.

    If the rationale file is published and the legacy file then fails, this
    raises explicitly and does not report full-stage success.
    """
    if not signals.empty:
        missing = _OPERATIONAL_REQUIRED_COLS - set(signals.columns)
        if missing:
            raise ValueError(f"operational signals schema missing: {sorted(missing)}")
        tickers = signals["ticker"].astype(str)
        if tickers.duplicated().any():
            raise ValueError("operational signals must have one current row per ticker")
    if "rationale" in signals.columns:
        signals_no_rat = signals.drop(columns=["rationale"]).rename(
            columns={"confidence": "confidence_score"}
        )
    else:
        signals_no_rat = signals.copy()
    atomic_publish_csv(signals, OUT_WITH_RATIONALE)
    try:
        atomic_publish_csv(signals_no_rat, OUT_SIGNALS)
    except Exception as exc:
        raise RuntimeError(
            "signals.csv atomic publish failed after signals_with_rationale.csv "
            f"was already published: {exc}"
        ) from exc


# Legacy thresholds retained for diagnostics only. Short-history fallback no longer
# emits BUY/SELL from Random Forest residual; RF is shadow-only.
BUY_DELTA = 0.002  # +0.20%
SELL_DELTA = -0.002  # -0.20%
RF_SHADOW_ONLY = True
RF_EVIDENCE_VALID = "valid"
RF_EVIDENCE_STALE = "stale"
RF_EVIDENCE_UNAVAILABLE = "unavailable"
RF_EVIDENCE_MALFORMED = "malformed"

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
def _finite_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def classify_rf_shadow_evidence(
    *,
    predicted_close_raw: Any,
    close: Any,
    row_date: Any,
    session_asof: date_type,
    column_present: bool,
) -> Dict[str, Any]:
    """Classify Random Forest residual as diagnostic shadow evidence only.

    Never returns a synthetic prediction. Missing/malformed values stay
    unavailable; a finite prediction that is older than the session is stale.
    """
    ts = ""
    row_dt = pd.to_datetime(row_date, errors="coerce")
    if not pd.isna(row_dt):
        try:
            ts = str(pd.Timestamp(row_dt).normalize().date())
        except Exception:
            ts = str(row_dt)

    pred = _finite_float(predicted_close_raw)
    close_f = _finite_float(close)
    rf_delta: Optional[float] = None
    if pred is not None and close_f is not None and close_f != 0.0:
        rf_delta = (pred / close_f) - 1.0
        if not math.isfinite(rf_delta):
            rf_delta = None

    status = RF_EVIDENCE_UNAVAILABLE
    if not column_present:
        status = RF_EVIDENCE_UNAVAILABLE
    elif predicted_close_raw is None or (
        isinstance(predicted_close_raw, float) and not math.isfinite(predicted_close_raw)
    ):
        status = RF_EVIDENCE_UNAVAILABLE
    elif pred is None:
        status = RF_EVIDENCE_MALFORMED
    elif close_f is None or close_f == 0.0 or rf_delta is None:
        status = RF_EVIDENCE_MALFORMED
    else:
        status = RF_EVIDENCE_VALID
        if not pd.isna(row_dt):
            try:
                rd = pd.Timestamp(row_dt).normalize().date()
                if rd < session_asof:
                    status = RF_EVIDENCE_STALE
            except Exception:
                status = RF_EVIDENCE_STALE

    return {
        "rf_predicted_close": pred if pred is not None else float("nan"),
        "rf_delta_pct": rf_delta if rf_delta is not None else float("nan"),
        "rf_evidence_timestamp": ts,
        "rf_evidence_status": status,
        "rf_shadow_only": True,
    }


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
        raise RuntimeError(
            "NYSE calendar returned no sessions in window; weekday fallback is forbidden."
        )

    closes = pd.to_datetime(sched["market_close"], utc=True)
    completed = sched[closes <= now_utc]
    if completed.empty:
        raise RuntimeError(
            "No completed NYSE session in calendar window; weekday fallback is forbidden."
        )

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


def _processed_path_for_ticker(ticker: str) -> Optional[str]:
    path = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
    if os.path.isfile(path):
        return path
    alt = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    if os.path.isfile(alt):
        return alt
    return None


class _ProcessedSessionView:
    """Single validity decision for coverage counting and operational emission."""

    __slots__ = ("ok", "path", "close", "hist", "reason", "actual")

    def __init__(
        self,
        *,
        ok: bool,
        path: Optional[str],
        close: Optional[float],
        hist: Optional[pd.DataFrame],
        reason: Optional[str],
        actual: str,
    ) -> None:
        self.ok = ok
        self.path = path
        self.close = close
        self.hist = hist
        self.reason = reason
        self.actual = actual


def _session_ohlcv_skip_reason(row: pd.Series) -> Optional[str]:
    """Reject missing, nonfinite, nonpositive, or same-basis OHLC-shape failures.

    Equality is valid. Prices are not normalized or synthesized.
    """
    open_v = _finite_float(row.get("open"))
    high_v = _finite_float(row.get("high"))
    low_v = _finite_float(row.get("low"))
    close_v = _finite_float(row.get("close"))
    volume_v = _finite_float(row.get("volume"))
    if open_v is None or high_v is None or low_v is None or close_v is None or volume_v is None:
        return SKIP_INVALID_PROCESSED_SESSION_BAR
    if open_v <= 0.0 or high_v <= 0.0 or low_v <= 0.0 or close_v <= 0.0 or volume_v < 0.0:
        return SKIP_INVALID_PROCESSED_SESSION_BAR
    if high_v < open_v or high_v < close_v or low_v > open_v or low_v > close_v or high_v < low_v:
        return SKIP_INVALID_PROCESSED_SESSION_BAR
    return None


def _inspect_processed_session(ticker: str, session_asof: date_type) -> _ProcessedSessionView:
    """Shared first-pass / second-pass processed-bar contract.

    RF prediction files are never consulted. Invalid evidence is not repaired.
    """
    path = _processed_path_for_ticker(ticker)
    missing = _ProcessedSessionView(
        ok=False,
        path=path,
        close=None,
        hist=None,
        reason=SKIP_MISSING_EXPECTED_SESSION_BAR,
        actual="MISSING",
    )
    if not path:
        return missing
    try:
        proc = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    except Exception:
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=SKIP_INVALID_PROCESSED_SESSION_BAR,
            actual="MISSING",
        )
    if proc is None or proc.empty:
        return missing
    cols = {str(c).lower(): c for c in proc.columns}
    date_col = cols.get("date") or cols.get("datetime")
    if not date_col:
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=SKIP_MALFORMED_DATE,
            actual="MISSING",
        )
    work = pd.DataFrame({"date": normalize_date(proc[date_col])})
    for name in ("open", "high", "low", "close", "volume"):
        src = cols.get(name)
        work[name] = pd.to_numeric(proc[src], errors="coerce") if src else float("nan")
    dated = work.dropna(subset=["date"])
    if dated.empty:
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=SKIP_MALFORMED_DATE,
            actual="MISSING",
        )
    if bool(dated["date"].duplicated().any()):
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=SKIP_DUPLICATE_DATE,
            actual=str(pd.Timestamp(dated["date"].max()).date()),
        )
    sess_ts = pd.Timestamp(session_asof).normalize()
    clipped = dated[dated["date"] <= sess_ts].sort_values("date")
    latest = str(pd.Timestamp(clipped["date"].max()).date()) if not clipped.empty else "MISSING"
    exact = clipped[clipped["date"] == sess_ts]
    if exact.empty:
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=SKIP_MISSING_EXPECTED_SESSION_BAR,
            actual=latest,
        )
    if len(exact) > 1:
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=SKIP_DUPLICATE_DATE,
            actual=str(session_asof),
        )
    row = exact.iloc[-1]
    close_v = _finite_float(row.get("close"))
    if close_v is None:
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=SKIP_MISSING_EXPECTED_SESSION_BAR,
            actual=str(session_asof),
        )
    shape_reason = _session_ohlcv_skip_reason(row)
    if shape_reason:
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=shape_reason,
            actual=str(session_asof),
        )
    feature_hist = clipped.dropna(subset=["close"]).copy()
    if "high" not in feature_hist.columns:
        feature_hist["high"] = feature_hist["close"]
    else:
        feature_hist["high"] = feature_hist["high"].where(
            feature_hist["high"].notna(), feature_hist["close"]
        )
    if len(feature_hist) < MIN_FEATURE_BARS:
        return _ProcessedSessionView(
            ok=False,
            path=path,
            close=None,
            hist=None,
            reason=SKIP_INVALID_PROCESSED_SESSION_BAR,
            actual=str(session_asof),
        )
    close_v = float(row["close"])
    return _ProcessedSessionView(
        ok=True,
        path=path,
        close=close_v,
        hist=feature_hist.reset_index(drop=True),
        reason=None,
        actual=str(session_asof),
    )


def _processed_session_close(
    ticker: str, session_asof: date_type
) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[str]]:
    """
    Return (session_ts, close, source_path) when processed has a valid session bar.

    Used so operational feature freshness does not depend on a current RF prediction
    container. RF remains diagnostic-only and is never inferred from this path.
    """
    view = _inspect_processed_session(ticker, session_asof)
    if not view.ok or view.close is None or not view.path:
        return None, None, None
    return pd.Timestamp(session_asof).normalize(), float(view.close), view.path


def _processed_session_history(
    ticker: str, session_asof: date_type
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[float]]:
    """Return (OHLCV history clipped to session, path, session close).

    Requires the exact completed-session bar and the same validity decision as
    coverage. Bars after session_asof are dropped. Invalid bars are not repaired.
    """
    view = _inspect_processed_session(ticker, session_asof)
    if not view.ok or view.hist is None or view.close is None or not view.path:
        return None, None, None
    return view.hist, view.path, float(view.close)


def _processed_actual_latest(ticker: str, session_asof: date_type) -> str:
    """Latest processed bar date on or before session, else MISSING."""
    path = _processed_path_for_ticker(ticker)
    if not path:
        return "MISSING"
    try:
        proc = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    except Exception:
        return "MISSING"
    if proc is None or proc.empty:
        return "MISSING"
    cols = {str(c).lower(): c for c in proc.columns}
    date_col = cols.get("date") or cols.get("datetime")
    if not date_col:
        return "MISSING"
    dts = normalize_date(proc[date_col]).dropna()
    sess_ts = pd.Timestamp(session_asof).normalize()
    dts = dts[dts <= sess_ts]
    if dts.empty:
        return "MISSING"
    return str(pd.Timestamp(dts.max()).date())


def _rf_shadow_from_pred_df(
    pred_df: Optional[pd.DataFrame],
    *,
    session_asof: date_type,
    close: float,
) -> Dict[str, Any]:
    """Classify RF residual from an optional prediction frame. Never infers values."""
    if pred_df is None or pred_df.empty:
        return classify_rf_shadow_evidence(
            predicted_close_raw=None,
            close=close,
            row_date="",
            session_asof=session_asof,
            column_present=False,
        )
    work = pred_df.copy()
    if "date" not in work.columns and "Date" in work.columns:
        work = work.rename(columns={"Date": "date"})
    if "date" not in work.columns:
        return classify_rf_shadow_evidence(
            predicted_close_raw=None,
            close=close,
            row_date="",
            session_asof=session_asof,
            column_present=False,
        )
    work["date"] = normalize_date(work["date"])
    sess_ts = pd.Timestamp(session_asof).normalize()
    work = work.dropna(subset=["date"])
    work = work[work["date"] <= sess_ts]
    rf_col = "predicted_close" in work.columns
    if rf_col:
        work["predicted_close"] = pd.to_numeric(work["predicted_close"], errors="coerce")
    if work.empty:
        return classify_rf_shadow_evidence(
            predicted_close_raw=None,
            close=close,
            row_date="",
            session_asof=session_asof,
            column_present=rf_col,
        )
    exact = work[work["date"] == sess_ts]
    row = (exact if not exact.empty else work.sort_values("date")).iloc[-1]
    pred_raw = row["predicted_close"] if rf_col else None
    return classify_rf_shadow_evidence(
        predicted_close_raw=pred_raw,
        close=close,
        row_date=row["date"],
        session_asof=session_asof,
        column_present=rf_col,
    )


def _score_from_history(
    hist: pd.DataFrame, sess_ts: pd.Timestamp
) -> Tuple[str, float, float, Dict[str, float], pd.Series, str]:
    """Feature signal from history clipped to the completed session. RF is not used."""
    if not hist.empty:
        hist = hist[hist["date"] <= sess_ts].reset_index(drop=True)
    operational_delta_pct = float("nan")
    if len(hist) >= MIN_FEATURE_BARS:
        feats = compute_feature_signals(hist)
        feat_row = feats.iloc[-1]
        components = score_components(feat_row)
        score = composite_score(components)
        sig = decide_signal(score)
        confidence = signal_confidence(score)
        rationale = build_enhanced_rationale(
            sig, score, components, feat_row, operational_delta_pct
        )
    else:
        feat_row = pd.Series(dtype=float)
        components = {
            "momentum_score": 0.5,
            "trend_score": 0.5,
            "volatility_score": 0.5,
            "breakout_score": 0.5,
        }
        sig = "HOLD"
        confidence = 0.5
        score = 0.5
        rationale = "Insufficient feature history; HOLD. Random Forest residual is shadow-only."
    if pd.isna(confidence):
        confidence = 0.5
    if pd.isna(score):
        score = 0.5
    return sig, float(confidence), float(score), components, feat_row, rationale


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
            return f"Price breakout above recent highs with {pct_r7} 7d momentum; bullish setup."
        if mom > 0.6 and trend > 0.6:
            return (
                f"Strong upward momentum ({pct_r7} over 7d) with bullish trend alignment "
                f"(MA7 {pct_trend} vs MA21)."
            )
        if is_breakout:
            return "Breakout above 10-day high with supportive structure."
        if trend > 0.6:
            return (
                f"Bullish trend alignment (MA7 {pct_trend} above MA21) with constructive momentum."
            )
        if mom > 0.6:
            return f"Positive momentum ({pct_r7} over 7d) leading a BUY bias."
        return (
            f"Composite score {score:.2f} above BUY threshold; "
            f"led by {top_name} and {second_name}."
        )

    if signal == "SELL":
        if mom < 0.4 and trend < 0.4:
            return (
                f"Downward momentum ({pct_r7} over 7d) with bearish trend (MA7 {pct_trend} "
                f"below MA21)."
            )
        if trend < 0.4:
            return f"Bearish trend alignment (MA7 {pct_trend} below MA21) pressuring price."
        if mom < 0.4:
            return f"Negative momentum ({pct_r7} over 7d) with weakening structure."
        if vol < 0.35:
            return (
                f"Elevated volatility (daily sigma {volatility * 100:.2f}%) eroding signal quality; "
                f"de-risk bias."
            )
        return (
            f"Composite score {score:.2f} below SELL threshold; "
            f"led by {top_name} and {second_name}."
        )

    # HOLD
    if abs(score - 0.5) < 0.05:
        return (
            f"Low conviction signal with mixed trend and weak momentum "
            f"(score {score:.2f}); HOLD."
        )
    if mom > 0.55 and trend < 0.45:
        return f"Momentum positive ({pct_r7}) but trend not yet aligned; wait for confirmation."
    if trend > 0.55 and mom < 0.45:
        return f"Trend supportive (MA7 {pct_trend}) but momentum soft; HOLD pending follow-through."
    return (
        f"Mixed signals: {top_name} and {second_name} diverging; "
        f"composite {score:.2f} in HOLD band."
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


def _is_index_ticker(ticker: str) -> bool:
    return str(ticker or "").startswith(INDEX_PREFIX)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional_tickers(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    seen: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            token = str(line).strip().upper()
            if not token or token.startswith("#"):
                continue
            if token not in seen:
                seen.append(token)
    return seen


def _empty_operational_frame() -> pd.DataFrame:
    cols = list(_OPERATIONAL_REQUIRED_COLS) + [
        "momentum_score",
        "trend_score",
        "breakout_score",
        "volatility_score",
        "rf_predicted_close",
        "rf_delta_pct",
        "rf_evidence_timestamp",
        "rf_evidence_status",
        "rf_shadow_only",
    ]
    ordered: List[str] = []
    for col in cols:
        if col not in ordered:
            ordered.append(col)
    return pd.DataFrame(columns=ordered)


# -----------------------------
# AS_OF policy
# -----------------------------
def compute_asof_policy(
    latest_by_ticker: Dict[str, date_type],
    session_asof: date_type,
    *,
    tradable_tickers: Optional[List[str]] = None,
    index_tickers: Optional[List[str]] = None,
) -> Tuple[date_type, Dict[str, Any]]:
    """
    Authoritative expected session is always session_asof.

    Coverage is tradable-only. Below ASOF_COVERAGE the batch is non-actionable.
    There is no data-supported date substitution.
    """
    index_set = {str(t).strip().upper() for t in (index_tickers or []) if t}
    if tradable_tickers is None:
        tradable_list = [
            t for t in latest_by_ticker.keys() if not _is_index_ticker(t) and t not in index_set
        ]
    else:
        tradable_list = [str(t).strip().upper() for t in tradable_tickers if t]
        tradable_list = [t for t in tradable_list if t and not _is_index_ticker(t)]

    if index_tickers is None:
        index_list = [t for t in latest_by_ticker.keys() if _is_index_ticker(t)]
    else:
        index_list = [str(t).strip().upper() for t in index_tickers if t]

    n = len(tradable_list)
    have_session = sum(1 for t in tradable_list if latest_by_ticker.get(t) == session_asof)
    coverage = (have_session / float(n)) if n else 0.0

    all_dates = sorted(latest_by_ticker.values()) if latest_by_ticker else []
    max_date = all_dates[-1] if all_dates else None
    med_date = all_dates[len(all_dates) // 2] if all_dates else None

    batch_actionable = bool(n > 0 and coverage >= ASOF_COVERAGE)
    diag: Dict[str, Any] = {
        "session_asof_date": str(session_asof),
        "tickers": n,
        "governed_tickers": n + len(index_list),
        "tradable_tickers": n,
        "index_tickers": len(index_list),
        "have_session": have_session,
        "coverage": round(coverage, 4),
        "max_prediction_date": str(max_date) if max_date is not None else None,
        "median_prediction_date": str(med_date) if med_date is not None else None,
        "asof_coverage_threshold": ASOF_COVERAGE,
        "strict_asof": STRICT_ASOF,
        "max_lag_sessions": MAX_LAG_SESSIONS,
        "batch_actionable": batch_actionable,
        "effective_asof_date": str(session_asof),
        "lag_sessions_vs_session": 0,
    }

    if n == 0:
        diag["effective_reason"] = "no_tradable_tickers"
        diag["fail"] = bool(STRICT_ASOF)
        return session_asof, diag

    if batch_actionable:
        diag["effective_reason"] = "coverage_ok_use_session"
        diag["fail"] = False
        return session_asof, diag

    diag["effective_reason"] = "coverage_low_batch_unavailable"
    diag["fail"] = bool(STRICT_ASOF)
    return session_asof, diag


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
    # Prediction files are optional RF shadow evidence. A missing predictions
    # directory must not prevent operational rows from current processed bars.

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
    skip_records: List[Dict[str, Any]] = []
    source_hash_by_path: Dict[str, str] = {}
    source_hash_by_ticker: Dict[str, str] = {}
    current_tradable_symbols: List[str] = []
    current_index_symbols: List[str] = []

    def _file_hash(path: Optional[str]) -> str:
        if not path:
            return ""
        if path not in source_hash_by_path:
            try:
                source_hash_by_path[path] = _sha256_file(path)
            except OSError:
                source_hash_by_path[path] = ""
        return source_hash_by_path[path]

    def _skip(
        label: str,
        reason: str,
        *,
        actual: str = "MISSING",
        expected: Optional[date_type] = None,
        source_path: Optional[str] = None,
        source_sha256: Optional[str] = None,
        symbol_class: Optional[str] = None,
    ) -> None:
        if not label:
            label = "<unknown>"
        attempted_tickers.add(label)
        if label in ticker_skip_reason:
            return
        ticker_skip_reason[label] = reason
        sha = source_sha256 if source_sha256 is not None else _file_hash(source_path)
        if label and sha:
            source_hash_by_ticker.setdefault(label, sha)
        rec = {
            "symbol": label,
            "expected_session_date": str(expected if expected is not None else session_asof),
            "actual_latest_bar_date": actual if actual else "MISSING",
            "reason": reason,
            "source_sha256": sha or "",
            "evidence_timestamp": generated_at,
            "class": symbol_class or ("index" if _is_index_ticker(label) else "tradable"),
        }
        skip_records.append(rec)
        print(
            f"[SIGNAL_SKIP] ticker={label} reason={reason} "
            f"expected={rec['expected_session_date']} actual={rec['actual_latest_bar_date']} "
            f"source_sha256={rec['source_sha256']} evidence_ts={generated_at}",
            flush=True,
        )

    def _persist_diag(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "generated_at_utc": generated_at,
            "now_utc": str(now_utc),
            **diag,
            "per_symbol_skips": skip_records,
            "current_tradable_symbols": list(current_tradable_symbols),
            "current_index_symbols": list(current_index_symbols),
            "skip_count": len(skip_records),
        }
        if extra:
            payload.update(extra)
        try:
            os.makedirs(os.path.dirname(ASOF_DIAG_PATH) or ".", exist_ok=True)
            with open(ASOF_DIAG_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception:
            pass
        return payload

    governed = _load_optional_tickers(TICKERS_PATH)
    tradable_universe = [t for t in governed if not _is_index_ticker(t)]
    index_universe = [t for t in governed if _is_index_ticker(t)]

    # First pass: discover latest date per ticker
    latest_by_ticker: Dict[str, date_type] = {}
    parsed_files: List[Tuple[str, str]] = []

    pred_df_by_ticker: Dict[str, pd.DataFrame] = {}
    pred_error_by_ticker: Dict[str, str] = {}

    def _processed_session_ok(label: str) -> bool:
        ticker = _u(label)
        if not is_valid_ticker(ticker):
            return False
        hit, close_v, _path = _processed_session_close(ticker, session_asof)
        return hit is not None and close_v is not None

    def _rf_file_issue(
        label: str,
        reason: str,
        *,
        actual: str = "MISSING",
        source_path: Optional[str] = None,
        source_sha256: Optional[str] = None,
    ) -> None:
        """RF-container defects never skip a current processed session bar."""
        label_u = _u(label) if label else ""
        if label_u:
            pred_error_by_ticker[label_u] = reason
        if label_u and _processed_session_ok(label_u):
            attempted_tickers.add(label_u)
            return
        _skip(
            label,
            reason,
            actual=actual,
            source_path=source_path,
            source_sha256=source_sha256,
        )

    for path in pred_files:
        file_sha = _file_hash(path)
        df = load_pred_file(path)
        if df.empty:
            label = infer_ticker_from_filename(path) or os.path.basename(path)
            _rf_file_issue(
                label,
                SKIP_EMPTY_PREDICTION_FILE,
                actual="MISSING",
                source_path=path,
                source_sha256=file_sha,
            )
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
            _rf_file_issue(label, reason, source_path=path, source_sha256=file_sha)
            continue

        attempted_tickers.add(ticker)
        source_hash_by_ticker[ticker] = file_sha

        if "date" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

        if "date" not in df.columns:
            _rf_file_issue(ticker, "missing_date_column", source_path=path, source_sha256=file_sha)
            continue

        dser = normalize_date(df["date"]).dropna()
        if dser.empty:
            _rf_file_issue(
                ticker,
                SKIP_MALFORMED_DATE,
                actual="MISSING",
                source_path=path,
                source_sha256=file_sha,
            )
            continue

        latest = pd.to_datetime(dser.max(), errors="coerce")
        if pd.isna(latest):
            _rf_file_issue(
                ticker,
                SKIP_MALFORMED_DATE,
                actual="MISSING",
                source_path=path,
                source_sha256=file_sha,
            )
            continue

        parsed_files.append((ticker, path))
        pred_df_by_ticker[ticker] = df

    if not governed:
        discovered: List[str] = []
        if os.path.isdir(PROCESSED_DIR):
            for name in sorted(os.listdir(PROCESSED_DIR)):
                if not (name.endswith(".parquet") or name.endswith(".csv")):
                    continue
                stem = _u(name.rsplit(".", 1)[0])
                if is_valid_ticker(stem) and stem not in discovered:
                    discovered.append(stem)
        tradable_universe = [t for t in discovered if not _is_index_ticker(t)]
        index_universe = [t for t in discovered if _is_index_ticker(t)]

    # Operational freshness is processed-only. Prediction dates never raise coverage.
    for ticker in sorted(set(tradable_universe + index_universe)):
        sess_hit, close_v, _proc_path = _processed_session_close(ticker, session_asof)
        if sess_hit is not None and close_v is not None:
            latest_by_ticker[ticker] = session_asof
            attempted_tickers.add(ticker)
            continue
        actual = _processed_actual_latest(ticker, session_asof)
        if actual and actual != "MISSING":
            try:
                actual_d = date_type.fromisoformat(str(actual)[:10])
            except ValueError:
                actual_d = None
            if actual_d is not None and actual_d != session_asof:
                latest_by_ticker[ticker] = actual_d
            attempted_tickers.add(ticker)

    for label in tradable_universe + index_universe:
        if label in latest_by_ticker or label in attempted_tickers:
            continue
        _skip(
            label,
            SKIP_MISSING_EXPECTED_SESSION_BAR,
            actual="MISSING",
            symbol_class="index" if _is_index_ticker(label) else "tradable",
        )

    effective_asof, diag = compute_asof_policy(
        latest_by_ticker,
        session_asof,
        tradable_tickers=tradable_universe or None,
        index_tickers=index_universe or None,
    )
    batch_actionable = bool(diag.get("batch_actionable"))
    diag_payload = _persist_diag()

    print(f"✅ effective_as_of_date: {effective_asof} (see {ASOF_DIAG_PATH})")

    # Second pass: governed universe. Processed session bar is operational source.
    rows_out: List[Dict[str, Any]] = []
    operational_provenance: List[Dict[str, Any]] = []
    sess_ts = pd.Timestamp(session_asof).normalize()
    emit_order = list(tradable_universe) + list(index_universe)

    def _append_row(
        *,
        ticker: str,
        close: float,
        hist: pd.DataFrame,
        rf_shadow: Dict[str, Any],
        row_date: pd.Timestamp,
        source_path: str,
        file_sha: str,
        source_kind: str,
    ) -> None:
        current_tradable_symbols.append(ticker)
        if not batch_actionable:
            _skip(
                ticker,
                SKIP_BATCH_COVERAGE_BELOW_THRESHOLD,
                actual=str(session_asof),
                source_path=source_path,
                source_sha256=file_sha,
            )
            return
        sig, confidence, score, components, _feat_row, rationale = _score_from_history(
            hist, sess_ts
        )
        pred_display = rf_shadow["rf_predicted_close"]
        rows_out.append(
            {
                "date": row_date,
                "as_of_date": sess_ts,
                "session_as_of_date": sess_ts,
                "ticker": ticker,
                "close": close,
                "predicted_close": pred_display,
                "delta_pct": float("nan"),
                "signal": sig,
                "confidence": float(round(float(confidence), 6)),
                "score": float(round(float(score), 6)),
                "momentum_score": float(round(components["momentum_score"], 6)),
                "trend_score": float(round(components["trend_score"], 6)),
                "breakout_score": float(round(components["breakout_score"], 6)),
                "volatility_score": float(round(components["volatility_score"], 6)),
                "rationale": rationale,
                "generated_at_utc": generated_at,
                "rf_predicted_close": rf_shadow["rf_predicted_close"],
                "rf_delta_pct": rf_shadow["rf_delta_pct"],
                "rf_evidence_timestamp": rf_shadow["rf_evidence_timestamp"],
                "rf_evidence_status": rf_shadow["rf_evidence_status"],
                "rf_shadow_only": True,
            }
        )
        operational_provenance.append(
            {
                "ticker": ticker,
                "field_group": "operational_signal",
                "session_date": str(session_asof),
                "input_date": str(pd.Timestamp(row_date).date()),
                "source_path": source_path,
                "source_sha256": file_sha,
                "source_kind": source_kind,
                "close": close,
                "signal": sig,
                "confidence": float(round(float(confidence), 6)),
            }
        )

    for ticker in emit_order:
        proc_hist, proc_path, proc_close = _processed_session_history(ticker, session_asof)
        pred_path = next((p for t, p in parsed_files if t == ticker), None)
        pred_df = pred_df_by_ticker.get(ticker)
        file_sha = source_hash_by_ticker.get(ticker, "")
        if pred_path:
            file_sha = file_sha or _file_hash(pred_path)

        if proc_hist is not None and proc_path and proc_close is not None:
            file_sha = _file_hash(proc_path) or file_sha
            source_hash_by_ticker[ticker] = file_sha
            attempted_tickers.add(ticker)
            if _is_index_ticker(ticker):
                current_index_symbols.append(ticker)
                _skip(
                    ticker,
                    SKIP_INDEX_ONLY_EXCLUDED,
                    actual=str(session_asof),
                    source_path=proc_path,
                    source_sha256=file_sha,
                    symbol_class="index",
                )
                continue
            rf_shadow = _rf_shadow_from_pred_df(
                pred_df, session_asof=session_asof, close=float(proc_close)
            )
            _append_row(
                ticker=ticker,
                close=float(proc_close),
                hist=proc_hist,
                rf_shadow=rf_shadow,
                row_date=sess_ts,
                source_path=proc_path,
                file_sha=file_sha,
                source_kind="processed",
            )
            continue

        # No valid current processed session bar: fail closed. Prediction OHLCV is never operational.
        view = _inspect_processed_session(ticker, session_asof)
        actual = view.actual or _processed_actual_latest(ticker, session_asof)
        if _is_index_ticker(ticker):
            current_index_symbols.append(ticker)
            _skip(
                ticker,
                SKIP_INDEX_ONLY_EXCLUDED,
                actual=actual if actual != "MISSING" else "MISSING",
                source_path=view.path,
                source_sha256=_file_hash(view.path) if view.path else "",
                symbol_class="index",
            )
            continue
        _skip(
            ticker,
            view.reason or SKIP_MISSING_EXPECTED_SESSION_BAR,
            actual=actual,
            source_path=view.path,
            source_sha256=_file_hash(view.path) if view.path else "",
        )

    def _publish_empty_and_maybe_fail(message: str) -> None:
        empty = _empty_operational_frame()
        publish_operational_signal_csvs(empty)
        _persist_diag(
            {
                "operational_rows": 0,
                "batch_actionable": False,
            }
        )
        _emit_signal_universe_block(
            attempted_tickers=attempted_tickers,
            ticker_skip_reason=ticker_skip_reason,
            processed_tickers=set(),
            prediction_files_found=len(pred_files),
        )
        print(message)
        if diag.get("fail", False):
            stale = sorted(
                [(t, d) for t, d in latest_by_ticker.items() if d != session_asof],
                key=lambda x: x[0],
            )
            lines = "\n".join([f"- {t} latest={d} expected={session_asof}" for t, d in stale[:50]])
            raise RuntimeError(
                "AS_OF invariant violated: predictions are too far behind the last completed NYSE session.\n"
                f"session_as_of_date={session_asof}\n"
                f"effective_as_of_date={effective_asof}\n"
                f"coverage={diag.get('coverage')}\n"
                f"Stale tickers (up to 50 shown):\n{lines}\n"
                f"See: {ASOF_DIAG_PATH}\n"
                "Fix upstream: ensure train_model fetch window includes the completed bar (end-date inclusivity)."
            )
        raise SystemExit(0)

    if (not batch_actionable) or (not rows_out):
        _publish_empty_and_maybe_fail(
            "🚫 No lifecycle-authoritative signals generated (freshness contract fail-closed)."
        )

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

    bar_dates = pd.to_datetime(signals["date"], errors="coerce").dt.normalize()
    as_of_dates = pd.to_datetime(signals["as_of_date"], errors="coerce").dt.normalize()
    session_dates = pd.to_datetime(signals["session_as_of_date"], errors="coerce").dt.normalize()
    if (
        not bar_dates.eq(as_of_dates).all()
        or not as_of_dates.eq(session_dates).all()
        or not as_of_dates.eq(sess_ts).all()
    ):
        raise RuntimeError("freshness contract: refused to publish relabeled or non-session bars")

    # Publish only after the complete operational frame is built and valid.
    publish_operational_signal_csvs(signals)
    _persist_diag(
        {
            "operational_rows": int(len(signals)),
            "batch_actionable": True,
            "operational_provenance": operational_provenance,
        }
    )
    signals_no_rat = signals.drop(columns=["rationale"]).rename(
        columns={"confidence": "confidence_score"}
    )

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
