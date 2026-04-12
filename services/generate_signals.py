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
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_WITH_RATIONALE = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
OUT_SIGNALS = os.path.join(RESULTS_DIR, "signals.csv")  # backward compatibility
ASOF_DIAG_PATH = os.path.join(RESULTS_DIR, "signals_asof_diagnostics.json")

# Signal thresholds (as pct moves vs close)
BUY_DELTA = 0.002  # +0.20%
SELL_DELTA = -0.002  # -0.20%

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
    pct = f"{delta_pct*100:.2f}%"
    if delta_pct >= buy_thr:
        return f"Predicted ↑ {pct} vs close (>{buy_thr*100:.2f}%). Upside expected; BUY bias."
    if delta_pct <= sell_thr:
        return f"Predicted ↓ {pct} vs close (<{sell_thr*100:.2f}%). Downside risk; SELL bias."
    return f"Predicted {pct} vs close within band; momentum unclear; HOLD."


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

    # First pass: discover latest date per ticker
    latest_by_ticker: Dict[str, date_type] = {}
    parsed_files: List[Tuple[str, str]] = []

    for path in pred_files:
        df = load_pred_file(path)
        if df.empty:
            continue

        ticker = resolve_ticker(path, df)
        if not ticker:
            continue

        if "date" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

        if "date" not in df.columns:
            continue

        dser = normalize_date(df["date"]).dropna()
        if dser.empty:
            continue

        latest = pd.to_datetime(dser.max(), errors="coerce")
        if pd.isna(latest):
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
            continue

        if "date" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

        required = {"date", "close", "predicted_close"}
        if not required.issubset(df.columns):
            continue

        df = df.copy()
        df["date"] = normalize_date(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["predicted_close"] = pd.to_numeric(df["predicted_close"], errors="coerce")
        df = df.dropna(subset=["date", "close", "predicted_close"])
        if df.empty:
            continue
        df = df.sort_values("date")

        eff_dt = pd.Timestamp(effective_asof).normalize()
        df_upto = df[df["date"] <= eff_dt]
        if df_upto.empty:
            continue

        latest_row = df_upto.iloc[-1]
        row_date = pd.to_datetime(latest_row["date"], errors="coerce")
        if pd.isna(row_date):
            continue
        row_date = row_date.normalize()

        close = float(latest_row["close"])
        pred = float(latest_row["predicted_close"])
        delta_pct = (pred / close) - 1.0

        if delta_pct >= BUY_DELTA:
            sig = "BUY"
        elif delta_pct <= SELL_DELTA:
            sig = "SELL"
        else:
            sig = "HOLD"

        rows_out.append(
            {
                "date": row_date,
                "as_of_date": eff_dt,  # effective authority (deterministic outputs)
                "session_as_of_date": pd.Timestamp(
                    session_asof
                ),  # informational (what NYSE says is complete)
                "ticker": ticker,
                "close": close,
                "predicted_close": pred,
                "delta_pct": float(delta_pct),
                "signal": sig,
                "confidence": round(abs(delta_pct), 4),
                "rationale": build_rationale(delta_pct, BUY_DELTA, SELL_DELTA),
                "generated_at_utc": generated_at,
            }
        )

    if not rows_out:
        print("🚫 No signals generated (no valid prediction rows).")
        raise SystemExit(0)

    signals = pd.DataFrame(rows_out)
    signals["ticker"] = signals["ticker"].astype(str).map(_u)
    signals = signals[signals["ticker"].map(is_valid_ticker)].copy()

    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    signals["as_of_date"] = pd.to_datetime(signals["as_of_date"], errors="coerce")
    signals["session_as_of_date"] = pd.to_datetime(signals["session_as_of_date"], errors="coerce")
    signals = signals.dropna(subset=["date", "as_of_date"])
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
