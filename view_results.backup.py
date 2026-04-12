# view_results.py — TRITON Command Center (Phase 1.5)
# ---------------------------------------------------
# ✅ ZERO st.tabs
# ✅ Single router
# ✅ Real-data only (no fake metrics)
# ✅ Capital Preservation First
#
# CLEAN UPDATE (Dec 2025):
#   ✅ Prefer portfolio_history.csv + trade_log.csv as primary
#   ✅ Prefer signals_with_rationale.csv (fallback to signals.csv)
#   ✅ Pipeline health uses heartbeat.json (fallback pipeline_status.json)
#   ✅ Optional artifacts never force overall FAIL
#   ✅ Safe cached loaders (CSV/JSON) + column sanitization
#   ✅ Embedded Data Contract Validator (Phase 1.5)
#   ✅ FIX: No DeltaGenerator leakage (no "return st.*" anywhere)
#
# PATCH (this update):
#   ✅ Align Portfolio History contract to real schema: date, total_value (+ cash, market_value)
#   ✅ Fix latest_portfolio_status(): aggregates total_value by date across tickers for equity + drawdown
#   ✅ Upgrade Portfolio History page: KPIs + Plotly chart (fallback table) + raw tail
#   ✅ ADD: Manual Order Desk page (UI wrapper) ONLY if pages/manual_order_desk.py exists
#   ✅ ADD: Run CSV Orders page (UI wrapper) ONLY if ui/pages/run_csv_orders_page.py exists
#
# PHASE 2.3 (this update):
#   ✅ ADD: 🔴 Live Trading (Safe Gates + Broker Snapshot) — if dashboard/live_trading_panel.py exists
#   ✅ ADD: 🚦 Live Orders Panel (Read-only) — reads live_orders.csv, shows open orders/status/age/side/qty/price
#
# HOTFIX (live_orders.csv LOAD_FAILED / DeltaGenerator spam):
#   ✅ Contract validator treats *optional* LOAD_FAILED as WARN (not ERROR)
#   ✅ Validator loads CSVs in "silent mode" (no st.error inside validation path)
#   ✅ Cached CSV loader returns a clean, short error string (prevents giant repr spam)
#
# NEW HOTFIX (sidebar widgets appear once then disappear):
#   ✅ NO optional module imports at top-level (prevents import-time st.sidebar leakage)
#   ✅ Optional pages detected by FILE EXISTS (not callable(imported_fn))
#   ✅ Lazy import inside the page render only

from __future__ import annotations

import json
import importlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────
# PAGE CONFIG  (MUST BE FIRST Streamlit call)
# ──────────────────────────────
st.set_page_config(
    page_title="TRITON • Command Center (Phase 1.5)",
    page_icon="🧠",
    layout="wide",
)

# ──────────────────────────────
# COMPACT UI CSS (fix truncation on laptops)
# ──────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 3.2rem; }

    @keyframes tritonPulse {0%{transform:scale(1);opacity:1;}50%{transform:scale(1.004);opacity:.96;}100%{transform:scale(1);opacity:1;}}
    .triton-clock{position:sticky;top:0;z-index:999;border-radius:12px;border:1px solid rgba(148,163,184,.35);padding:.55rem .75rem;margin:.25rem 0 .85rem 0;backdrop-filter:blur(8px);background:rgba(15,23,42,.62);}
    .triton-clock .row{display:flex;gap:.65rem;align-items:center;justify-content:space-between;flex-wrap:wrap;}
    .triton-clock .left{font-weight:700;letter-spacing:.2px;}
    .triton-clock .mid{opacity:.92;font-size:.92rem;}
    .triton-clock .right{opacity:.85;font-size:.85rem;white-space:nowrap;}
    .triton-clock.fresh{border-color:rgba(34,197,94,.55);} .triton-clock.aging{border-color:rgba(234,179,8,.55);} .triton-clock.stale{border-color:rgba(239,68,68,.65);animation:tritonPulse 1.6s ease-in-out infinite;} .triton-clock.unknown{border-color:rgba(148,163,184,.45);}
    .triton-pill{display:inline-flex;align-items:center;gap:.35rem;padding:.2rem .5rem;border-radius:999px;font-size:.82rem;border:1px solid rgba(148,163,184,.35);background:rgba(2,6,23,.35);}
    .triton-pill.fresh{border-color:rgba(34,197,94,.55);} .triton-pill.aging{border-color:rgba(234,179,8,.55);} .triton-pill.stale{border-color:rgba(239,68,68,.65);} .triton-pill.unknown{border-color:rgba(148,163,184,.45);}

    /* Top status stack (Freshness Clock + Execution Gate) */
    .triton-topwrap{position:sticky;top:0;z-index:9999;}
    .triton-topwrap .triton-clock{margin:.25rem 0 .35rem 0;}
    .triton-execbar{border-radius:10px;border:1px solid rgba(148,163,184,.35);padding:.40rem .75rem;margin:0 0 .85rem 0;backdrop-filter:blur(8px);background:rgba(15,23,42,.62);}
    .triton-execbar .row{display:flex;gap:.65rem;align-items:center;justify-content:space-between;flex-wrap:wrap;}
    .triton-execbar .left{font-weight:700;letter-spacing:.2px;}
    .triton-execbar .right{opacity:.85;font-size:.85rem;white-space:nowrap;}
    .triton-execbar.enabled{border-color:rgba(34,197,94,.55);}
    .triton-execbar.locked{border-color:rgba(239,68,68,.65);}



    div[data-testid="stMetric"] {
        padding: 0.15rem 0.25rem !important;
        border-radius: 8px;
    }

    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        line-height: 1.0rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        margin-bottom: 0.10rem !important;
    }

    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
        line-height: 1.25rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    div[data-testid="stMetricDelta"] > div,
    div[data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
        line-height: 1.0rem !important;
        white-space: nowrap !important;
    }

    div[data-testid="stDataFrame"] * {
        font-size: 0.86rem !important;
    }

    div[data-testid="stDataFrame"] thead th,
    div[data-testid="stDataFrame"] tbody td {
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }

    button[data-testid="stExpanderToggleIcon"] + div {
        font-size: 0.92rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_VERSION = "r25-30_2026-01-08a"
TZ_ET = timezone(timedelta(hours=-4))  # display ET as UTC-4 (simple display TZ)

# ──────────────────────────────
# PROJECT ROOT / PATHS
# ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

RESULTS_DIR = DATA_ROOT / "results"
ORDERS_DIR = DATA_ROOT / "orders"
PRED_DIR = DATA_ROOT / "predictions"
STRESS_DIR = DATA_ROOT / "stress_test_results"

for p in (RESULTS_DIR, ORDERS_DIR, PRED_DIR, STRESS_DIR):
    p.mkdir(parents=True, exist_ok=True)

PORTFOLIO_HISTORY_PATH = RESULTS_DIR / "portfolio_history.csv"
POSITIONS_SNAPSHOT_PATH = RESULTS_DIR / "positions_snapshot.csv"
TRADE_LOG_PATH = RESULTS_DIR / "trade_log.csv"
SIGNALS_PATH = RESULTS_DIR / "signals.csv"
SIGNALS_RATIONALE_PATH = RESULTS_DIR / "signals_with_rationale.csv"

SIGNAL_LIFECYCLE_PATH = RESULTS_DIR / "signal_lifecycle.csv"
STOCK_SCORES_PATH = RESULTS_DIR / "stock_scores.csv"
TARGET_WEIGHTS_PATH = RESULTS_DIR / "target_weights.csv"

GUARD_SNAPSHOT_PATH = RESULTS_DIR / "guard_snapshot.json"

# Risk artifacts (prefer risk_report.json; fallback to adaptive_risk_state.json if you still write it)
RISK_REPORT_PATH = RESULTS_DIR / "risk_report.json"  # produced by services/generate_risk_report.py
RISK_STATE_PATH = RESULTS_DIR / "adaptive_risk_state.json"  # legacy/optional fallback

LIVE_ORDERS_PATH = RESULTS_DIR / "live_orders.csv"

LIVE_ARMED_PATH = RESULTS_DIR / "live_armed.json"
# Phase 1.5 Health / Heartbeat
HEARTBEAT_PATH = RESULTS_DIR / "heartbeat.json"  # preferred
PIPELINE_STATUS_PATH = RESULTS_DIR / "pipeline_status.json"  # fallback

# Nice-to-have (do not force FAIL)
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"

# ──────────────────────────────
# OPTIONAL PAGES — DETECT BY FILE EXISTS (NO IMPORTS HERE)
#   Prevents "sidebar appears once then disappears" due to import-time st.sidebar side-effects.
# ──────────────────────────────
LIVE_TRADING_PANEL_FILE = PROJECT_ROOT / "dashboard" / "live_trading_panel.py"
MANUAL_ORDER_DESK_FILE = PROJECT_ROOT / "pages" / "manual_order_desk.py"
RUN_CSV_ORDERS_FILE = PROJECT_ROOT / "ui" / "pages" / "run_csv_orders_page.py"

HAS_LIVE_TRADING_PANEL = LIVE_TRADING_PANEL_FILE.exists()
HAS_MANUAL_ORDER_DESK = MANUAL_ORDER_DESK_FILE.exists()
HAS_RUN_CSV_ORDERS = RUN_CSV_ORDERS_FILE.exists()

# ──────────────────────────────
# HELPERS
# ──────────────────────────────
STATUS_RANK = {"🟢": 0, "🟡": 1, "🔴": 2, "⚪": 3}  # ⚪ treated as unknown/worst


def _lazy_import(func_path: str):
    """
    Lazy import a callable only when needed.
    func_path example: "pages.manual_order_desk:render_manual_order_desk"
    """
    mod_name, func_name = func_path.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name, None)
    return fn if callable(fn) else None


def sanitize_df_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten, trim, and dedupe column names (prevents Arrow duplicate-name errors)."""
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x) for x in tup if x not in ("", None)) for tup in df.columns]

    seen: Dict[str, int] = {}
    cols: List[str] = []
    for raw in df.columns:
        c = str(raw).strip()
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)

    df.columns = cols
    return df


def _short_err(e: Exception, limit: int = 280) -> str:
    """Return a compact error string to avoid gigantic repr spam."""
    try:
        msg = str(e)
    except Exception:
        msg = repr(e)
    msg = " ".join((msg or "").split())
    if len(msg) > limit:
        msg = msg[:limit].rstrip() + "…"
    return msg or repr(e)


@st.cache_data(show_spinner=False)
def _read_csv_bytesafe(path_str: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Cached CSV loader with safe error return (no Streamlit side effects inside cache)."""
    path = Path(path_str)
    if (not path.exists()) or path.stat().st_size == 0:
        return None, None
    try:
        df = pd.read_csv(path)
        df = sanitize_df_cols(df)
        return df, None
    except Exception as e:
        return None, _short_err(e)


def load_csv(path: Path, *, show_error: bool = True) -> Optional[pd.DataFrame]:
    """
    CSV loader.
      - show_error=True: emits st.error on load failure (good for pages)
      - show_error=False: silent failure (required for validator to avoid DeltaGenerator noise)
    """
    df, err = _read_csv_bytesafe(str(path))
    if err and show_error:
        st.error(f"❌ Failed loading {path.name}: {err}")
    return df


def load_first_nonempty_csv(paths: List[Path]) -> Optional[pd.DataFrame]:
    """Try CSVs in order; return first that loads and has at least 1 row."""
    for p in paths:
        df = load_csv(p, show_error=True)
        if df is not None and not df.empty:
            return df
    return None


@st.cache_data(show_spinner=False)
def _read_json_bytesafe(path_str: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Cached JSON loader with safe error return (no Streamlit side effects inside cache)."""
    path = Path(path_str)
    if (not path.exists()) or path.stat().st_size == 0:
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, _short_err(e)


def load_json(path: Path, *, show_error: bool = True) -> Optional[Dict[str, Any]]:
    obj, err = _read_json_bytesafe(str(path))
    if err and show_error:
        st.error(f"❌ Failed loading {path.name}: {err}")
    return obj


def now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(TZ_ET)


def now_et_str() -> str:
    return now_et().strftime("%Y-%m-%d %H:%M ET")


def dt_to_et_str(dt: Optional[datetime]) -> str:
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_ET).strftime("%Y-%m-%d %H:%M ET")


def file_mtime_dt(path: Path) -> Optional[datetime]:
    """Best-effort file modified time (tz-aware UTC)."""
    try:
        p = Path(path)
        if (not p.exists()) or p.stat().st_size == 0:
            return None
        return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def parse_any_datetime(x: Any) -> Optional[datetime]:
    """Parse timestamps from heartbeat/pipeline_status (ISO, epoch, datetime). Returns tz-aware UTC dt."""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            val = float(x)
            if val > 10_000_000_000:  # ms
                return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except Exception:
            return None
    try:
        s = str(x).strip()
        if not s:
            return None
        dtp = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dtp):
            return None
        return dtp.to_pydatetime()
    except Exception:
        return None


def ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df['date'] exists as datetime (naive)."""
    df = df.copy()
    candidates = [
        "date",
        "timestamp",
        "time",
        "datetime",
        "as_of",
        "submitted_at",
        "created_at",
        "updated_at",
        "filled_at",
        "canceled_at",
    ]
    chosen = next((c for c in candidates if c in df.columns), None)

    if chosen is None:
        df["date"] = pd.NaT
        return df

    s = pd.to_datetime(df[chosen], errors="coerce", utc=False)

    # If tz-aware, convert to naive
    try:
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            s = s.dt.tz_convert(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass

    df["date"] = s
    return df


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def aggregate_portfolio_history(ph: pd.DataFrame) -> Tuple[pd.DataFrame, bool, str]:
    """Return (daily_df, per_ticker, method)."""

    if ph is None or ph.empty:
        return ph, False, "empty"

    df = ph.copy()
    if "date" not in df.columns:
        return ph, False, "missing_date"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    for c in ("cash", "market_value", "total_value"):
        if c in df.columns:
            df[c] = safe_numeric(df, c)

    # Detect per-ticker layout (multiple rows per normalized date)
    per_ticker = ("ticker" in df.columns) and (
        df.groupby(df["date"].dt.normalize()).size().max() > 1
    )

    if not per_ticker:
        cols = [c for c in ("date", "cash", "market_value", "total_value") if c in df.columns]
        out = df[cols].copy()
        out["date"] = out["date"].dt.normalize()
        out = out.sort_values("date").reset_index(drop=True)
        return out, False, "daily_rows"

    # Per-ticker layout: aggregate to daily.
    df["date"] = df["date"].dt.normalize()

    method = "per_ticker"
    constant_like = False
    if "total_value" in df.columns:
        stats = df.groupby("date")["total_value"].agg(["mean", "std"]).reset_index()
        rel_std = (stats["std"] / stats["mean"].replace(0, np.nan)).fillna(0)
        constant_like = bool((rel_std < 1e-9).all())

    cash_series = None
    if "cash" in df.columns:
        cash_series = df.groupby("date")["cash"].max()

    # If total_value repeated on each ticker row (constant-like), take max.
    if constant_like and "total_value" in df.columns:
        daily_total = df.groupby("date")["total_value"].max()
        daily = pd.DataFrame({"date": daily_total.index, "total_value": daily_total.values})
        if cash_series is not None:
            daily["cash"] = cash_series.reindex(daily["date"]).values
        if "market_value" in df.columns and cash_series is not None:
            daily["market_value"] = (daily["total_value"] - daily["cash"]).astype(float)
        method = "per_ticker_total_is_constant"
        return daily.sort_values("date").reset_index(drop=True), True, method

    # Otherwise: sum market_value; add cash once.
    if "market_value" in df.columns:
        mv_series = df.groupby("date")["market_value"].sum()
        daily = pd.DataFrame({"date": mv_series.index, "market_value": mv_series.values})
        if cash_series is not None:
            daily["cash"] = cash_series.reindex(daily["date"]).values
        if "cash" in daily.columns:
            daily["total_value"] = daily["cash"].astype(float) + daily["market_value"].astype(float)
        method = "per_ticker_sum_market_value"
        return daily.sort_values("date").reset_index(drop=True), True, method

    # Last resort: max total_value
    if "total_value" in df.columns:
        daily_total = df.groupby("date")["total_value"].max()
        daily = pd.DataFrame({"date": daily_total.index, "total_value": daily_total.values})
        if cash_series is not None:
            daily["cash"] = cash_series.reindex(daily["date"]).values
        method = "per_ticker_max_total_value"
        return daily.sort_values("date").reset_index(drop=True), True, method

    return df, True, "per_ticker_unhandled"


def market_status_simple() -> Tuple[str, str]:
    """Simple US market status (no holiday calendar): Mon–Fri 09:30–16:00 ET."""
    et = now_et()
    weekday = et.weekday()
    open_today = et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_today = et.replace(hour=16, minute=0, second=0, microsecond=0)

    if weekday < 5 and open_today <= et < close_today:
        return "🟢 Market OPEN", f"Closes at {close_today.strftime('%H:%M ET')}"
    return "🔴 Market CLOSED", "Opens 09:30 ET (next trading day)"


def kpi(val: Any, kind: str = "text") -> str:
    if val is None:
        return "—"
    if kind == "pct":
        try:
            v = float(val)
        except Exception:
            return "—"
        if not np.isfinite(v):
            return "—"
        return f"{v * 100:.2f}%"
    if kind == "usd":
        try:
            v = float(val)
        except Exception:
            return "—"
        if not np.isfinite(v):
            return "—"
        return f"${v:,.2f}"
    s = str(val).strip()
    return s if s else "—"


# ──────────────────────────────
# DISPLAY-ONLY NUMBER FORMATTING
#   - Never use formatted strings for math/execution.
#   - Apply only right before st.dataframe/st.table.
# ──────────────────────────────


def _fmt_int_commas(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "" if v is None else str(v)
    if not np.isfinite(x):
        return ""
    return f"{x:,.0f}"


def _fmt_usd(v: Any, decimals: int = 0) -> str:
    try:
        x = float(v)
    except Exception:
        return "" if v is None else str(v)
    if not np.isfinite(x):
        return ""
    return f"${x:,.{decimals}f}"


def _fmt_pct(v: Any, decimals: int = 2) -> str:
    try:
        x = float(v)
    except Exception:
        return "" if v is None else str(v)
    if not np.isfinite(x):
        return ""
    # Handle both 0.12 (12%) and 12 (12%) gracefully
    if abs(x) > 1.5:
        return f"{x:.{decimals}f}%"
    return f"{x * 100:.{decimals}f}%"


def format_df_for_display(
    df: pd.DataFrame,
    money_cols: Optional[List[str]] = None,
    pct_cols: Optional[List[str]] = None,
    int_cols: Optional[List[str]] = None,
    usd_decimals: int = 0,
) -> pd.DataFrame:
    """Return a COPY of df with select columns formatted as strings for readability."""
    out = df.copy()
    money_cols = money_cols or []
    pct_cols = pct_cols or []
    int_cols = int_cols or []

    for c in money_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda v: _fmt_usd(v, usd_decimals))

    for c in pct_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(_fmt_pct)

    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(_fmt_int_commas)

    return out


def _stance_cell_css(v: Any) -> str:
    """CSS for stance/decision cells (display only)."""
    s = str(v).strip().upper()
    # Defaults
    bg = "rgba(148,163,184,.18)"  # slate
    fg = "#e2e8f0"
    bd = "rgba(148,163,184,.35)"

    if s in ("BUY",):
        bg, fg, bd = "rgba(34,197,94,.28)", "#dcfce7", "rgba(34,197,94,.55)"
    elif s in ("ADD",):
        bg, fg, bd = "rgba(59,130,246,.28)", "#dbeafe", "rgba(59,130,246,.55)"
    elif s in ("HOLD", "WAIT"):
        bg, fg, bd = "rgba(148,163,184,.12)", "#cbd5e1", "rgba(148,163,184,.25)"
    elif s in ("TRIM", "REDUCE"):
        bg, fg, bd = "rgba(245,158,11,.28)", "#ffedd5", "rgba(245,158,11,.55)"
    elif s in ("EXIT", "SELL"):
        bg, fg, bd = "rgba(239,68,68,.30)", "#fee2e2", "rgba(239,68,68,.60)"

    return (
        f"background-color:{bg};"
        f"color:{fg};"
        f"border:1px solid {bd};"
        "font-weight:800;"
        "text-align:center;"
        "border-radius:999px;"
        "padding:2px 10px;"
        "letter-spacing:.2px;"
    )


def _dim_hold_rows(row: pd.Series) -> List[str]:
    """Row-wise styling: de-emphasize HOLD rows (display only)."""
    try:
        s = str(row.get("stance", "")).strip().upper()
    except Exception:
        s = ""

    if s in ("HOLD", "WAIT"):
        # Lighten text a bit across the whole row
        return ["color: rgba(203,213,225,.75);"] * len(row)
    return [""] * len(row)


def latest_portfolio_status() -> Dict[str, Any]:
    """
    Real-data-only snapshot:
      - portfolio_history.csv: latest equity + drawdown vs peak (aggregated across tickers per date)
      - guard_snapshot.json: optional overrides if present (mode/reason/buying_power/reserve_pct)
    """
    out: Dict[str, Any] = {
        "mode": "UNKNOWN",
        "reason": "",
        "latest_equity": np.nan,
        "drawdown_pct": np.nan,
        "buying_power": np.nan,
        "reserve_pct": np.nan,
        "updated": now_et_str(),
    }

    ph = load_csv(PORTFOLIO_HISTORY_PATH, show_error=False)
    if ph is not None and not ph.empty:
        ph = sanitize_df_cols(ph)
        ph = ensure_date_col(ph)

        if "total_value" not in ph.columns:
            for alt in ("equity", "portfolio_value", "portfolio_total", "value", "total"):
                if alt in ph.columns:
                    ph = ph.rename(columns={alt: "total_value"})
                    break

        daily, _per_ticker, _method = aggregate_portfolio_history(ph)
        if daily is not None and not daily.empty and "total_value" in daily.columns:
            tv = pd.to_numeric(daily["total_value"], errors="coerce")
            if tv.notna().any():
                latest_equity = float(tv.iloc[-1])
                peak_equity = float(tv.max())
                out["latest_equity"] = latest_equity
                if np.isfinite(latest_equity) and np.isfinite(peak_equity) and peak_equity > 0:
                    out["drawdown_pct"] = float((latest_equity / peak_equity) - 1.0)

    guard = load_json(GUARD_SNAPSHOT_PATH, show_error=False)
    if guard:
        mode = str(guard.get("mode", out["mode"])).strip()
        out["mode"] = mode.upper() if mode else out["mode"]
        out["reason"] = str(guard.get("reason", out["reason"]) or "")

        if "buying_power" in guard:
            try:
                out["buying_power"] = float(guard.get("buying_power"))
            except Exception:
                pass
        if "reserve_pct" in guard:
            try:
                out["reserve_pct"] = float(guard.get("reserve_pct"))
            except Exception:
                pass

        ts = guard.get("timestamp") or guard.get("updated_at") or guard.get("time")
        if ts is not None:
            out["updated"] = str(ts)

    return out


def _soften_status(status: str, max_bad: str) -> str:
    """Cap worstness. Example: cap to 🟡 means 🔴/⚪ become 🟡."""
    s_rank = STATUS_RANK.get(status, 0)
    cap_rank = STATUS_RANK.get(max_bad, 0)
    return status if s_rank <= cap_rank else max_bad


# ──────────────────────────────
# PHASE 1.5 — DATA CONTRACT VALIDATOR (embedded)
# ──────────────────────────────
@dataclass
class ContractIssue:
    level: str  # "ERROR" | "WARN" | "INFO"
    code: str
    message: str
    hint: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractResult:
    name: str
    path: str
    ok: bool
    row_count: int = 0
    col_count: int = 0
    issues: List[ContractIssue] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, hint: str = "", **context: Any) -> None:
        safe_ctx: Dict[str, Any] = {}
        for k, v in context.items():
            try:
                json.dumps(v)
                safe_ctx[k] = v
            except Exception:
                safe_ctx[k] = str(v)
        self.issues.append(
            ContractIssue(level=level, code=code, message=message, hint=hint, context=safe_ctx)
        )
        if level == "ERROR":
            self.ok = False


@dataclass
class DataContract:
    name: str
    path: Path
    fmt: str  # "csv" | "json"
    required_cols: List[str] = field(default_factory=list)
    optional_cols: List[str] = field(default_factory=list)

    min_rows: int = 1
    allow_empty: bool = False  # used as "optional artifact" flag in this project

    enforce_date_col: Optional[str] = None
    enforce_ticker_col: Optional[str] = "ticker"
    ticker_regex: str = r"^[A-Z0-9\.\-\_]{1,15}$"

    max_null_frac_by_col: Dict[str, float] = field(default_factory=dict)
    unique_keys: List[str] = field(default_factory=list)

    coerce: Dict[str, str] = field(default_factory=dict)
    numeric_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)

    custom_checks: List[Callable[[pd.DataFrame], List[ContractIssue]]] = field(default_factory=list)


def _contract_load(
    contract: DataContract,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """Returns (df, obj, err) based on contract fmt."""
    p = contract.path
    if not p.exists():
        return None, None, "MISSING_FILE"
    if p.stat().st_size == 0:
        return None, None, "EMPTY_FILE"

    if contract.fmt == "csv":
        df = load_csv(p, show_error=False)  # silent load (NO st.error during validation)
        if df is None:
            return None, None, "LOAD_FAILED"
        return df, None, None

    if contract.fmt == "json":
        obj = load_json(p, show_error=False)
        if obj is None:
            return None, None, "LOAD_FAILED"
        return None, obj, None

    return None, None, "UNSUPPORTED_FMT"


def validate_contract(contract: DataContract) -> ContractResult:
    res = ContractResult(name=contract.name, path=str(contract.path.resolve()), ok=True)
    df, obj, err = _contract_load(contract)

    # Optional artifacts never force FAIL:
    if err in ("MISSING_FILE", "EMPTY_FILE", "LOAD_FAILED") and contract.allow_empty:
        msg = {
            "MISSING_FILE": f"Optional file missing: {contract.path}",
            "EMPTY_FILE": f"Optional file exists but has 0 bytes: {contract.path}",
            "LOAD_FAILED": f"Optional file failed to load: {contract.path}",
        }.get(err, f"Optional file issue: {contract.path}")
        res.add(
            "WARN", err, msg, hint="Optional artifact — OK to ignore unless you need that page."
        )
        return res

    if err == "MISSING_FILE":
        res.ok = False
        res.add(
            "ERROR",
            "MISSING_FILE",
            f"Missing required file: {contract.path}",
            hint="Run the pipeline step that generates this artifact.",
        )
        return res

    if err == "EMPTY_FILE":
        res.ok = False
        res.add(
            "ERROR",
            "EMPTY_FILE",
            f"File is empty (0 bytes): {contract.path}",
            hint="Upstream pipeline may have failed or produced no output.",
        )
        return res

    if err == "LOAD_FAILED":
        res.ok = False
        res.add(
            "ERROR",
            "LOAD_FAILED",
            f"Failed to load file: {contract.path}",
            hint="Check file integrity and schema.",
        )
        return res

    if err == "UNSUPPORTED_FMT":
        res.ok = False
        res.add("ERROR", "UNSUPPORTED_FMT", f"Unsupported contract fmt={contract.fmt}")
        return res

    if contract.fmt == "json":
        if not isinstance(obj, dict):
            res.ok = False
            res.add(
                "ERROR",
                "JSON_NOT_OBJECT",
                "JSON root is not an object/dict.",
                hint="Ensure the snapshot writer outputs a JSON object.",
            )
        else:
            res.add("INFO", "JSON_OK", "JSON snapshot loaded.")
        return res

    assert df is not None
    res.row_count = int(len(df))
    res.col_count = int(len(df.columns))

    if len(df) == 0:
        if contract.allow_empty:
            res.add(
                "INFO",
                "EMPTY_ROWS_OPTIONAL",
                "CSV contains 0 rows (optional artifact).",
                hint="Optional artifact — OK to ignore unless you need that page.",
            )
            return res
        res.ok = False
        res.add(
            "ERROR",
            "EMPTY_ROWS",
            "CSV contains 0 rows.",
            hint="Upstream pipeline may have produced an empty dataset.",
        )
        return res

    if contract.min_rows and len(df) < contract.min_rows:
        res.add(
            "WARN",
            "LOW_ROW_COUNT",
            f"Row count below expected minimum ({len(df)} < {contract.min_rows}).",
            min_rows=contract.min_rows,
            row_count=len(df),
        )

    cols = set(map(str, df.columns))
    missing = [c for c in contract.required_cols if c not in cols]
    if missing:
        res.ok = False
        res.add(
            "ERROR",
            "MISSING_COLUMNS",
            f"Missing required columns: {missing}",
            hint="Fix upstream generator or update contract intentionally.",
            missing=missing,
        )

    # Coerce (best-effort)
    for col, dtype in contract.coerce.items():
        if col not in df.columns:
            continue
        try:
            if dtype.startswith("datetime"):
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
            elif dtype.startswith("float"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dtype.startswith("int"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype in ("string", "str"):
                df[col] = df[col].astype("string")
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            res.add(
                "WARN",
                "COERCE_FAILED",
                f"Could not coerce '{col}' to {dtype}.",
                col=col,
                dtype=dtype,
                error=_short_err(e),
            )

    for col, max_frac in contract.max_null_frac_by_col.items():
        if col not in df.columns:
            continue
        frac = float(df[col].isna().mean()) if len(df) else 0.0
        if frac > max_frac:
            res.add(
                "WARN",
                "NULL_FRACTION_HIGH",
                f"'{col}' null fraction {frac:.2%} > {max_frac:.2%}",
                col=col,
                null_frac=frac,
                max_null_frac=max_frac,
            )

    if contract.unique_keys and all(k in df.columns for k in contract.unique_keys):
        dupes = df.duplicated(subset=contract.unique_keys, keep=False)
        dupe_count = int(dupes.sum())
        if dupe_count:
            res.add(
                "WARN",
                "DUPLICATE_KEYS",
                f"{dupe_count} rows duplicate keys {contract.unique_keys}",
                keys=contract.unique_keys,
                dupe_rows=dupe_count,
            )

    if contract.enforce_date_col and contract.enforce_date_col in df.columns:
        dtc = pd.to_datetime(df[contract.enforce_date_col], errors="coerce", utc=False)
        bad = int(dtc.isna().sum())
        if bad:
            res.add(
                "WARN",
                "BAD_DATES",
                f"{bad} unparsable dates in '{contract.enforce_date_col}'.",
                bad_dates=bad,
            )
        if dtc.notna().any():
            res.add(
                "INFO",
                "DATE_RANGE",
                f"{dtc.min()} → {dtc.max()}",
                min_date=str(dtc.min()),
                max_date=str(dtc.max()),
            )

    if contract.enforce_ticker_col and contract.enforce_ticker_col in df.columns:
        s = df[contract.enforce_ticker_col].astype("string")
        try:
            patt = re.compile(contract.ticker_regex)
            bad = ~s.fillna("").apply(lambda x: bool(patt.match(str(x))))
        except Exception:
            bad = ~s.fillna("").str.match(contract.ticker_regex)
        bad_count = int(bad.sum())
        if bad_count:
            res.add(
                "WARN",
                "BAD_TICKERS",
                f"{bad_count} tickers fail regex {contract.ticker_regex}",
                bad_count=bad_count,
            )

    for col, (mn, mx) in contract.numeric_bounds.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if mn is not None:
            below = int((x < mn).sum(skipna=True))
            if below:
                res.add(
                    "WARN",
                    "NUMERIC_BELOW_MIN",
                    f"{below} values in '{col}' below {mn}",
                    col=col,
                    min_bound=mn,
                    below=below,
                )
        if mx is not None:
            above = int((x > mx).sum(skipna=True))
            if above:
                res.add(
                    "WARN",
                    "NUMERIC_ABOVE_MAX",
                    f"{above} values in '{col}' above {mx}",
                    col=col,
                    max_bound=mx,
                    above=above,
                )

    for fn in contract.custom_checks:
        try:
            issues = fn(df)
            for iss in issues:
                res.issues.append(iss)
                if iss.level == "ERROR":
                    res.ok = False
        except Exception as e:
            res.add(
                "WARN",
                "CUSTOM_CHECK_FAILED",
                "A custom contract check crashed.",
                error=_short_err(e),
                function=getattr(fn, "__name__", "unknown"),
            )

    return res


def validate_all_contracts() -> List[ContractResult]:
    def _signals_check(df: pd.DataFrame) -> List[ContractIssue]:
        issues: List[ContractIssue] = []
        if "signal" in df.columns:
            s = pd.to_numeric(df["signal"], errors="coerce")
            bad = (~s.isna()) & (~s.isin([-1, 0, 1]))
            bad_n = int(bad.sum())
            if bad_n:
                issues.append(
                    ContractIssue(
                        level="WARN",
                        code="SIGNAL_VALUES_ODD",
                        message=f"'signal' has {bad_n} numeric values outside [-1,0,1].",
                        hint="If you changed the signal scale, update this contract rule.",
                        context={"bad_n": bad_n},
                    )
                )
        return issues

    contracts: List[DataContract] = [
        DataContract(
            name="Portfolio History",
            path=PORTFOLIO_HISTORY_PATH,
            fmt="csv",
            required_cols=["date", "total_value"],
            optional_cols=["cash", "market_value", "ticker"],
            min_rows=2,
            enforce_date_col="date",
            enforce_ticker_col=None,
            unique_keys=[],
            coerce={
                "date": "datetime64[ns]",
                "cash": "float64",
                "market_value": "float64",
                "total_value": "float64",
            },
            numeric_bounds={
                "cash": (0.0, None),
                "market_value": (0.0, None),
                "total_value": (0.0, None),
            },
            max_null_frac_by_col={"date": 0.0, "total_value": 0.0},
        ),
        DataContract(
            name="Positions Snapshot",
            path=RESULTS_DIR / "positions_snapshot.csv",
            fmt="csv",
            required_cols=["snapshot_ts", "symbol", "qty"],
            optional_cols=[
                "market_value",
                "avg_entry_price",
                "current_price",
                "unrealized_pl",
                "side",
                "asset_id",
                "exchange",
            ],
            min_rows=1,
            allow_empty=False,
            coerce={"snapshot_ts": "datetime64[ns]", "qty": "float64"},
            max_null_frac_by_col={"snapshot_ts": 0.0, "symbol": 0.0, "qty": 0.0},
        ),
        DataContract(
            name="Trade Log",
            path=TRADE_LOG_PATH,
            fmt="csv",
            required_cols=["date", "ticker"],
            optional_cols=[
                "signal",
                "side",
                "action",
                "entry_price",
                "exit_price",
                "pnl",
                "profit",
                "pnl_pct",
                "quantity",
                "reason",
                "sl",
                "tp",
                "stop_loss",
                "take_profit",
                "exit_reason",
                "holding_days",
            ],
            min_rows=1,
            enforce_date_col="date",
            coerce={
                "date": "datetime64[ns]",
                "entry_price": "float64",
                "exit_price": "float64",
                "pnl": "float64",
                "profit": "float64",
                "pnl_pct": "float64",
                "quantity": "int64",
                "sl": "float64",
                "tp": "float64",
                "stop_loss": "float64",
                "take_profit": "float64",
                "holding_days": "float64",
            },
            max_null_frac_by_col={"date": 0.0, "ticker": 0.0},
        ),
        DataContract(
            name="Signals (Rationale)",
            path=SIGNALS_RATIONALE_PATH,
            fmt="csv",
            required_cols=["date", "ticker"],
            optional_cols=["signal", "pred", "confidence", "rationale", "regime", "capital_mode"],
            min_rows=1,
            enforce_date_col="date",
            unique_keys=["date", "ticker"],
            coerce={"date": "datetime64[ns]", "pred": "float64", "confidence": "float64"},
            max_null_frac_by_col={"date": 0.0, "ticker": 0.0},
            custom_checks=[_signals_check],
        ),
        DataContract(
            name="Signals (Fallback)",
            path=SIGNALS_PATH,
            fmt="csv",
            required_cols=["date", "ticker"],
            optional_cols=["signal", "pred", "confidence"],
            min_rows=1,
            enforce_date_col="date",
            unique_keys=["date", "ticker"],
            coerce={"date": "datetime64[ns]", "pred": "float64", "confidence": "float64"},
            max_null_frac_by_col={"date": 0.0, "ticker": 0.0},
            custom_checks=[_signals_check],
        ),
        DataContract(
            name="Signal Lifecycle",
            path=SIGNAL_LIFECYCLE_PATH,
            fmt="csv",
            required_cols=["ticker", "stance"],
            optional_cols=[
                "position_state",
                "lifecycle_action",
                "last_action",
                "state_changed",
                "as_of_date",
                "signal",
                "confidence",
                "edge_pct",
                "delta_pct",
                "freshness",
                "freshness_age_min",
                "freshness_source",
                "updated_utc",
                "state_source_file",
            ],
            min_rows=1,
            unique_keys=["ticker"],
            coerce={
                "confidence": "float64",
                "edge_pct": "float64",
                "delta_pct": "float64",
                "freshness_age_min": "float64",
            },
            max_null_frac_by_col={"ticker": 0.0, "stance": 0.0},
            allow_empty=True,
        ),
        DataContract(
            name="Stock Scores",
            path=STOCK_SCORES_PATH,
            fmt="csv",
            required_cols=["ticker"],
            optional_cols=[
                "score_total",
                "score_value",
                "score_growth",
                "score_momentum",
                "score_size",
                "total_score",
                "score",
            ],
            min_rows=1,
            unique_keys=["ticker"],
            coerce={
                "score_total": "float64",
                "score_value": "float64",
                "score_growth": "float64",
                "score_momentum": "float64",
                "score_size": "float64",
                "total_score": "float64",
                "score": "float64",
            },
            max_null_frac_by_col={"ticker": 0.0},
        ),
        DataContract(
            name="Target Weights",
            path=TARGET_WEIGHTS_PATH,
            fmt="csv",
            required_cols=["ticker"],
            optional_cols=[
                "weight",
                "target_weight",
                "allocation",
                "pct",
                "reason",
                "regime",
                "as_of",
                "date",
                "capital_mode",
            ],
            min_rows=1,
            unique_keys=["ticker"],
            coerce={
                "weight": "float64",
                "target_weight": "float64",
                "allocation": "float64",
                "pct": "float64",
            },
            max_null_frac_by_col={"ticker": 0.0},
            allow_empty=True,
        ),
        DataContract(name="Guard Snapshot", path=GUARD_SNAPSHOT_PATH, fmt="json", allow_empty=True),
        DataContract(
            name="Risk Report (preferred)", path=RISK_REPORT_PATH, fmt="json", allow_empty=True
        ),
        DataContract(
            name="Adaptive Risk State (fallback)",
            path=RISK_STATE_PATH,
            fmt="json",
            allow_empty=True,
        ),
        DataContract(name="Live Orders", path=LIVE_ORDERS_PATH, fmt="csv", allow_empty=True),
        DataContract(
            name="Model Comparison",
            path=MODEL_COMPARISON_PATH,
            fmt="csv",
            required_cols=["ticker", "date", "model"],
            optional_cols=["close", "predicted_close", "pred", "yhat"],
            min_rows=1,
            allow_empty=True,
            enforce_date_col="date",
        ),
        DataContract(
            name="Feature Importance",
            path=FEATURE_IMPORTANCE_PATH,
            fmt="csv",
            required_cols=["ticker", "model", "feature", "importance"],
            min_rows=1,
            allow_empty=True,
        ),
    ]
    return [validate_contract(c) for c in contracts]


def contracts_summary(results: List[ContractResult]) -> Dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if r.ok)
    errs = sum(1 for r in results for i in r.issues if i.level == "ERROR")
    warns = sum(1 for r in results for i in r.issues if i.level == "WARN")
    infos = sum(1 for r in results for i in r.issues if i.level == "INFO")
    return {
        "total": total,
        "ok": ok,
        "failed": total - ok,
        "error_count": errs,
        "warn_count": warns,
        "info_count": infos,
    }


def contracts_badge(summary: Dict[str, Any]) -> Tuple[str, str]:
    if summary.get("failed", 0) > 0 or summary.get("error_count", 0) > 0:
        return (
            "🔴 Data Contracts FAIL",
            f"{summary.get('failed', 0)} failed · {summary.get('error_count', 0)} errors",
        )
    if summary.get("warn_count", 0) > 0:
        return "🟡 Data Contracts WARN", f"{summary.get('warn_count', 0)} warnings"
    return "🟢 Data Contracts OK", "All required artifacts healthy"


def run_contracts_if_needed(force: bool = False) -> None:
    if force or ("contract_results" not in st.session_state):
        results = validate_all_contracts()
        st.session_state["contract_results"] = results
        st.session_state["contract_summary"] = contracts_summary(results)

    cs = st.session_state.get("contract_summary", {})
    failing = (cs.get("failed", 0) > 0) or (cs.get("error_count", 0) > 0)
    st.session_state["contracts_ok"] = not failing
    return None


# ──────────────────────────────
# PHASE 1.5 — PIPELINE HEALTH / HEARTBEAT
# ──────────────────────────────
def load_heartbeat() -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    hb = load_json(HEARTBEAT_PATH, show_error=False)
    if hb:
        return hb, HEARTBEAT_PATH
    hb2 = load_json(PIPELINE_STATUS_PATH, show_error=False)
    if hb2:
        return hb2, PIPELINE_STATUS_PATH
    return None, None


def file_meta(path: Path) -> Dict[str, Any]:
    out = {
        "path": str(path),
        "exists": False,
        "size_bytes": 0,
        "mtime_utc": None,
        "age_seconds": np.nan,
    }
    try:
        if not path.exists():
            return out
        stt = path.stat()
        out["exists"] = True
        out["size_bytes"] = int(stt.st_size)
        mtime = datetime.fromtimestamp(stt.st_mtime, tz=timezone.utc)
        out["mtime_utc"] = mtime
        out["age_seconds"] = float((datetime.now(timezone.utc) - mtime).total_seconds())
        return out
    except Exception:
        return out


def fmt_age(seconds: Any) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    s = float(seconds)
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s // 60)}m"
    if s < 86400:
        return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"
    return f"{int(s // 86400)}d {int((s % 86400) // 3600)}h"


def health_color_for_age(age_seconds: float, warn_s: float, fail_s: float) -> str:
    if not np.isfinite(age_seconds):
        return "⚪"
    if age_seconds <= warn_s:
        return "🟢"
    if age_seconds <= fail_s:
        return "🟡"
    return "🔴"


def _status_bucket(status: str) -> str:
    if status == "🔴":
        return "fail"
    if status == "🟡":
        return "warn"
    if status == "🟢":
        return "ok"
    return "unknown"


def compute_pipeline_health() -> Dict[str, Any]:
    warn_minutes = 90
    fail_minutes = 360

    hb, hb_path = load_heartbeat()
    hb_ts = None
    hb_age = np.nan
    hb_status = "⚪"
    hb_stage = ""
    hb_msg = ""
    hb_error = ""

    if isinstance(hb, dict):
        ts_raw = hb.get("timestamp") or hb.get("ts") or hb.get("updated_at") or hb.get("time")
        hb_ts = parse_any_datetime(ts_raw)
        if hb_ts:
            hb_age = float((datetime.now(timezone.utc) - hb_ts).total_seconds())
            hb_status = health_color_for_age(hb_age, warn_minutes * 60, fail_minutes * 60)

        hb_stage = str(hb.get("stage") or hb.get("step") or hb.get("pipeline_stage") or "")
        hb_msg = str(hb.get("message") or hb.get("notes") or "")
        hb_error = str(hb.get("error") or hb.get("last_error") or "")

        raw_status = str(hb.get("status") or "").lower().strip()
        if raw_status in ("fail", "failed", "error", "crash"):
            hb_status = "🔴"
        elif raw_status in ("warn", "warning", "degraded"):
            hb_status = "🟡"
        elif raw_status in ("ok", "success", "healthy"):
            if hb_status == "⚪":
                hb_status = "🟢"

    rationale_exists = SIGNALS_RATIONALE_PATH.exists() and (
        SIGNALS_RATIONALE_PATH.stat().st_size > 0
    )

    tracked: List[Tuple[str, Path, float, float, bool]] = [
        (
            "heartbeat.json / pipeline_status.json",
            (hb_path or HEARTBEAT_PATH),
            warn_minutes * 60,
            fail_minutes * 60,
            True,
        ),
        (
            "portfolio_history.csv",
            PORTFOLIO_HISTORY_PATH,
            warn_minutes * 60,
            fail_minutes * 60,
            True,
        ),
        ("trade_log.csv", TRADE_LOG_PATH, warn_minutes * 60, fail_minutes * 60, True),
        (
            "signals_with_rationale.csv",
            SIGNALS_RATIONALE_PATH,
            warn_minutes * 60,
            fail_minutes * 60,
            True,
        ),
        ("signals.csv (fallback)", SIGNALS_PATH, warn_minutes * 60, fail_minutes * 60, False),
        ("stock_scores.csv", STOCK_SCORES_PATH, warn_minutes * 60, fail_minutes * 60, False),
        ("target_weights.csv", TARGET_WEIGHTS_PATH, warn_minutes * 60, fail_minutes * 60, False),
        (
            "model_comparison.csv",
            MODEL_COMPARISON_PATH,
            warn_minutes * 60,
            fail_minutes * 60,
            False,
        ),
        (
            "feature_importance.csv",
            FEATURE_IMPORTANCE_PATH,
            warn_minutes * 60,
            fail_minutes * 60,
            False,
        ),
        ("risk_report.json", RISK_REPORT_PATH, warn_minutes * 60, fail_minutes * 60, False),
        (
            "adaptive_risk_state.json (fallback)",
            RISK_STATE_PATH,
            warn_minutes * 60,
            fail_minutes * 60,
            False,
        ),
        ("live_orders.csv", LIVE_ORDERS_PATH, warn_minutes * 60, fail_minutes * 60, False),
        ("guard_snapshot.json", GUARD_SNAPSHOT_PATH, warn_minutes * 60, fail_minutes * 60, False),
    ]

    rows: List[Dict[str, Any]] = []
    worst_any = "🟢"
    worst_any_score = 0
    worst_critical = "🟢"
    worst_critical_score = 0

    counts_all = {"ok": 0, "warn": 0, "fail": 0, "unknown": 0}
    counts_crit = {"ok": 0, "warn": 0, "fail": 0, "unknown": 0}

    offenders_fail: List[Dict[str, Any]] = []
    offenders_warn: List[Dict[str, Any]] = []
    offenders_unknown: List[Dict[str, Any]] = []

    for label, path, warn_s, fail_s, critical in tracked:
        meta = file_meta(path)
        age = float(meta.get("age_seconds")) if meta.get("exists") else np.nan
        status = "⚪" if not meta.get("exists") else health_color_for_age(age, warn_s, fail_s)

        # If rationale exists, signals.csv being missing/stale should not be treated as worse than warn.
        if label.startswith("signals.csv") and rationale_exists:
            status = _soften_status(status, "🟡")

        # Optional artifacts never force FAIL
        if (not critical) and status == "🔴":
            status = "🟡"

        bucket = _status_bucket(status)
        counts_all[bucket] += 1
        if critical:
            counts_crit[bucket] += 1

        row = {
            "Artifact": label,
            "Status": status,
            "Critical": critical,
            "Exists": bool(meta.get("exists")),
            "Size (KB)": (meta.get("size_bytes", 0) / 1024.0) if meta.get("exists") else 0.0,
            "Modified (ET)": dt_to_et_str(meta.get("mtime_utc")),
            "Age": fmt_age(age),
            "Path": str(path),
            "_age_seconds": age if np.isfinite(age) else np.inf,
        }
        rows.append(row)

        if bucket == "fail":
            offenders_fail.append(row)
        elif bucket == "warn":
            offenders_warn.append(row)
        elif bucket == "unknown":
            offenders_unknown.append(row)

        sc = STATUS_RANK.get(status, 0)
        if sc > worst_any_score:
            worst_any_score = sc
            worst_any = status
        if critical and sc > worst_critical_score:
            worst_critical_score = sc
            worst_critical = status

    overall = worst_critical

    # Heartbeat can downgrade overall
    if hb_status == "🔴":
        overall = "🔴"
    elif hb_status == "🟡" and overall == "🟢":
        overall = "🟡"
    elif hb_status == "⚪" and overall == "🟢":
        overall = "🟡"

    offenders_fail = sorted(
        offenders_fail, key=lambda r: r.get("_age_seconds", np.inf), reverse=True
    )
    offenders_warn = sorted(
        offenders_warn, key=lambda r: r.get("_age_seconds", np.inf), reverse=True
    )
    offenders_unknown = sorted(
        offenders_unknown, key=lambda r: r.get("_age_seconds", np.inf), reverse=True
    )

    for r in rows:
        r.pop("_age_seconds", None)
    for r in offenders_fail:
        r.pop("_age_seconds", None)
    for r in offenders_warn:
        r.pop("_age_seconds", None)
    for r in offenders_unknown:
        r.pop("_age_seconds", None)

    detail_bits = []
    if hb_path:
        detail_bits.append(f"Heartbeat: {hb_status}")
    detail_bits.append(f"Critical: {worst_critical}")
    detail_bits.append(f"All: {worst_any}")

    return {
        "overall": overall,
        "detail": " · ".join(detail_bits),
        "heartbeat": hb,
        "heartbeat_path": str(hb_path) if hb_path else "",
        "heartbeat_ts": hb_ts,
        "heartbeat_age_seconds": hb_age,
        "heartbeat_status": hb_status,
        "heartbeat_stage": hb_stage,
        "heartbeat_message": hb_msg,
        "heartbeat_error": hb_error,
        "warn_minutes": warn_minutes,
        "fail_minutes": fail_minutes,
        "counts_all": counts_all,
        "counts_critical": counts_crit,
        "offenders": {"fail": offenders_fail, "warn": offenders_warn, "unknown": offenders_unknown},
        "artifacts": rows,
    }


def pipeline_badge(health: Dict[str, Any]) -> Tuple[str, str]:
    overall = health.get("overall", "⚪")
    detail = str(health.get("detail", "")).strip()
    if overall == "🟢":
        return "🟢 Pipeline OK", detail or "Fresh artifacts"
    if overall == "🟡":
        return "🟡 Pipeline WARN", detail or "Some artifacts stale"
    if overall == "🔴":
        return "🔴 Pipeline FAIL", detail or "Missing/stale critical artifacts"
    return "⚪ Pipeline UNKNOWN", "No heartbeat / missing artifacts"


# ──────────────────────────────
# EXECUTION SAFETY GATE (A.3)
#   Locks any execution-capable UI when data is stale/unknown.
#   Source of truth: pipeline_health overall (computed from heartbeat + artifacts).
# ──────────────────────────────


def compute_execution_lock() -> Tuple[bool, str]:
    """
    HARD EXECUTION GATE:
      - Locks ALL execution-capable UIs unless BOTH are true:
          1) Pipeline health overall == 🟢
          2) Data Contracts are passing (no required artifact failures)

    Notes:
      - This gate is intentionally strict (Capital Preservation First).
      - Display-only pages remain accessible even when locked.
    """

    # Ensure pipeline health exists
    health = st.session_state.get("pipeline_health")
    if not isinstance(health, dict):
        try:
            health = compute_pipeline_health()
            st.session_state["pipeline_health"] = health
        except Exception:
            return True, "Execution locked: pipeline health unavailable"

    # Ensure contract status exists
    try:
        run_contracts_if_needed(force=False)
    except Exception:
        # If validator crashes, default to LOCK.
        st.session_state["contracts_ok"] = False

    overall = str(health.get("overall", "⚪")).strip()
    contracts_ok = bool(st.session_state.get("contracts_ok", False))

    reasons: List[str] = []
    if overall != "🟢":
        reasons.append(f"pipeline health {overall}")
    if not contracts_ok:
        reasons.append("data contracts FAIL")

    if reasons:
        return True, "Execution locked: " + " · ".join(reasons)

    return False, ""


# ──────────────────────────────
# LIVE EXEC PRECHECKS (ARM + GUARD)
#   Extra safety layer for Live Trading UI.
# ──────────────────────────────
def live_arm_status() -> Dict[str, Any]:
    """Return arm state from data/results/live_armed.json."""
    d = load_json(LIVE_ARMED_PATH, show_error=False) or {}
    armed = bool(d.get("armed", False))
    session = str(d.get("session", "") or "")
    expires_at = d.get("expires_at")

    exp_dt = parse_any_datetime(expires_at)
    if armed and exp_dt is not None:
        if datetime.now(timezone.utc) >= exp_dt:
            return {
                "armed": False,
                "reason": "EXPIRED",
                "expires_at": expires_at,
                "session": session,
            }

    return {
        "armed": armed,
        "reason": "ARMED" if armed else "DISARMED",
        "expires_at": expires_at,
        "session": session,
    }


def guard_status() -> Dict[str, Any]:
    d = load_json(GUARD_SNAPSHOT_PATH, show_error=False) or {}
    blocked = bool(d.get("blocked", False))
    kill = bool(d.get("kill_switch", False))
    code = str(d.get("code", "") or "")
    msg = str(d.get("message") or d.get("reason") or "")
    return {"blocked": blocked, "kill": kill, "code": code, "message": msg, "raw": d}


def _best_freshness_ts_and_source(health: dict) -> tuple:
    # Return (timestamp_utc, source_label) for the top freshness clock.
    ts = health.get("heartbeat_ts") if isinstance(health, dict) else None
    if ts is not None:
        return ts, "heartbeat"

    candidates = [
        ("portfolio_history.csv", file_mtime_dt(PORTFOLIO_HISTORY_PATH)),
        ("trade_log.csv", file_mtime_dt(TRADE_LOG_PATH)),
        ("signals_with_rationale.csv", file_mtime_dt(SIGNALS_RATIONALE_PATH)),
        ("signals.csv", file_mtime_dt(SIGNALS_PATH)),
    ]
    best_ts = None
    best_src = ""
    for name, dt in candidates:
        if dt is None:
            continue
        if best_ts is None or dt > best_ts:
            best_ts = dt
            best_src = name
    return best_ts, best_src


def render_top_status_stack() -> None:
    # Always-on top status stack:
    # 1) Freshness clock (authoritative)
    # 2) Execution gate (derived)
    if "pipeline_health" not in st.session_state:
        st.session_state["pipeline_health"] = compute_pipeline_health()
    health = st.session_state.get("pipeline_health") or {}

    # Freshness
    ts, src = _best_freshness_ts_and_source(health)
    if ts is None:
        css_class = "unknown"
        pill = "unknown"
        label = "DATA UNKNOWN"
        as_of = "-"
        age_str = "-"
    else:
        age_s = float((datetime.now(timezone.utc) - ts).total_seconds())
        as_of = dt_to_et_str(ts)
        age_str = fmt_age(age_s)
        if age_s <= 15 * 60:
            css_class = "fresh"
            pill = "fresh"
            label = "DATA FRESH"
        elif age_s <= 90 * 60:
            css_class = "aging"
            pill = "aging"
            label = "DATA AGING"
        else:
            css_class = "stale"
            pill = "stale"
            label = "DATA STALE"

    # Execution gate
    locked, reason = compute_execution_lock()
    exec_class = "locked" if locked else "enabled"
    exec_title = "EXECUTION LOCKED" if locked else "EXECUTION ENABLED"
    exec_detail = reason if locked else f"Pipeline overall: {health.get('overall','?')}"

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

    html = (
        f"<div class='triton-topwrap'>"
        f"<div class='triton-clock {css_class}'>"
        f"<div class='row'>"
        f"<div class='left'>{label} - as_of {as_of}</div>"
        f"<div class='mid'><span class='triton-pill {pill}'>age {age_str}</span></div>"
        f"<div class='right'>source {src or 'none'}</div>"
        f"</div></div>"
        f"<div class='triton-execbar {exec_class}'>"
        f"<div class='row'>"
        f"<div class='left'>{exec_title}</div>"
        f"<div class='right'>{exec_detail}</div>"
        f"</div></div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────
# PAGES (REAL DATA ONLY)
# ──────────────────────────────
def page_pipeline_health() -> None:
    st.markdown("### 🫀 Pipeline Health / Heartbeat")
    st.caption(
        "Real-data liveness + freshness. No fake metrics. Reads results artifacts + heartbeat.json."
    )

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        refresh = st.button("Refresh health", use_container_width=True)
    with c2:
        show_paths = st.toggle("Show paths", value=False)
    with c3:
        st.info(
            "Expected file: **data/results/heartbeat.json** (or pipeline_status.json fallback).",
            icon="💡",
        )

    if refresh:
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.session_state.pop("pipeline_health", None)
        st.rerun()

    if "pipeline_health" not in st.session_state:
        st.session_state["pipeline_health"] = compute_pipeline_health()

    health = st.session_state["pipeline_health"]
    badge, detail = pipeline_badge(health)

    counts_all = health.get("counts_all", {"ok": 0, "warn": 0, "fail": 0, "unknown": 0})
    counts_crit = health.get("counts_critical", {"ok": 0, "warn": 0, "fail": 0, "unknown": 0})

    with st.container(border=True):
        st.write(f"**Status:** {badge} · {detail}")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Critical FAIL", str(counts_crit.get("fail", 0)))
        a2.metric("Critical WARN", str(counts_crit.get("warn", 0)))
        a3.metric("All FAIL", str(counts_all.get("fail", 0)))
        a4.metric("All WARN", str(counts_all.get("warn", 0)))

        hb = health.get("heartbeat")
        hb_path = health.get("heartbeat_path", "")
        hb_ts = health.get("heartbeat_ts")
        hb_age = health.get("heartbeat_age_seconds", np.nan)
        hb_stage = health.get("heartbeat_stage", "")
        hb_msg = health.get("heartbeat_message", "")
        hb_err = health.get("heartbeat_error", "")

        if hb:
            left, right = st.columns([1, 1])
            with left:
                st.write("**Heartbeat source:**")
                st.code(hb_path or "—")
                st.write("**Last heartbeat (ET):**", dt_to_et_str(hb_ts) if hb_ts else "—")
                st.write("**Heartbeat age:**", fmt_age(hb_age))
                st.write("**Stage:**", hb_stage or "—")
            with right:
                if hb_msg:
                    st.write("**Message:**")
                    st.info(hb_msg)
                if hb_err:
                    st.write("**Error:**")
                    st.error(hb_err)
        else:
            st.warning(
                "No heartbeat file found. Create **data/results/heartbeat.json** (preferred) or pipeline_status.json.",
                icon="⚠️",
            )

    offenders = health.get("offenders", {})
    fail_list = offenders.get("fail", [])
    warn_list = offenders.get("warn", [])
    unk_list = offenders.get("unknown", [])

    st.markdown("#### 🔎 Breakdown (what exactly is failing / warning)")

    def _offender_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
        if not items:
            return pd.DataFrame()
        dfo = pd.DataFrame(items)
        keep = ["Artifact", "Status", "Critical", "Exists", "Modified (ET)", "Age"]
        keep = [c for c in keep if c in dfo.columns]
        return dfo[keep]

    b1, b2, b3 = st.columns(3)

    with b1:
        st.subheader("🔴 Fail")
        dff = _offender_df(fail_list)
        if not dff.empty:
            st.dataframe(dff, use_container_width=True, hide_index=True)
        else:
            st.caption("None")

    with b2:
        st.subheader("🟡 Warn")
        dfw = _offender_df(warn_list)
        if not dfw.empty:
            st.dataframe(dfw, use_container_width=True, hide_index=True)
        else:
            st.caption("None")

    with b3:
        st.subheader("⚪ Missing/Unknown")
        dfu = _offender_df(unk_list)
        if not dfu.empty:
            st.dataframe(dfu, use_container_width=True, hide_index=True)
        else:
            st.caption("None")

    df = pd.DataFrame(health.get("artifacts", []))
    if not df.empty:
        st.markdown("#### Artifact Freshness")
        df_view = df.drop(columns=["Path"]) if (not show_paths and "Path" in df.columns) else df
        st.dataframe(df_view, use_container_width=True, hide_index=True)


def render_global_header() -> None:
    st.markdown("## TRITON · COMMAND CENTER")
    st.caption(f"Capital Preservation First · Adaptive AI Execution · {APP_VERSION}")

    snap = latest_portfolio_status()
    market_lbl, market_detail = market_status_simple()

    run_contracts_if_needed(force=False)
    cs = st.session_state.get(
        "contract_summary",
        {"total": 0, "ok": 0, "failed": 0, "error_count": 0, "warn_count": 0, "info_count": 0},
    )
    badge_c, badge_detail_c = contracts_badge(cs)

    if "pipeline_health" not in st.session_state:
        st.session_state["pipeline_health"] = compute_pipeline_health()
    ph = st.session_state["pipeline_health"]
    badge_p, badge_detail_p = pipeline_badge(ph)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Regime", kpi(snap.get("mode"), "text"))
    c2.metric("Max Drawdown", kpi(snap.get("drawdown_pct", np.nan), "pct"))
    c3.metric("Buying Power", kpi(snap.get("buying_power", np.nan), "usd"))
    c4.metric("Updated", kpi(snap.get("updated"), "text"))
    c5.metric("Integrity", badge_c.replace(" Data Contracts ", " "), badge_detail_c)
    c6.metric("Pipeline", badge_p.replace(" Pipeline ", " "), badge_detail_p)

    st.markdown(
        f"<div style='padding:.5rem .75rem;border:1px solid rgba(148,163,184,.35);border-radius:10px;'>"
        f"<div style='font-weight:600;'>{market_lbl}</div>"
        f"<div style='opacity:.8;font-size:.9rem;'>{market_detail}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("🛡 Guard Details (real snapshot)"):
        if snap.get("reason"):
            st.write(snap["reason"])
        guard = load_json(GUARD_SNAPSHOT_PATH, show_error=False)
        if guard:
            st.json(guard)
        else:
            st.caption("No guard_snapshot.json yet.")


def page_data_contracts() -> None:
    st.markdown("### 🧾 Data Contracts (Validator)")
    st.caption("Phase 1.5 integrity gate. If this fails, downstream pages are not trusted.")

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        run_now = st.button("Run validation", use_container_width=True)
    with c2:
        strict_mode = st.toggle("Strict mode (warnings fail)", value=False)
    with c3:
        st.info("Run this after any pipeline changes, machine rebuild, or schema edits.", icon="💡")

    run_contracts_if_needed(force=run_now)

    results: List[ContractResult] = st.session_state.get("contract_results", [])
    summary = st.session_state.get("contract_summary", contracts_summary(results))
    badge, badge_detail = contracts_badge(summary)

    with st.container(border=True):
        st.write(
            f"**Status:** {badge} · {badge_detail}  |  "
            f"**Files:** {summary.get('total', 0)}  |  "
            f"**OK:** {summary.get('ok', 0)}  |  "
            f"**Failed:** {summary.get('failed', 0)}  |  "
            f"**Errors:** {summary.get('error_count', 0)}  |  "
            f"**Warnings:** {summary.get('warn_count', 0)}"
        )

        failed = (summary.get("failed", 0) > 0) or (summary.get("error_count", 0) > 0)
        if strict_mode and summary.get("warn_count", 0) > 0:
            failed = True

        if failed:
            st.error("Integrity gate FAILED. Fix artifacts before trusting analytics.", icon="🛑")
        else:
            st.success("Integrity gate PASSED.", icon="✅")

    rows: List[Dict[str, Any]] = []
    for r in results:
        err_n = sum(1 for i in r.issues if i.level == "ERROR")
        warn_n = sum(1 for i in r.issues if i.level == "WARN")
        info_n = sum(1 for i in r.issues if i.level == "INFO")
        rows.append(
            {
                "Contract": r.name,
                "OK": bool(r.ok),
                "Rows": int(r.row_count),
                "Cols": int(r.col_count),
                "Errors": int(err_n),
                "Warnings": int(warn_n),
                "Info": int(info_n),
                "Path": r.path,
            }
        )

    st.markdown("#### Summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Details")
    for r in results:
        title = f"{'✅' if r.ok else '❌'} {r.name} — {r.row_count} rows"
        with st.expander(title, expanded=not r.ok):
            st.code(r.path)
            if not r.issues:
                st.write("No issues.")
                continue
            for iss in r.issues:
                if iss.level == "ERROR":
                    st.error(f"[{iss.code}] {iss.message}")
                elif iss.level == "WARN":
                    st.warning(f"[{iss.code}] {iss.message}")
                else:
                    st.info(f"[{iss.code}] {iss.message}")
                if iss.hint:
                    st.caption(f"Hint: {iss.hint}")
                if iss.context:
                    st.json(iss.context)


def page_portfolio_history() -> None:
    st.markdown("### 📈 Portfolio History")

    df = load_csv(PORTFOLIO_HISTORY_PATH, show_error=True)
    if df is None or df.empty:
        st.warning("No portfolio history available (portfolio_history.csv missing/empty).")
        return

    df = sanitize_df_cols(df)
    df = ensure_date_col(df)

    if "ticker" not in df.columns:
        for alt in ("symbol", "sym", "Ticker", "Symbol"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break

    if "total_value" not in df.columns:
        for alt in ("equity", "portfolio_value", "portfolio_total", "value", "total"):
            if alt in df.columns:
                df = df.rename(columns={alt: "total_value"})
                break

    if "total_value" not in df.columns:
        st.error("portfolio_history.csv missing required column: total_value")
        st.dataframe(df.head(50), use_container_width=True)
        return

    df["total_value"] = safe_numeric(df, "total_value")
    if "cash" in df.columns:
        df["cash"] = safe_numeric(df, "cash")
    if "market_value" in df.columns:
        df["market_value"] = safe_numeric(df, "market_value")

    daily, per_ticker, method = aggregate_portfolio_history(df)
    if daily is None or daily.empty:
        st.warning("portfolio_history.csv loaded, but no valid (date,total_value) rows.")
        st.dataframe(df.head(50), use_container_width=True)
        return

    latest_dt = pd.to_datetime(daily["date"].iloc[-1], errors="coerce")
    latest_dt_py = latest_dt.to_pydatetime() if not pd.isna(latest_dt) else None
    mtime = file_mtime_dt(PORTFOLIO_HISTORY_PATH)

    range_opt = st.selectbox(
        "Range",
        options=["All", "30D", "90D", "1Y", "2Y"],
        index=2,
        key="ph_range",
        help="Filters the chart/table only (does not modify any files).",
    )
    if range_opt != "All" and latest_dt_py is not None:
        days = {"30D": 30, "90D": 90, "1Y": 365, "2Y": 730}[range_opt]
        cutoff = latest_dt_py - timedelta(days=days)
        daily_plot = daily[daily["date"] >= cutoff].copy()
    else:
        daily_plot = daily.copy()

    try:
        today = pd.Timestamp.today().normalize().to_pydatetime()
        if latest_dt_py is not None:
            age_days = (today - latest_dt_py).days
            if age_days >= 7:
                st.warning(
                    f"Portfolio data is **{age_days} days** old (latest row: {latest_dt_py.date()}). "
                    "Run the pipeline/simulation to refresh artifacts."
                )
    except Exception:
        pass

    parts = []
    if latest_dt_py is not None:
        parts.append(f"Data as-of: **{latest_dt_py:%Y-%m-%d}**")
    if mtime is not None:
        parts.append(f"File modified: **{dt_to_et_str(mtime)}**")
    if parts:
        st.caption(" · ".join(parts))

    tv = pd.to_numeric(daily_plot["total_value"], errors="coerce")
    tv = tv.dropna()
    if tv.empty:
        st.warning("No numeric total_value after coercion.")
        st.dataframe(daily_plot.tail(200), use_container_width=True)
        return

    latest = float(tv.iloc[-1])
    peak = float(tv.max())
    dd = (latest / peak - 1.0) if peak > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Total Value", f"${latest:,.2f}")
    c2.metric("Peak Total Value", f"${peak:,.2f}")
    c3.metric("Drawdown vs Peak", "—" if not np.isfinite(dd) else f"{dd*100:.2f}%")

    plotted = False
    try:
        import plotly.express as px  # type: ignore

        fig = px.line(daily_plot, x="date", y="total_value", title="Portfolio Total Value")
        st.plotly_chart(fig, use_container_width=True)
        plotted = True
    except Exception:
        plotted = False

    if per_ticker:
        st.caption(
            f"Per-ticker rows detected — aggregation method: **{method}** (prevents multiplying totals)."
        )

    if not plotted:
        st.caption("Plotly not available — showing aggregated table instead.")
        st.dataframe(daily_plot.tail(400), use_container_width=True)

    st.markdown("#### Aggregated daily totals (tail)")
    st.dataframe(daily_plot.tail(120), use_container_width=True, hide_index=True)

    st.markdown("#### Raw history (tail)")
    st.dataframe(df.tail(200), use_container_width=True)


def page_positions_exposure() -> None:
    st.markdown("### 🧾 Positions & Exposure")
    st.caption(
        "Read-only, real-data snapshot from **positions_snapshot.csv** (preferred). No broker calls."
    )

    df = load_csv(POSITIONS_SNAPSHOT_PATH, show_error=False)

    if (df is None or df.empty) and PORTFOLIO_HISTORY_PATH.exists():
        legacy = load_csv(PORTFOLIO_HISTORY_PATH, show_error=False)
        if legacy is not None and (
            "ticker" in legacy.columns or "symbol" in legacy.columns or "sym" in legacy.columns
        ):
            df = legacy

    if df is None or df.empty:
        st.warning(
            "No positions snapshot available. Create **data/results/positions_snapshot.csv** "
            "(recommended), or write per-ticker rows into portfolio_history.csv (legacy)."
        )
        return

    df = sanitize_df_cols(df)
    df = ensure_date_col(df)

    if "ticker" not in df.columns:
        for alt in ("symbol", "sym", "Ticker", "Symbol"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break

    if "ticker" not in df.columns:
        st.error("Positions snapshot missing required column: ticker")
        st.dataframe(df.head(80), use_container_width=True)
        return

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    value_col = None
    for c in ("market_value", "position_value", "value", "notional", "mv", "amount"):
        if c in df.columns:
            value_col = c
            break
    if value_col is None and "total_value" in df.columns and "market_value" not in df.columns:
        value_col = "total_value"
    if value_col is None:
        st.error(
            "Positions snapshot missing a value column (expected market_value / value / notional)."
        )
        st.dataframe(df.head(80), use_container_width=True)
        return

    df[value_col] = safe_numeric(df, value_col)

    df2 = df.dropna(subset=["date", "ticker", value_col]).copy()
    if df2.empty:
        st.warning("Positions snapshot has no valid (date,ticker,value) rows.")
        st.dataframe(df.head(80), use_container_width=True)
        return

    latest_dt = df2["date"].max()
    snap = df2[df2["date"] == latest_dt].copy()

    invested_value = float(snap[value_col].sum())
    if not np.isfinite(invested_value) or invested_value <= 0:
        st.warning("Latest snapshot has 0 invested value (all values are 0/NaN).")
        st.dataframe(snap.head(80), use_container_width=True)
        return

    snap["weight"] = snap[value_col] / invested_value
    snap["weight_pct"] = snap["weight"] * 100.0

    cash_val = np.nan
    if "cash" in snap.columns and snap["cash"].notna().any():
        cash_val = float(pd.to_numeric(snap["cash"], errors="coerce").dropna().iloc[0])

    total_val = np.nan
    if "total_value" in snap.columns and snap["total_value"].notna().any():
        total_val = float(pd.to_numeric(snap["total_value"], errors="coerce").dropna().iloc[0])

    n_pos = int(snap["ticker"].nunique())
    top1 = float(snap["weight"].max())
    top3 = float(snap.sort_values("weight", ascending=False)["weight"].head(3).sum())
    hhi = float((snap["weight"] ** 2).sum())

    # display as ET even if the source is naive
    latest_dt_disp = latest_dt.to_pydatetime() if hasattr(latest_dt, "to_pydatetime") else latest_dt
    if isinstance(latest_dt_disp, datetime) and latest_dt_disp.tzinfo is None:
        latest_dt_disp = latest_dt_disp.replace(tzinfo=timezone.utc)
    asof = dt_to_et_str(latest_dt_disp if isinstance(latest_dt_disp, datetime) else None)

    src_path = (
        POSITIONS_SNAPSHOT_PATH if POSITIONS_SNAPSHOT_PATH.exists() else PORTFOLIO_HISTORY_PATH
    )
    mtime = file_mtime_dt(src_path)
    if mtime is not None:
        st.caption(f"Source file: `{src_path.as_posix()}` • Modified: {dt_to_et_str(mtime)}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("As Of (ET)", asof)
    c2.metric("Positions", str(n_pos))
    c3.metric("Invested Value", kpi(invested_value, "usd"))
    c4.metric("Top 1 Weight", kpi(top1, "pct"))
    c5.metric("Top 3 Weight", kpi(top3, "pct"))

    if top1 >= 0.25:
        st.warning(
            f"Concentration risk: largest position is {top1*100:.1f}% of invested value.", icon="⚠️"
        )
    if top3 >= 0.60:
        st.warning(
            f"Concentration risk: top 3 positions are {top3*100:.1f}% of invested value.", icon="⚠️"
        )
    if hhi >= 0.18:
        st.warning(
            f"Portfolio concentration (HHI={hhi:.3f}) is elevated. Consider diversification.",
            icon="⚠️",
        )

    if np.isfinite(cash_val) and np.isfinite(total_val) and total_val > 0:
        st.caption(
            f"Cash (best-effort): {kpi(cash_val,'usd')} · Cash%: {kpi(cash_val/total_val,'pct')}"
        )
    elif np.isfinite(cash_val):
        st.caption(f"Cash (best-effort): {kpi(cash_val,'usd')}")

    snap_view = snap.sort_values("weight", ascending=False).copy()

    plotted = False
    try:
        import plotly.express as px  # type: ignore

        topn = min(25, len(snap_view))
        fig = px.bar(
            snap_view.head(topn),
            x="ticker",
            y="weight_pct",
            title="Top positions by weight (%, latest snapshot)",
        )
        st.plotly_chart(fig, use_container_width=True)
        plotted = True
    except Exception:
        plotted = False

    cols = ["ticker", value_col, "weight_pct"]
    for extra in ["shares", "price", "cash", "total_value"]:
        if extra in snap_view.columns and extra not in cols:
            cols.insert(2, extra)

    snap_tbl = snap_view.rename(columns={value_col: "value"})
    show_cols = [
        c
        for c in ["ticker", "value"] + [c for c in cols if c not in ("ticker", value_col)]
        if c in snap_tbl.columns
    ]

    # Display-only formatting for readability
    snap_disp = format_df_for_display(
        snap_tbl[show_cols],
        money_cols=[c for c in ["value", "cash", "total_value"] if c in snap_tbl.columns],
        pct_cols=[c for c in ["weight_pct"] if c in snap_tbl.columns],
    )
    st.dataframe(snap_disp, use_container_width=True, hide_index=True)

    st.markdown("#### Raw snapshot rows (tail)")
    st.dataframe(snap.tail(200), use_container_width=True)


def page_trade_log() -> None:
    st.markdown("### 📜 Trade Log")
    df = load_csv(TRADE_LOG_PATH, show_error=True)
    if df is None or df.empty:
        st.warning("No trade log available (trade_log.csv missing/empty).")
        return
    # Display-only formatting (best-effort)
    df2 = sanitize_df_cols(df.copy())
    money_cols = [
        c
        for c in [
            "pnl",
            "profit",
            "loss",
            "notional",
            "value",
            "limit_price",
            "fill_price",
            "price",
        ]
        if c in df2.columns
    ]
    df_disp = format_df_for_display(
        df2,
        money_cols=money_cols,
        pct_cols=[c for c in ["pnl_pct", "return", "return_pct"] if c in df2.columns],
        usd_decimals=2,
    )
    st.dataframe(df_disp, use_container_width=True)


def page_signal_lifecycle() -> None:
    st.markdown("### 🧭 Signal Lifecycle")
    st.caption(
        "Authoritative, stateful stance per ticker: **BUY / ADD / HOLD / TRIM / EXIT**. "
        "This page reads the STATE artifact (one row per ticker)."
    )

    df = load_csv(SIGNAL_LIFECYCLE_PATH, show_error=False)
    if df is None or df.empty:
        st.error("signal_lifecycle.csv not found or empty.")
        st.code(
            "python services/build_signal_lifecycle_state.py --results-dir data/results",
            language="bash",
        )
        return

    df = sanitize_df_cols(df)

    # Freshness flag (prefer per-row column, else derive from heartbeat)
    freshness = None
    if "freshness" in df.columns:
        freshness = str(df["freshness"].iloc[0]) if len(df) else None
    if not freshness or freshness.lower() in ("nan", "none", ""):
        hb = load_json(HEARTBEAT_PATH, show_error=False)
        if hb and isinstance(hb, dict):
            freshness = hb.get("freshness") or hb.get("status")

    freshness_u = str(freshness).upper() if freshness else ""
    if freshness:
        icon = "✅" if freshness_u in ("OK", "FRESH") else "⚠️"
        st.info(f"Lifecycle freshness flag: **{freshness}**", icon=icon)

    # Freeze banner (visual authority): if stale/unknown, lifecycle should be treated as frozen
    if freshness_u not in ("OK", "FRESH", "AGING") and freshness_u:
        st.warning(
            "Lifecycle is **FROZEN** due to stale/unknown pipeline freshness. "
            "Do not act on BUY/ADD/EXIT until freshness returns to OK.",
            icon="🧊",
        )

    # Filters
    stance_col = (
        "stance"
        if "stance" in df.columns
        else ("lifecycle_action" if "lifecycle_action" in df.columns else None)
    )
    ticker_col = (
        "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
    )

    if ticker_col and ticker_col != "ticker":
        df = df.rename(columns={ticker_col: "ticker"})
        ticker_col = "ticker"
    if stance_col and stance_col != "stance":
        df = df.rename(columns={stance_col: "stance"})
        stance_col = "stance"

    if ticker_col != "ticker" or "ticker" not in df.columns:
        st.error("signal_lifecycle.csv missing required column: ticker")
        st.dataframe(df.head(50), use_container_width=True)
        return

    if "stance" not in df.columns:
        st.error("signal_lifecycle.csv missing required column: stance")
        st.dataframe(df.head(50), use_container_width=True)
        return

    # Summary strip (instant posture)
    counts = df["stance"].astype(str).str.upper().value_counts(dropna=False)
    order = ["BUY", "ADD", "HOLD", "TRIM", "EXIT"]
    ccols = st.columns(len(order))
    for i, k in enumerate(order):
        with ccols[i]:
            st.metric(k, int(counts.get(k, 0)))

    stances = sorted([s for s in df["stance"].dropna().astype(str).unique()])
    default = [s for s in ["BUY", "ADD", "EXIT", "TRIM"] if s in stances] or stances

    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        stance_pick = st.multiselect("Filter stances", options=stances, default=default)
    with col2:
        only_actionable = st.checkbox("Only actionable", value=True)
    with col3:
        q = st.text_input("Ticker contains", value="")

    view = df.copy()
    if stance_pick:
        view = view[view["stance"].astype(str).isin(stance_pick)]
    if only_actionable:
        view = view[~view["stance"].astype(str).isin(["HOLD", "WAIT"])]
    if q:
        view = view[view["ticker"].astype(str).str.contains(q.upper(), na=False)]

    # Sort and show
    if "ticker" in view.columns:
        view = view.sort_values("ticker")

    # Visual authority: stance badges + dim HOLD rows (display-only)
    if "stance" in view.columns:
        view["stance"] = view["stance"].astype(str).str.upper()
        sty = view.style.apply(_dim_hold_rows, axis=1)
        sty = sty.applymap(_stance_cell_css, subset=["stance"])
        st.dataframe(sty, use_container_width=True, hide_index=True)
    else:
        st.dataframe(view, use_container_width=True, hide_index=True)


def page_ai_signals() -> None:
    st.markdown("### 🤖 AI Signals")
    st.caption(
        "Priority order: **Signal Lifecycle (STATE) → Signals w/ Rationale → Raw Signals**. Lifecycle rows represent Triton’s current stance per ticker."
    )

    df = load_first_nonempty_csv(
        [
            SIGNAL_LIFECYCLE_PATH,
            SIGNALS_RATIONALE_PATH,
            SIGNALS_PATH,
        ]
    )

    if df is None or df.empty:
        st.warning(
            "No signals available (signal_lifecycle.csv / signals_with_rationale.csv / signals.csv missing or empty)."
        )
        return

    df = sanitize_df_cols(df)

    # Display-only formatting (keep raw numerics intact elsewhere)
    df_disp = format_df_for_display(
        df,
        int_cols=[c for c in ["market_cap", "revenue", "volume"] if c in df.columns],
        pct_cols=[
            c
            for c in ["confidence", "edge_pct", "delta_pct", "return", "returns"]
            if c in df.columns
        ],
    )
    # Apply stance badge styling if present
    if "stance" in df_disp.columns:
        df_disp["stance"] = df_disp["stance"].astype(str).str.upper()
        sty = df_disp.style.apply(_dim_hold_rows, axis=1)
        sty = sty.applymap(_stance_cell_css, subset=["stance"])
        st.dataframe(sty, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_disp, use_container_width=True, hide_index=True)


def page_trade_rationale() -> None:
    st.markdown("### 🧠 Trade Rationale")
    df = load_csv(SIGNALS_RATIONALE_PATH, show_error=True)
    if df is None or df.empty:
        st.info("No signals_with_rationale.csv found yet. Run your training step that writes it.")
        return
    df = sanitize_df_cols(df)
    df = ensure_date_col(df)
    cols = [
        c
        for c in ["date", "ticker", "signal", "confidence", "position_size", "rationale"]
        if c in df.columns
    ]
    view = df[cols] if cols else df
    view_disp = format_df_for_display(
        view,
        pct_cols=[c for c in ["confidence", "edge_pct", "delta_pct"] if c in view.columns],
    )
    st.dataframe(view_disp, use_container_width=True)


def page_target_weights() -> None:
    st.markdown("### 🎛 Target Weights")
    st.caption(
        "Reads: data/results/target_weights.csv (real allocation targets; no synthetic numbers)."
    )

    df = load_csv(TARGET_WEIGHTS_PATH, show_error=True)
    if df is None or df.empty:
        st.info(
            "No target_weights.csv found yet. Generate it from your weighting step (e.g., allocator)."
        )
        return

    df = sanitize_df_cols(df)

    if "ticker" not in df.columns:
        for alt in ("symbol", "sym", "Ticker", "Symbol"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break

    if "ticker" not in df.columns:
        st.error("target_weights.csv missing required column: 'ticker'")
        st.dataframe(df.head(50), use_container_width=True)
        return

    weight_candidates = ["target_weight", "weight", "allocation", "pct", "percent"]
    wcol = next((c for c in weight_candidates if c in df.columns), None)

    if wcol:
        df[wcol] = safe_numeric(df, wcol)
        finite = df[wcol].dropna()
        if not finite.empty and float(finite.max()) > 1.5:
            df[wcol] = df[wcol] / 100.0
        df = df.sort_values(wcol, ascending=False, na_position="last")

    cols = ["ticker"] + ([wcol] if wcol else [])
    for extra in ["regime", "capital_mode", "reason", "as_of", "date"]:
        if extra in df.columns:
            cols.append(extra)

    st.dataframe(df[cols] if cols else df, use_container_width=True)

    if wcol:
        total = float(pd.to_numeric(df[wcol], errors="coerce").sum(skipna=True))
        st.caption(
            f"Sum of weights ({wcol}): {total:.4f}  (not forced to 1.0; depends on your allocator)"
        )


def page_top_picks() -> None:
    st.markdown("### ⭐ Top Picks")
    df = load_csv(STOCK_SCORES_PATH, show_error=True)
    if df is None or df.empty:
        st.warning("No scores available (stock_scores.csv missing/empty).")
        return

    df2 = sanitize_df_cols(df.copy())

    # Sort by best-available score column
    score_candidates = ["total_score", "score_total", "score", "final_score"]
    score_col = next((c for c in score_candidates if c in df2.columns), None)
    if score_col:
        df2[score_col] = safe_numeric(df2, score_col)
        df2 = df2.sort_values(score_col, ascending=False, na_position="last")

    # Display-only formatting: commas for large money-like fields
    money_like = [
        "revenue",
        "market_cap",
        "enterprise_value",
        "ebitda",
        "free_cash_flow",
        "total_assets",
        "total_liabilities",
    ]

    usd_like = [c for c in df2.columns if c.lower().endswith("_usd")]
    fmt_int_cols = [c for c in money_like + usd_like if c in df2.columns]

    df_disp = format_df_for_display(
        df2,
        int_cols=fmt_int_cols,
        pct_cols=[c for c in ["confidence", "edge_pct", "delta_pct"] if c in df2.columns],
    )

    st.dataframe(df_disp, use_container_width=True, hide_index=True)


def page_feature_importance() -> None:
    st.markdown("### 🧠 Feature Importance")
    df = load_csv(FEATURE_IMPORTANCE_PATH, show_error=True)
    if df is None or df.empty:
        st.info("No feature_importance.csv yet. Your train step writes it when available.")
        return
    df = sanitize_df_cols(df)
    df_disp = format_df_for_display(
        df,
        money_cols=[c for c in ["pnl", "profit", "notional", "amount", "value"] if c in df.columns],
        pct_cols=[c for c in ["return", "returns"] if c in df.columns],
    )
    st.dataframe(df_disp, use_container_width=True)


def page_model_comparison() -> None:
    st.markdown("### 📊 Model Comparison")
    df = load_csv(MODEL_COMPARISON_PATH, show_error=True)
    if df is None or df.empty:
        st.info("No model_comparison.csv yet. Your train step writes it when available.")
        return
    st.dataframe(df, use_container_width=True)


def page_risk_report() -> None:
    st.markdown("### 🛡 Risk Report")
    st.caption(
        "Prefers **risk_report.json** (generate_risk_report.py). Falls back to adaptive_risk_state.json if present."
    )

    data = load_json(RISK_REPORT_PATH, show_error=False)
    used = "risk_report.json"

    if not data:
        data = load_json(RISK_STATE_PATH, show_error=False)
        used = "adaptive_risk_state.json"

    if not data:
        st.warning(
            "No risk JSON found (risk_report.json missing/empty, and adaptive_risk_state.json missing/empty)."
        )
        return

    st.caption(f"Loaded: **{used}**")
    st.json(data)


def page_strategy_diagnostics() -> None:
    st.markdown("### 🧪 Strategy Diagnostics")
    df = load_csv(TRADE_LOG_PATH, show_error=True)
    if df is None or df.empty:
        st.warning("No trade log available (trade_log.csv missing/empty).")
        return

    df = sanitize_df_cols(df)
    df = ensure_date_col(df)

    pnl_col = "pnl" if "pnl" in df.columns else ("profit" if "profit" in df.columns else None)
    if pnl_col is None:
        st.error("trade_log.csv is missing P&L. Expected a 'pnl' or 'profit' column.")
        st.caption("Add realized P&L on exits, then this panel will auto-activate.")
        st.dataframe(df.head(30), use_container_width=True)
        return

    df[pnl_col] = safe_numeric(df, pnl_col)
    df2 = df.dropna(subset=[pnl_col]).copy()

    total = int(len(df2))
    net = float(df2[pnl_col].sum()) if total else 0.0
    win_rate = float((df2[pnl_col] > 0).mean()) if total else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", str(total))
    c2.metric("Net P&L", f"${net:,.2f}")
    c3.metric("Win Rate", "—" if not np.isfinite(win_rate) else f"{win_rate * 100:.1f}%")

    st.dataframe(df2, use_container_width=True)


def page_news_sentiment() -> None:
    st.markdown("### 📰 News Sentiment")
    df = load_csv(RESULTS_DIR / "news_sentiment.csv", show_error=True)
    if df is None or df.empty:
        st.warning("No news sentiment available (news_sentiment.csv missing/empty).")
        return
    st.dataframe(df, use_container_width=True)


def page_smart_alerts() -> None:
    st.markdown("### 🚨 Smart Alerts")
    df = load_csv(RESULTS_DIR / "smart_alerts.csv", show_error=True)
    if df is None or df.empty:
        st.info("No smart alerts yet (smart_alerts.csv missing/empty).")
        return
    st.dataframe(df, use_container_width=True)


def page_econ_calendar() -> None:
    st.markdown("### 📅 Economic Calendar")
    df = load_csv(RESULTS_DIR / "economic_calendar.csv", show_error=True)
    if df is None or df.empty:
        st.info("No economic calendar file found (economic_calendar.csv missing/empty).")
        return
    st.dataframe(df, use_container_width=True)


def page_execution_health() -> None:
    st.markdown("### ⚙ Execution / Health")

    c1, c2 = st.columns(2)

    with c1:
        st.caption("Open orders log (live_orders.csv)")
        df = load_csv(LIVE_ORDERS_PATH, show_error=True)
        if df is not None and not df.empty:
            st.dataframe(df.tail(50), use_container_width=True)
        else:
            st.info("No live_orders.csv yet.")

    with c2:
        st.caption("Guard snapshot (guard_snapshot.json)")
        guard = load_json(GUARD_SNAPSHOT_PATH, show_error=True)
        if guard:
            st.json(guard)
        else:
            st.info("No guard_snapshot.json yet.")


def _fmt_timedelta(td: Optional[pd.Timedelta]) -> str:
    if td is None or pd.isna(td):
        return "—"
    secs = float(td.total_seconds())
    if secs < 60:
        return f"{int(secs)}s"
    if secs < 3600:
        return f"{int(secs // 60)}m"
    if secs < 86400:
        return f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m"
    return f"{int(secs // 86400)}d {int((secs % 86400) // 3600)}h"


def page_live_orders_panel() -> None:
    """
    Phase 2.3 — Live Orders Panel (READ-ONLY)
      - reads: data/results/live_orders.csv
      - shows: open orders + status + age + side/qty/price
      - NO execution authority
    """
    st.markdown("### 🚦 Live Orders Panel (Read-only)")
    st.caption("Reads **data/results/live_orders.csv**. No broker actions here (display only).")

    df = load_csv(LIVE_ORDERS_PATH, show_error=True)
    if df is None:
        st.warning("live_orders.csv not found.")
        return

    df = sanitize_df_cols(df)
    if df.empty:
        st.info("live_orders.csv loaded, but contains 0 rows.")
        return

    # Normalize common columns
    if "ticker" not in df.columns:
        for alt in ("symbol", "sym", "Ticker", "Symbol"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break

    if "qty" not in df.columns:
        for alt in ("quantity", "order_qty", "qty_ordered"):
            if alt in df.columns:
                df = df.rename(columns={alt: "qty"})
                break

    if "price" not in df.columns:
        for alt in (
            "limit_price",
            "lmt_price",
            "order_price",
            "filled_avg_price",
            "avg_fill_price",
        ):
            if alt in df.columns:
                df = df.rename(columns={alt: "price"})
                break

    if "status" not in df.columns:
        for alt in ("order_status", "state"):
            if alt in df.columns:
                df = df.rename(columns={alt: "status"})
                break

    # Choose a timestamp column for age
    ts_col = next(
        (
            c
            for c in ["submitted_at", "created_at", "updated_at", "timestamp", "date", "time"]
            if c in df.columns
        ),
        None,
    )
    if ts_col:
        df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        df["_ts"] = pd.NaT

    # ✅ FIX: Series subtraction already yields Timedelta; don't wrap in pd.to_timedelta()
    now_utc = pd.Timestamp.now(tz="UTC")
    df["_age"] = (now_utc - df["_ts"]) if df["_ts"].notna().any() else pd.NaT

    # Filter to "open-ish" orders if we have status
    open_df = df.copy()
    if "status" in open_df.columns:
        s = open_df["status"].astype("string").str.lower().fillna("")
        closed = {
            "filled",
            "canceled",
            "cancelled",
            "rejected",
            "expired",
            "done_for_day",
            "stopped",
            "replaced",
        }
        open_df = open_df[~s.isin(closed)]

    open_df = open_df.sort_values("_ts", ascending=False, na_position="last")

    n_open = int(len(open_df))
    oldest = open_df["_age"].max() if n_open else pd.NaT
    newest = open_df["_age"].min() if n_open else pd.NaT

    c1, c2, c3 = st.columns(3)
    c1.metric("Open Orders", str(n_open))
    c2.metric("Oldest Age", _fmt_timedelta(oldest))
    c3.metric("Newest Age", _fmt_timedelta(newest))

    show_cols_pref = [
        "ticker",
        "side",
        "qty",
        "price",
        "status",
        "type",
        "time_in_force",
        "filled_qty",
        "filled_avg_price",
        "submitted_at",
        "created_at",
        "updated_at",
    ]
    show_cols = [c for c in show_cols_pref if c in open_df.columns]

    open_view = open_df.copy()
    open_view["age"] = open_view["_age"].apply(_fmt_timedelta)

    cols_final: List[str] = []
    for c in ["age"] + show_cols:
        if c in open_view.columns and c not in cols_final:
            cols_final.append(c)

    if cols_final:
        st.dataframe(open_view[cols_final], use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            open_view.drop(columns=["_ts", "_age"], errors="ignore"), use_container_width=True
        )

    with st.expander("Raw live_orders.csv (tail)"):
        st.dataframe(df.tail(200), use_container_width=True)


# ──────────────────────────────
# OPTIONAL PAGES (LAZY IMPORT INSIDE RENDER)
# ──────────────────────────────
def page_live_trading() -> None:
    st.markdown("### 🔴 Live Trading (Hard-Gated)")
    st.caption("ARM + GUARD controls and broker snapshot. No direct order placement in this UI.")

    # HARD Execution Gate — if locked, do not render this panel.
    locked, reason = compute_execution_lock()
    if locked:
        st.error("Execution blocked: " + (reason or "Gate is locked"))
        st.info("Fix freshness / contracts first.")
        return

    # Extra safety readouts (ARM + GUARD). The panel also enforces these before showing broker data.
    arm = live_arm_status()
    guard = guard_status()

    c1, c2 = st.columns(2)
    with c1:
        if arm.get("armed"):
            st.success(f"ARM: {arm.get('reason', 'ARMED')}")
            if arm.get("expires_at"):
                st.caption(f"expires_at={arm.get('expires_at')}")
            if arm.get("session"):
                st.caption(f"session={arm.get('session')}")
        else:
            st.error(f"ARM: {arm.get('reason', 'DISARMED')}")
            st.caption("Arm live trading before expecting broker snapshot visibility.")

    with c2:
        if guard.get("blocked") or guard.get("kill"):
            st.error("GUARD: BLOCKED")
            st.caption(f"{guard.get('code','')}: {guard.get('message','')}".strip())
        else:
            st.success("GUARD: CLEAR")
            st.caption("blocked=false • kill_switch=false")

    st.divider()

    if not HAS_LIVE_TRADING_PANEL:
        st.error("Live Trading panel file not found: dashboard/live_trading_panel.py")
        return

    fn = _lazy_import("dashboard.live_trading_panel:render_live_trading_panel")
    if not callable(fn):
        st.error(
            "render_live_trading_panel() not importable. Check dashboard/live_trading_panel.py for errors."
        )
        return

    fn()


def page_run_csv_orders() -> None:
    st.markdown("### ▶ Run CSV Orders")

    locked, reason = compute_execution_lock()
    if locked:
        st.error(reason)
        st.info("Refresh the pipeline (fresh artifacts + heartbeat) before using execution pages.")
        return

    st.caption(
        "Dashboard wrapper for place_orders_from_csv.py. DRY RUN by default; REAL placement only when toggled."
    )

    if not HAS_RUN_CSV_ORDERS:
        st.error("Run CSV Orders page not found.")
        st.info(
            "Create: **ui/pages/run_csv_orders_page.py** exposing `render()` and restart Streamlit."
        )
        return

    fn = _lazy_import("ui.pages.run_csv_orders_page:render")
    if not fn:
        st.error(
            "ui/pages/run_csv_orders_page.py exists, but `render()` is missing or not callable."
        )
        st.info("Export a function named `render()` from that module.")
        return

    fn()


def page_manual_order_desk() -> None:
    st.markdown("### 🧾 Manual Order Desk")

    locked, reason = compute_execution_lock()
    if locked:
        st.error(reason)
        st.info("Refresh the pipeline (fresh artifacts + heartbeat) before using execution pages.")
        return

    st.caption("UI wrapper for manual broker actions (place/cancel/view).")

    if not HAS_MANUAL_ORDER_DESK:
        st.error("Manual Order Desk module not found.")
        st.info(
            "Create: **pages/manual_order_desk.py** with `render_manual_order_desk()` and restart Streamlit."
        )
        return

    fn = _lazy_import("pages.manual_order_desk:render_manual_order_desk")
    if not fn:
        st.error(
            "pages/manual_order_desk.py exists, but `render_manual_order_desk()` is missing or not callable."
        )
        st.info("Export a function named `render_manual_order_desk()` from that module.")
        return

    fn()


def page_live_run() -> None:
    st.markdown("### 🟢 Live Run (Phase 1.5)")

    locked, reason = compute_execution_lock()
    if locked:
        st.error(reason)
        st.info("Refresh the pipeline (fresh artifacts + heartbeat) before using execution pages.")
        return

    st.success("Live Run page is active. Wire it to ui.panels.live_run_panel when ready.")
    st.caption("Tip: keep this page real-data only (log pipeline heartbeat to results/).")


# ──────────────────────────────
# ROUTER (single source of truth)
# ──────────────────────────────
PAGE_REGISTRY: Dict[Tuple[str, str], Callable[[], None]] = {
    ("Portfolio", "Portfolio History"): page_portfolio_history,
    ("Portfolio", "Positions & Exposure"): page_positions_exposure,
    ("Portfolio", "Trade Log"): page_trade_log,
    ("Signals", "🧭 Signal Lifecycle"): page_signal_lifecycle,
    ("Signals", "AI Signals"): page_ai_signals,
    ("Signals", "🧠 Trade Rationale"): page_trade_rationale,
    ("Signals", "🎛 Target Weights"): page_target_weights,
    ("Signals", "Top Picks"): page_top_picks,
    ("Signals", "Feature Importance"): page_feature_importance,
    ("Signals", "Model Comparison"): page_model_comparison,
    ("Risk & Guardrails", "Risk Report"): page_risk_report,
    ("Risk & Guardrails", "Strategy Diagnostics"): page_strategy_diagnostics,
    ("Research / Intel", "News Sentiment"): page_news_sentiment,
    ("Research / Intel", "Smart Alerts"): page_smart_alerts,
    ("Research / Intel", "Economic Calendar"): page_econ_calendar,
    ("System", "🧾 Data Contracts"): page_data_contracts,
    ("System", "🫀 Pipeline Health / Heartbeat"): page_pipeline_health,
    ("System", "Execution / Health"): page_execution_health,
    ("System", "🚦 Live Orders Panel"): page_live_orders_panel,
    ("System", "🟢 Live Run (Phase 1.5)"): page_live_run,
}

# Optional pages
if HAS_LIVE_TRADING_PANEL:
    PAGE_REGISTRY[("System", "🔴 Live Trading")] = page_live_trading
if HAS_RUN_CSV_ORDERS:
    PAGE_REGISTRY[("System", "▶ Run CSV Orders")] = page_run_csv_orders
if HAS_MANUAL_ORDER_DESK:
    PAGE_REGISTRY[("System", "🧾 Manual Order Desk")] = page_manual_order_desk

SECTIONS: Dict[str, List[str]] = {
    "Portfolio": ["Portfolio History", "Positions & Exposure", "Trade Log"],
    "Signals": [
        "🧭 Signal Lifecycle",
        "AI Signals",
        "🧠 Trade Rationale",
        "🎛 Target Weights",
        "Top Picks",
        "Feature Importance",
        "Model Comparison",
    ],
    "Risk & Guardrails": ["Risk Report", "Strategy Diagnostics"],
    "Research / Intel": ["News Sentiment", "Smart Alerts", "Economic Calendar"],
    "System": [
        "🧾 Data Contracts",
        "🫀 Pipeline Health / Heartbeat",
        "Execution / Health",
        "🚦 Live Orders Panel",
        "🟢 Live Run (Phase 1.5)",
    ],
}

# Insert optional system pages near the bottom but before Live Run
if HAS_LIVE_TRADING_PANEL:
    SECTIONS["System"].insert(-1, "🔴 Live Trading")
if HAS_RUN_CSV_ORDERS:
    SECTIONS["System"].insert(-1, "▶ Run CSV Orders")
if HAS_MANUAL_ORDER_DESK:
    SECTIONS["System"].insert(-1, "🧾 Manual Order Desk")


def sidebar_controls() -> Tuple[str, str]:
    with st.sidebar:
        st.title("TRITON Nav")
        st.caption(f"Build: {APP_VERSION}")

        if st.button("⟳ Refresh data", key="refresh"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.session_state.pop("contract_results", None)
            st.session_state.pop("contract_summary", None)
            st.session_state.pop("pipeline_health", None)
            st.rerun()

        locked, reason = compute_execution_lock()
        if locked:
            # Use emoji in text for broad Streamlit compatibility (avoid icon kwarg issues)
            st.error("🛑 EXECUTION LOCKED")
            st.caption(reason)
        else:
            st.success("✅ Execution enabled")

        st.markdown("---")
        section = st.selectbox("Section", list(SECTIONS.keys()), index=0, key="section_choice")
        page = st.radio("View", SECTIONS[section], index=0, key="sub_choice")

        st.markdown("---")
        st.caption(f"Repo root: {PROJECT_ROOT}")

    return section, page


def main() -> None:
    section, page = sidebar_controls()
    render_top_status_stack()
    render_global_header()

    fn = PAGE_REGISTRY.get((section, page))
    if fn is None:
        st.error(f"❌ Page not wired: {section} → {page}")
        return

    run_contracts_if_needed(force=False)
    if st.session_state.get("contracts_ok") is False and not (
        section == "System" and page == "🧾 Data Contracts"
    ):
        st.warning(
            "Data Contracts are failing. Some pages may be unreliable. Open **System → Data Contracts**.",
            icon="⚠️",
        )

    if "pipeline_health" not in st.session_state:
        st.session_state["pipeline_health"] = compute_pipeline_health()
    overall = st.session_state["pipeline_health"].get("overall", "⚪")
    if overall in ("🔴", "⚪") and not (
        section == "System" and page == "🫀 Pipeline Health / Heartbeat"
    ):
        st.warning(
            "Pipeline health looks degraded/missing. Open **System → Pipeline Health / Heartbeat**.",
            icon="🫀",
        )

    fn()


if __name__ == "__main__":
    main()
