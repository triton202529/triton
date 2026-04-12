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
#
# PATCH (lifecycle authority / freshness):
#   ✅ AI Signals uses signal_lifecycle.csv only when mtime ≥ upstream rationale/raw signals

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

from services.live_gate import compute_live_gates
from services.schema_guard import (
    dedupe_columns,
    find_duplicate_columns,
    require_columns,
    schema_snapshot,
)

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
TRADE_OPPORTUNITIES_PATH = RESULTS_DIR / "trade_opportunities.csv"
LIFECYCLE_RECONCILIATION_PATH = RESULTS_DIR / "lifecycle_reconciliation.csv"
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

# Execution Dashboard (read-only observability)
PAPER_TRADE_CYCLE_SUMMARY_PATH = RESULTS_DIR / "paper_trade_cycle_summary.json"
PAPER_TRADE_CYCLE_LOG_PATH = RESULTS_DIR / "paper_trade_cycle_log.csv"
EXECUTION_PLAN_JSON_PATH = RESULTS_DIR / "execution_plan.json"
EXECUTION_PLAN_CSV_PATH = RESULTS_DIR / "execution_plan.csv"
MANAGE_POSITIONS_PLAN_JSON_PATH = RESULTS_DIR / "manage_positions_plan.json"
MANAGE_POSITIONS_PLAN_CSV_PATH = RESULTS_DIR / "manage_positions_plan.csv"
CAPITAL_REALLOCATION_JSON_PATH = RESULTS_DIR / "capital_reallocation.json"
REALLOCATION_PLAN_CSV_PATH = RESULTS_DIR / "reallocation_plan.csv"
EXEC_DROP_JSON_PATH = RESULTS_DIR / "execution_drop_diagnostics.json"
EXEC_DROP_CSV_PATH = RESULTS_DIR / "execution_drop_diagnostics.csv"
SIGNAL_PRESSURE_DIAG_PATH = RESULTS_DIR / "signal_pressure_diagnostics.json"
SIGNAL_PRESSURE_DIAG_CSV_PATH = RESULTS_DIR / "signal_pressure_diagnostics.csv"
EXECUTION_PRESSURE_PATH = RESULTS_DIR / "execution_pressure.json"
SESSION_FILL_PRESSURE_PATH = RESULTS_DIR / "session_fill_pressure.json"
OPEN_ORDER_PRESSURE_PATH = RESULTS_DIR / "open_order_pressure.json"
STALE_OPEN_ORDERS_CSV_PATH = RESULTS_DIR / "stale_open_orders.csv"
REPRICE_OPEN_ORDERS_PATH = RESULTS_DIR / "reprice_open_orders.json"
REPRICE_LADDER_RUN_PATH = RESULTS_DIR / "reprice_ladder_run.json"
SIGNAL_LIFECYCLE_EFFECTIVE_PATH = RESULTS_DIR / "signal_lifecycle_effective.csv"
OPEN_ORDERS_SNAPSHOT_PATH = RESULTS_DIR / "open_orders_snapshot.csv"
RECENT_ORDERS_PATH = RESULTS_DIR / "recent_orders.csv"
LIVE_ORDERS_LOG_PATH = RESULTS_DIR / "live_orders_log.csv"

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


def _signals_upstream_reference_mtime() -> Optional[datetime]:
    """Latest mtime among non-empty rationale and raw signals CSVs (lifecycle freshness reference)."""
    ts: List[datetime] = []
    for p in (SIGNALS_RATIONALE_PATH, SIGNALS_PATH):
        t = file_mtime_dt(p)
        if t is not None:
            ts.append(t)
    if not ts:
        return None
    return max(ts)


def _lifecycle_is_stale_vs_upstream() -> bool:
    """True when signal_lifecycle.csv exists but is older than upstream signals CSVs."""
    lc = file_mtime_dt(SIGNAL_LIFECYCLE_PATH)
    up = _signals_upstream_reference_mtime()
    if lc is None or up is None:
        return False
    return lc < up


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


_fmt_int = _fmt_int_commas


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


def _fmt_float(v: Any, decimals: int = 2) -> str:
    try:
        x = float(v)
    except Exception:
        return "" if v is None else str(v)
    if not np.isfinite(x):
        return ""
    return f"{x:.{decimals}f}"


def format_df_for_display(
    df: pd.DataFrame,
    money_cols: Optional[List[str]] = None,
    pct_cols: Optional[List[str]] = None,
    int_cols: Optional[List[str]] = None,
    float_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    usd_decimals: int = 0,
    pct_decimals: int = 2,
    float_decimals: int = 2,
) -> pd.DataFrame:
    """Return a COPY of df with select columns formatted as strings for readability."""
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    money_cols = money_cols or []
    pct_cols = pct_cols or []
    int_cols = int_cols or []
    float_cols = float_cols or []
    date_cols = date_cols or []

    for c in money_cols:
        if c in out.columns:
            s = out[c]

            # --- FIX: handle duplicate column returning DataFrame ---
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]

            out[c] = pd.to_numeric(s, errors="coerce").map(lambda v: _fmt_usd(v, usd_decimals))

    for c in pct_cols:
        if c in out.columns:
            s = out[c]

            # --- FIX: handle duplicate column returning DataFrame ---
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]

            out[c] = pd.to_numeric(s, errors="coerce").map(lambda v: _fmt_pct(v, pct_decimals))

    for c in int_cols:
        if c in out.columns:
            s = out[c]

            # --- FIX: handle duplicate column returning DataFrame ---
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]

            out[c] = pd.to_numeric(s, errors="coerce").map(_fmt_int)

    for c in float_cols:
        if c in out.columns:
            s = out[c]

            # --- FIX: handle duplicate column returning DataFrame ---
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]

            out[c] = pd.to_numeric(s, errors="coerce").map(lambda v: _fmt_float(v, float_decimals))

    for c in date_cols:
        if c in out.columns:
            s = out[c]

            # --- FIX: handle duplicate column returning DataFrame ---
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]

            out[c] = pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

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
        s = str(row.get("stance") or row.get("STANCE") or "").strip().upper()
    except Exception:
        s = ""

    if s in ("HOLD", "WAIT"):
        # Lighten text a bit across the whole row
        return ["color: rgba(203,213,225,.75);"] * len(row)
    return [""] * len(row)


def _opportunity_type_cell_css(v: Any) -> str:
    """CSS for trade opportunity_type cells (ENTRY / TRIM / EXIT emphasized; other types neutral)."""
    s = str(v).strip().upper()
    bg = "rgba(148,163,184,.18)"
    fg = "#e2e8f0"
    bd = "rgba(148,163,184,.35)"
    if s == "ENTRY":
        bg, fg, bd = "rgba(34,197,94,.28)", "#dcfce7", "rgba(34,197,94,.55)"
    elif s == "TRIM":
        bg, fg, bd = "rgba(245,158,11,.28)", "#ffedd5", "rgba(245,158,11,.55)"
    elif s == "EXIT":
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


def _exploration_flag_true(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    try:
        if v is None or pd.isna(v):
            return False
    except Exception:
        pass
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "t")


def _sort_trade_opportunities_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by confidence descending when numeric values exist; otherwise preserve row order."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "confidence" in out.columns:
        nc = pd.to_numeric(out["confidence"], errors="coerce")
        if nc.notna().any():
            return (
                out.assign(_triton_sort=nc)
                .sort_values("_triton_sort", ascending=False, na_position="last")
                .drop(columns=["_triton_sort"])
            )
    return out


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


def _read_pipeline_heartbeat_file() -> Optional[Dict[str, Any]]:
    """Load only data/results/heartbeat.json (no pipeline_status fallback)."""
    if not HEARTBEAT_PATH.is_file():
        return None
    try:
        raw = HEARTBEAT_PATH.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def render_heartbeat_freshness_banner() -> None:
    """
    Hard freshness banner from heartbeat.json only (60-minute threshold).
    Visible on every page via main() before the top status stack.
    """
    hb = _read_pipeline_heartbeat_file()
    if hb is None:
        st.warning("⚠ No heartbeat found. Freshness unknown.")
        return

    ts_raw = hb.get("timestamp") or hb.get("ts") or hb.get("updated_at") or hb.get("time")
    ts = parse_any_datetime(ts_raw)
    stage = str(hb.get("stage") or hb.get("step") or hb.get("pipeline_stage") or "").strip() or "—"
    status = str(hb.get("status") or "").strip() or "—"

    if ts is None:
        st.warning("⚠ Heartbeat file exists but no usable timestamp. Freshness unknown.")
        return

    now = datetime.now(timezone.utc)
    age_minutes = (now - ts).total_seconds() / 60.0
    if age_minutes > 60 or not np.isfinite(age_minutes):
        st.error("⚠ Data may be stale. Last pipeline heartbeat is older than 60 minutes.")
        return

    ts_disp = dt_to_et_str(ts)
    st.caption(f"Pipeline heartbeat · **{ts_disp}** · stage **{stage}** · status **{status}**")


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

    # Live execution gate (single source of truth)
    # Keep this aligned with the executor via services.live_gate.compute_live_gates.
    def _try_get_broker():
        try:
            from services.broker_alpaca import AlpacaBroker

            return AlpacaBroker(mode="paper")
        except Exception:
            return None

    broker = _try_get_broker()
    live_ready, gates = compute_live_gates(
        project_root=PROJECT_ROOT,
        confirm_ttl_minutes=10,
        freshness_max_age_minutes=180,
        require_market_open=True,
        broker=broker,
    )
    gate_bits = {g.get("name"): bool(g.get("ok")) for g in gates}
    bad_bits = [k for k, v in gate_bits.items() if not v]

    # Keep the legacy execution lock (pipeline/contracts) visible as context
    locked, reason = compute_execution_lock()

    exec_class = "enabled" if live_ready else "locked"
    exec_title = "LIVE READY" if live_ready else "LIVE NOT READY"

    gate_summary = "All gates green" if live_ready else ("Blocked: " + ", ".join(bad_bits))
    if locked:
        exec_detail = f"{gate_summary} | EXEC_LOCK: {reason}"
    else:
        exec_detail = gate_summary

    # compact per-gate badges (ARM / GUARD / CONFIRM)
    def _badge(label, ok):
        return f"{label}:{'OK' if ok else 'NO'}"

    arm_b = _badge("ARM", gate_bits.get("ARM", False))
    guard_b = _badge("GUARD", gate_bits.get("GUARD", False))
    confirm_b = _badge("CONFIRM", gate_bits.get("CONFIRM", False))
    badges = f"{arm_b}  {guard_b}  {confirm_b}"
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
        f"<div class='right'>{exec_detail} · {badges}</div>"
        f"</div></div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

    _mg_snap = load_json(RESULTS_DIR / "master_execution_gate.json", show_error=False)
    if isinstance(_mg_snap, dict) and _mg_snap.get("checked_at"):
        _mg_ok = bool(_mg_snap.get("ok"))
        _mg_sum = str(_mg_snap.get("summary") or "").strip()
        _mg_rs = _mg_snap.get("reasons") or []
        _codes = ", ".join(str(x) for x in _mg_rs) if _mg_rs else ""
        if not _mg_ok:
            st.caption(
                f"Master execution gate: **BLOCKED** — {_mg_sum}"
                + (f" (`{_codes}`)" if _codes else "")
            )
        else:
            st.caption(f"Master execution gate: **READY** — {_mg_sum}")

    _ep = load_json(RESULTS_DIR / "execution_plan.json", show_error=False)
    if isinstance(_ep, dict) and _ep.get("timestamp"):
        _dr = "dry-run" if _ep.get("dry_run") else "execute"
        st.caption(
            f"Last execution plan: {_ep.get('timestamp')} | planned={_ep.get('orders_planned', '—')} | {_dr} | blocked={_ep.get('blocked', False)}"
        )

    _mp = load_json(RESULTS_DIR / "manage_positions_plan.json", show_error=False)
    if isinstance(_mp, dict) and _mp.get("timestamp"):
        _mdr = "dry-run" if _mp.get("dry_run") else "execute"
        st.caption(
            f"Last manage_positions: {_mp.get('timestamp')} | planned={_mp.get('orders_planned', '—')} | {_mdr} | blocked={_mp.get('blocked', False)}"
        )

    _ptc = load_json(RESULTS_DIR / "paper_trade_cycle_summary.json", show_error=False)
    if isinstance(_ptc, dict) and _ptc.get("timestamp"):
        _mex = _ptc.get("config", {}).get("manage_positions_execute", False)
        st.caption(
            f"Last paper trade cycle: {_ptc.get('timestamp')} | ok={_ptc.get('ok')} | blocked={_ptc.get('blocked')} | manage_execute={_mex}"
        )

    _edd = load_json(RESULTS_DIR / "execution_drop_diagnostics.json", show_error=False)
    if isinstance(_edd, dict) and _edd.get("timestamp"):
        _dr = _edd.get("drop_reasons") or {}
        if isinstance(_dr, dict) and _dr:

            def _drop_sort_key(kv: tuple) -> tuple:
                try:
                    return (-float(kv[1]), str(kv[0]))
                except Exception:
                    return (0.0, str(kv[0]))

            _top = ", ".join(f"{k}:{v}" for k, v in sorted(_dr.items(), key=_drop_sort_key)[:4])
        else:
            _top = "—"
        st.caption(
            f"Execution drop diagnostics: {_edd.get('timestamp')} | dropped={_edd.get('dropped_orders', '—')} | "
            f"submitted={_edd.get('submitted_orders', '—')} | top reasons: {_top}"
        )


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


def _positions_trace_snapshot(df: pd.DataFrame, label: str) -> None:
    """Debug-only: trace duplicate columns through positions pipeline (no logic change)."""
    schema_snapshot(df, label=label)
    dupes = find_duplicate_columns(df)
    if dupes:
        print(f"[🚨 DUPLICATES DETECTED] {label}: {dupes}")


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

    _positions_trace_snapshot(df, "positions.AFTER_CSV_LOAD")

    df = sanitize_df_cols(df)
    df = ensure_date_col(df)
    _positions_trace_snapshot(df, "positions.AFTER_SANITIZE_AND_DATE_COL")

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
    _positions_trace_snapshot(df2, "positions.AFTER_DF2_DROPNA")
    if df2.empty:
        st.warning("Positions snapshot has no valid (date,ticker,value) rows.")
        st.dataframe(df.head(80), use_container_width=True)
        return

    latest_dt = df2["date"].max()
    snap = df2[df2["date"] == latest_dt].copy()
    _positions_trace_snapshot(snap, "positions.AFTER_SNAP_LATEST_DATE_PRE_DEDUPE")
    snap = snap.loc[:, ~snap.columns.duplicated()].copy()
    _positions_trace_snapshot(snap, "positions.AFTER_SNAP_DEDUPE")

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
    _positions_trace_snapshot(snap_view, "positions.AFTER_SNAP_VIEW_SORT_PRE_DEDUPE")
    snap_view = snap_view.loc[:, ~snap_view.columns.duplicated()].copy()
    _positions_trace_snapshot(snap_view, "positions.AFTER_SNAP_VIEW_DEDUPE")

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
    # No merge() in this path; rename to "value" can create duplicate "value" if CSV had both value_col and "value".
    _positions_trace_snapshot(snap_tbl, "positions.AFTER_RENAME_TO_VALUE_PRE_SCHEMA_GUARD")
    show_cols = [
        c
        for c in ["ticker", "value"] + [c for c in cols if c not in ("ticker", value_col)]
        if c in snap_tbl.columns
    ]

    snap_tbl = dedupe_columns(snap_tbl, warn_label="positions_exposure.snap_tbl")
    schema_snapshot(snap_tbl, label="positions_exposure.snap_tbl")

    # --- FIX 1: remove duplicate columns ---
    snap_tbl = snap_tbl.loc[:, ~snap_tbl.columns.duplicated()].copy()

    # --- FIX 2: safe + unique column selection ---
    safe_cols = list(dict.fromkeys([c for c in show_cols if c in snap_tbl.columns]))

    require_columns(
        snap_tbl,
        safe_cols,
        label="positions_exposure.snap_tbl",
        hard_fail=False,
    )

    snap_disp = format_df_for_display(
        snap_tbl[safe_cols],
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

    if "stance" in view.columns:
        view["stance"] = view["stance"].astype(str).str.upper()

    # Column order: lifecycle truth first, then signal context, then remaining columns unchanged
    primary_cols = ["ticker", "stance", "lifecycle_action", "position_state", "last_action"]
    secondary_cols = ["signal", "confidence", "delta_pct", "rationale"]
    ordered = [c for c in primary_cols if c in view.columns]
    ordered += [c for c in secondary_cols if c in view.columns]
    ordered += [c for c in view.columns if c not in ordered]
    view_disp = view[ordered].copy()

    rename_disp = {
        "stance": "STANCE",
        "lifecycle_action": "ACTION",
        "position_state": "POSITION",
        "last_action": "LAST",
    }
    view_disp = view_disp.rename(
        columns={k: v for k, v in rename_disp.items() if k in view_disp.columns}
    )

    # Visual authority: STANCE badges (bold + color) + dim HOLD/WAIT rows (display-only)
    if "STANCE" in view_disp.columns:
        sty = view_disp.style.apply(_dim_hold_rows, axis=1)
        sty = sty.applymap(_stance_cell_css, subset=["STANCE"])
        st.dataframe(sty, use_container_width=True, hide_index=True)
    else:
        st.dataframe(view_disp, use_container_width=True, hide_index=True)


def page_trade_opportunities() -> None:
    st.markdown("### 🚀 Trade Opportunities")
    st.caption(
        "Execution-ready rows from **trade_opportunities.csv** — same file **place_live_orders** uses first when present."
    )

    if not TRADE_OPPORTUNITIES_PATH.exists():
        st.info("No trade opportunities available")
        return

    df = load_csv(TRADE_OPPORTUNITIES_PATH, show_error=False)
    if df is None:
        st.info("No trade opportunities available")
        return

    df = sanitize_df_cols(df.copy())
    if df.empty:
        st.info("System idle — no actionable opportunities")
        return

    ot = (
        df["opportunity_type"].astype(str).str.strip().str.upper()
        if "opportunity_type" in df.columns
        else pd.Series([""] * len(df))
    )
    n_total = len(df)
    n_entry = int((ot == "ENTRY").sum()) if "opportunity_type" in df.columns else 0
    n_trim = int((ot == "TRIM").sum()) if "opportunity_type" in df.columns else 0
    n_exit = int((ot == "EXIT").sum()) if "opportunity_type" in df.columns else 0
    has_expl = "exploration_flag" in df.columns
    n_expl = int(df["exploration_flag"].map(_exploration_flag_true).sum()) if has_expl else 0
    n_strict = int((~df["exploration_flag"].map(_exploration_flag_true)).sum()) if has_expl else 0

    metric_cols = st.columns(6 if has_expl else 4)
    with metric_cols[0]:
        st.metric("Total opportunities", n_total)
    with metric_cols[1]:
        st.metric("ENTRY", n_entry)
    with metric_cols[2]:
        st.metric("EXIT", n_exit)
    with metric_cols[3]:
        st.metric("TRIM", n_trim)
    if has_expl:
        with metric_cols[4]:
            st.metric("Exploratory", n_expl)
        with metric_cols[5]:
            st.metric("Strict", n_strict)

    view = _sort_trade_opportunities_df(df)

    display_cols = [
        "ticker",
        "opportunity_type",
        "effective_stance",
        "effective_position_state",
        "confidence",
        "delta_pct",
        "healed",
        "exploration_flag",
    ]
    ordered = [c for c in display_cols if c in view.columns]
    if not ordered:
        st.warning("trade_opportunities.csv has no recognizable columns for this view.")
        st.dataframe(view.head(50), use_container_width=True, hide_index=True)
        return

    view_disp = view[ordered].copy()

    if "exploration_flag" in view_disp.columns:
        view_disp["exploration_flag"] = view_disp["exploration_flag"].map(
            lambda x: "⚠ Exploratory" if _exploration_flag_true(x) else "Strict"
        )

    if "opportunity_type" in view_disp.columns:
        sty = view_disp.style.applymap(_opportunity_type_cell_css, subset=["opportunity_type"])
        st.dataframe(sty, use_container_width=True, hide_index=True)
    else:
        st.dataframe(view_disp, use_container_width=True, hide_index=True)


def _reconciliation_row_style(row: pd.Series) -> List[str]:
    """Red-tint full row when mismatch is true (CSV may store bool or string)."""
    mis = row.get("mismatch")
    if isinstance(mis, (bool, np.bool_)):
        is_m = bool(mis)
    else:
        s = str(mis).strip().lower()
        is_m = s in ("true", "1", "yes", "t")
    if is_m:
        return ["background-color: rgba(220,38,38,0.28);"] * len(row)
    return [""] * len(row)


def page_reconciliation() -> None:
    st.markdown("### 🔍 Reconciliation")
    st.caption(
        "Lifecycle STATE vs broker positions snapshot (from **lifecycle_reconciliation.csv**). Run `python -m services.reconcile_lifecycle_vs_positions` to refresh."
    )

    df = load_csv(LIFECYCLE_RECONCILIATION_PATH, show_error=False)
    if df is None or df.empty:
        st.info(
            "No reconciliation data yet. Generate **data/results/lifecycle_reconciliation.csv** with `python -m services.reconcile_lifecycle_vs_positions`."
        )
        return

    df = sanitize_df_cols(df.copy())

    cols = [
        "ticker",
        "lifecycle_stance",
        "lifecycle_position_state",
        "broker_qty",
        "reconciled_state",
        "mismatch",
        "mismatch_reason",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"lifecycle_reconciliation.csv missing columns: {missing}")
        st.dataframe(df.head(50), use_container_width=True)
        return

    view = df[cols].copy()

    mis_mask = view["mismatch"]
    if mis_mask.dtype != bool:
        mis_mask = mis_mask.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "t"))
    n_total = len(view)
    n_mis = int(mis_mask.sum())
    n_ok = n_total - n_mis
    match_pct = (100.0 * n_ok / n_total) if n_total else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total rows", n_total)
    with c2:
        st.metric("Mismatch count", n_mis)
    with c3:
        st.metric("Match %", f"{match_pct:.1f}%")

    mismatches_only = st.checkbox("Show mismatches only", value=False, key="recon_mismatches_only")
    if mismatches_only:
        view = view[mis_mask].copy()

    sty = view.style.apply(_reconciliation_row_style, axis=1)
    st.dataframe(sty, use_container_width=True, hide_index=True)


def page_ai_signals() -> None:
    st.markdown("### 🤖 AI Signals")
    st.caption(
        "Priority order: **Signal Lifecycle (STATE) → Signals w/ Rationale → Raw Signals**. Lifecycle rows represent Triton’s current stance per ticker."
    )

    stale = _lifecycle_is_stale_vs_upstream()
    df: Optional[pd.DataFrame] = None

    if not stale:
        if SIGNAL_LIFECYCLE_PATH.exists() and SIGNAL_LIFECYCLE_PATH.stat().st_size > 0:
            df = load_csv(SIGNAL_LIFECYCLE_PATH, show_error=True)
            if df is None or df.empty:
                df = None

    if df is None or df.empty:
        if stale:
            st.warning(
                "signal_lifecycle.csv is older than signals_with_rationale.csv / signals.csv. "
                "Showing rationale or raw signals until lifecycle is regenerated."
            )
        df = load_first_nonempty_csv([SIGNALS_RATIONALE_PATH, SIGNALS_PATH])

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


def _truncate_preview(s: Any, limit: int = 220) -> str:
    try:
        t = " ".join(str(s).split())
    except Exception:
        t = str(s)
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def page_execution_dashboard() -> None:
    """Read-only execution observability: paper cycle, entry, management, drops, lifecycle, snapshots."""
    st.markdown("### ⚙️ Execution Dashboard")
    st.caption(
        "Read-only. Summarizes scheduled paper cycle, execute_trades, manage_positions, drop diagnostics, "
        "and broker snapshot artifacts. No order placement or cancellation from this page."
    )

    st.subheader("ARM Mode Status")
    arm_st = load_json(RESULTS_DIR / "arm_mode_status.json", show_error=False)
    arm_cf = load_json(PROJECT_ROOT / "config" / "arm_mode.json", show_error=False)
    if isinstance(arm_st, dict) and arm_st.get("timestamp"):
        a0, a1, a2 = st.columns(3)
        a0.metric("Mode", str(arm_st.get("mode", "—")))
        a1.metric(
            "Paper auto (config)", "yes" if (arm_cf or {}).get("paper_auto_allowed") else "no"
        )
        a2.metric("Live auto (config)", "yes" if (arm_cf or {}).get("live_auto_allowed") else "no")
        perms = arm_st.get("permissions") if isinstance(arm_st.get("permissions"), dict) else {}
        st.caption(
            "**Execute permissions:** "
            + ", ".join(f"{k}={perms.get(k)}" for k in sorted(perms.keys()))
        )
        br = arm_st.get("block_reasons")
        if isinstance(br, list) and br:
            st.warning("Block reasons: " + ", ".join(str(x) for x in br))
        st.caption(f"Last snapshot: **{arm_st.get('timestamp')}**")
        if str((arm_cf or {}).get("mode", "")).upper() == "ASSISTED":
            pconf = load_json(
                PROJECT_ROOT / "data" / "live" / "paper_arm_confirm.json", show_error=False
            )
            if isinstance(pconf, dict) and pconf.get("expires_at"):
                st.caption(
                    f"ASSISTED **paper_arm_confirm.json**: allow_execute={pconf.get('allow_execute')} · expires **{pconf.get('expires_at')}**"
                )
            else:
                st.caption(
                    "ASSISTED mode: no valid **paper_arm_confirm.json** (mutations stay off until confirmed)."
                )
    else:
        st.info(
            "No **arm_mode_status.json** yet. Run `python -m services.run_scheduled_paper_cycle` to refresh."
        )

    # ── Section 1 — Paper cycle ─────────────────────────────────────────
    st.subheader("1 — Paper cycle status")
    ptc = load_json(PAPER_TRADE_CYCLE_SUMMARY_PATH, show_error=False)
    if not isinstance(ptc, dict) or not ptc.get("timestamp"):
        st.info(
            "No **paper_trade_cycle_summary.json** yet (or empty). Run `python -m services.run_scheduled_paper_cycle` to generate."
        )
    else:
        cfg = ptc.get("config") if isinstance(ptc.get("config"), dict) else {}
        mex = cfg.get("manage_positions_execute", "—")
        ok = bool(ptc.get("ok", False))
        blk = bool(ptc.get("blocked", False))
        warn = bool(ptc.get("had_warnings", False))
        c0, c1, c2, c3, c4 = st.columns(5)
        c0.metric("Last cycle (UTC)", str(ptc.get("timestamp") or "—"))
        c1.metric("Overall ok", "✅" if ok else "❌")
        c2.metric("Blocked", "🟡 yes" if blk else "🟢 no")
        c3.metric("Warnings", "⚠️ yes" if warn else "✓ none")
        c4.metric("manage_execute", str(mex))
        notes = ptc.get("cycle_notes")
        if isinstance(notes, list) and notes:
            st.caption("**Cycle notes:** " + " · ".join(str(x) for x in notes))
        elif isinstance(notes, str) and notes.strip():
            st.caption("**Cycle notes:** " + notes.strip())
        if ok and blk:
            st.info(
                "**Note:** `ok=true` with `blocked=true` often means the cycle finished safely but execution was "
                "blocked or no orders were placed (e.g. gate, duplicates, empty batch). This is not necessarily a system failure."
            )
        stages = ptc.get("stages")
        if isinstance(stages, dict) and stages:
            st.markdown("**Stages**")
            snames = [
                "snapshot_start",
                "pipeline",
                "execute_trades",
                "manage_positions",
                "snapshot_before_maintenance",
                "manage_open_orders",
                "reprice_order_ladder",
                "snapshot_refresh",
            ]
            cols = st.columns(len(snames))
            for i, sn in enumerate(snames):
                with cols[i]:
                    st.markdown(f"**{sn}**")
                    sd = stages.get(sn)
                    if not isinstance(sd, dict):
                        st.caption("—")
                        continue
                    s_ok = bool(sd.get("ok", False))
                    s_blk = bool(sd.get("blocked", False))
                    st.caption("✅ ok" if s_ok else "❌ fail")
                    st.caption("🟡 blocked" if s_blk else "")
                    st.caption(
                        f"exit **{sd.get('exit_code', '—')}** · {float(sd.get('duration_sec') or 0):.2f}s"
                    )
                    msg = _truncate_preview(sd.get("message") or "", 180)
                    if msg:
                        st.caption(msg)
        st.divider()

    # ── Section 2 — Entry execution ─────────────────────────────────────
    st.subheader("2 — Entry execution (execute_trades)")
    epj = load_json(EXECUTION_PLAN_JSON_PATH, show_error=False)
    if not isinstance(epj, dict) or not epj.get("timestamp"):
        st.info("No **execution_plan.json** yet.")
    else:
        i1, i2, i3 = st.columns(3)
        i1.metric("Mode", str(epj.get("mode", "—")))
        i2.metric("Dry run", "yes" if epj.get("dry_run") else "no")
        i3.metric("Blocked", "yes" if epj.get("blocked") else "no")
        st.caption(
            f"**Timestamp:** {epj.get('timestamp')} · **opportunities_seen:** {epj.get('opportunities_seen', '—')} · "
            f"**planned:** {epj.get('orders_planned', '—')} · **executed (intent):** {epj.get('orders_executed', '—')} · "
            f"**skipped:** {epj.get('orders_skipped', '—')}"
        )
        br = epj.get("block_reasons")
        if isinstance(br, list) and br:
            st.caption("**block_reasons:** " + "; ".join(str(x) for x in br))
        wr = epj.get("warnings")
        if isinstance(wr, list) and wr:
            st.warning("warnings: " + "; ".join(str(x) for x in wr))
    ep_csv = load_csv(EXECUTION_PLAN_CSV_PATH, show_error=False)
    if ep_csv is not None and not ep_csv.empty:
        want = ["symbol", "stance", "status", "skip_reason", "qty", "side", "planned_notional"]
        show = [c for c in want if c in ep_csv.columns]
        if show:
            st.dataframe(
                sanitize_df_cols(ep_csv[show].head(200)), use_container_width=True, hide_index=True
            )
        else:
            st.dataframe(
                sanitize_df_cols(ep_csv.head(100)), use_container_width=True, hide_index=True
            )
    elif epj:
        st.caption("No **execution_plan.csv** rows (optional).")
    st.divider()

    # ── Section 3 — Position management ─────────────────────────────────
    st.subheader("3 — Position management")
    mpj = load_json(MANAGE_POSITIONS_PLAN_JSON_PATH, show_error=False)
    if not isinstance(mpj, dict) or not mpj.get("timestamp"):
        st.info("No **manage_positions_plan.json** yet.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mode", str(mpj.get("mode", "—")))
        m2.metric("Dry run", "yes" if mpj.get("dry_run") else "no")
        m3.metric("Planned orders", str(mpj.get("orders_planned", "—")))
        m4.metric("Blocked", "yes" if mpj.get("blocked") else "no")
        st.caption(
            f"**Timestamp:** {mpj.get('timestamp')} · **symbols_seen:** {mpj.get('symbols_seen', '—')} · "
            f"**positions_seen:** {mpj.get('positions_seen', '—')} · **executed:** {mpj.get('orders_executed', '—')} · "
            f"**skipped:** {mpj.get('orders_skipped', '—')}"
        )
        mbr = mpj.get("block_reasons")
        if isinstance(mbr, list) and mbr:
            st.caption("**block_reasons:** " + "; ".join(str(x) for x in mbr))
        sk = mpj.get("skip_reasons")
        if isinstance(sk, dict) and sk:
            st.caption("**skip_reasons:** " + ", ".join(f"{k}={v}" for k, v in sk.items()))
        if int(mpj.get("orders_planned") or 0) == 0 and isinstance(sk, dict) and sk:
            top = max(sk.items(), key=lambda kv: kv[1])[0] if sk else ""
            if "NO_ACTION" in str(top).upper() or "HOLD" in str(top).upper():
                st.caption(
                    "**Interpretation:** Management engine often has no EXIT/TRIM rows when lifecycle says HOLD or "
                    "risk rules yield **NO_ACTION_HOLD** — plan-only runs are expected."
                )
        st.markdown("**Active Position Management** (read-only; Phase 1 intelligence)")
        st.caption(
            "Profit-aware TRIM, signal/stale EXIT, and diagnostics from `manage_positions.py`. "
            "Default remains dry-run unless `--execute`."
        )
        ap1, ap2, ap3, ap4 = st.columns(4)
        ap1.metric("Trim candidates", str(mpj.get("trim_candidates", "—")))
        ap2.metric("Exit candidates", str(mpj.get("exit_candidates", "—")))
        ap3.metric(
            "Approved actions", str(mpj.get("approved_actions", mpj.get("orders_planned", "—")))
        )
        if isinstance(sk, dict) and sk:
            top_rc = sorted(
                sk.items(), key=lambda kv: (-int(kv[1]) if str(kv[1]).isdigit() else 0, str(kv[0]))
            )[:8]
            ap4.metric(
                "Top skip reasons", ", ".join(f"{k}:{v}" for k, v in top_rc[:3]) if top_rc else "—"
            )
        else:
            ap4.metric("Top skip reasons", "—")
    mp_csv = load_csv(MANAGE_POSITIONS_PLAN_CSV_PATH, show_error=False)
    if mp_csv is not None and not mp_csv.empty:
        want_m = [
            "symbol",
            "stance",
            "status",
            "skip_reason",
            "qty",
            "side",
            "management_action",
            "planned_notional",
            "reason_code",
            "profit_pct",
            "lifecycle_stance",
            "effective_stance",
            "final_action",
        ]
        show_m = [c for c in want_m if c in mp_csv.columns]
        if show_m:
            st.dataframe(
                sanitize_df_cols(mp_csv[show_m].head(200)),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                sanitize_df_cols(mp_csv.head(100)), use_container_width=True, hide_index=True
            )
        cand = mp_csv.copy()
        if "reason_code" in cand.columns:
            try:
                sub = cand[
                    cand["reason_code"].astype(str).str.upper().str.contains("APPROVED", na=False)
                ]
                if not sub.empty:
                    st.markdown("**Approved / candidate rows (reason_code)**")
                    show_c = [c for c in want_m if c in sub.columns]
                    st.dataframe(
                        sanitize_df_cols(sub[show_c].head(50) if show_c else sub.head(50)),
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception:
                pass
    elif mpj:
        st.caption("No **manage_positions_plan.csv** rows (optional).")

    st.markdown("**Capital reallocation** (read-only; `services.capital_reallocation`)")
    st.caption(
        "Exit → entry bridge: estimated freed capital from manage exits/trims, ranked BUY opportunities, "
        "`reallocation_plan.csv`. Optional `--reallocate-after-exit` runs `execute_trades` on paper after successful exits."
    )
    cr = load_json(CAPITAL_REALLOCATION_JSON_PATH, show_error=False)
    if isinstance(cr, dict) and cr.get("timestamp"):
        cr1, cr2, cr3, cr4 = st.columns(4)
        cr1.metric("Freed capital ($)", str(cr.get("freed_capital", "—")))
        cr2.metric("Method", str(cr.get("freed_capital_method", "—")))
        cr3.metric(
            "Eligible / filtered out",
            f"{cr.get('eligible_candidates', '—')} / {cr.get('filtered_out', '—')}",
        )
        cr4.metric(
            "Selected",
            len(cr["selected_symbols"]) if isinstance(cr.get("selected_symbols"), list) else "—",
        )
        st.caption(
            f"**Session:** `{cr.get('source_session') or '—'}` · **n_exits:** {cr.get('n_exits', '—')} · "
            f"**n_trims:** {cr.get('n_trims', '—')} · **timestamp:** {cr.get('timestamp', '—')}"
        )
        st.caption(
            f"**Portfolio (pre-plan):** {cr.get('current_positions_count', '—')} positions · "
            f"exposure ~{cr.get('current_total_exposure', '—')} · **max_positions:** {cr.get('max_positions', '—')}"
        )
        sy = cr.get("selected_symbols")
        if isinstance(sy, list) and sy:
            st.caption("**Selected symbols:** " + ", ".join(str(x) for x in sy[:30]))
        prev = cr.get("preview_top_buy_opportunities")
        if isinstance(prev, list) and prev:
            st.caption(
                "**Top BUY previews (ranked):** "
                + ", ".join(str(p.get("symbol")) for p in prev[:10] if isinstance(p, dict))
            )
    else:
        st.info(
            "No **capital_reallocation.json** yet (run `manage_positions` dry-run or `--execute`)."
        )
    rp_csv = load_csv(REALLOCATION_PLAN_CSV_PATH, show_error=False)
    if rp_csv is not None and not rp_csv.empty:
        want_rp = [
            "symbol",
            "recommended_action",
            "priority_rank",
            "confidence",
            "delta_pct",
            "estimated_notional",
            "size_factor",
            "volatility_used",
            "vol_adjustment",
            "size_factor_final",
            "correlation_score",
            "correlation_penalty",
            "adjusted_notional",
            "normalized_notional",
            "portfolio_weight",
            "regime_label",
            "regime_exposure_multiplier",
            "allocation_fraction",
            "eligible",
            "exclusion_reason",
            "selected",
        ]
        show_rp = [c for c in want_rp if c in rp_csv.columns]
        st.dataframe(
            sanitize_df_cols(rp_csv[show_rp] if show_rp else rp_csv.head(50)),
            use_container_width=True,
            hide_index=True,
        )
    elif isinstance(cr, dict) and cr.get("timestamp"):
        st.caption("No rows in **reallocation_plan.csv** (below min freed capital or no BUY opps).")

    st.divider()

    # ── Section 4 — Drop diagnostics ───────────────────────────────────
    st.subheader("4 — Execution drop diagnostics")
    edj = load_json(EXEC_DROP_JSON_PATH, show_error=False)
    if not isinstance(edj, dict) or not edj.get("timestamp"):
        st.info("No **execution_drop_diagnostics.json** yet.")
    else:
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("Planned (rows)", str(edj.get("planned_orders", "—")))
        d2.metric("Submitted", str(edj.get("submitted_orders", "—")))
        d3.metric("In-flight (satisfied)", str(edj.get("in_flight_orders", "—")))
        d4.metric("Dropped (rows)", str(edj.get("dropped_orders", "—")))
        d5.metric("Blocked flag", "yes" if edj.get("blocked") else "no")
        d6.metric("Source", str(edj.get("source", "—")))
        st.caption(f"**Timestamp:** {edj.get('timestamp')}")
        dr = edj.get("drop_reasons")
        if isinstance(dr, dict) and dr:
            st.markdown("**Top drop reasons**")

            def _dr_sort_key(kv: Tuple[Any, Any]) -> Tuple[float, str]:
                try:
                    return (-float(kv[1]), str(kv[0]))
                except Exception:
                    return (0.0, str(kv[0]))

            top_items = sorted(dr.items(), key=_dr_sort_key)[:12]
            st.code(", ".join(f"{k}: {v}" for k, v in top_items))
        pj = int(edj.get("planned_orders") or 0)
        sj = int(edj.get("submitted_orders") or 0)
        infj = int(edj.get("in_flight_orders") or 0)
        drj = int(edj.get("dropped_orders") or 0)
        if pj > 0 and sj == 0 and not edj.get("blocked"):
            if infj > 0 and drj == 0:
                st.caption(
                    "**Interpretation:** Nothing new was submitted because each intent already has a matching **open order** "
                    "(in-flight). This is expected when orders are already working at the broker."
                )
            else:
                st.caption(
                    "**Interpretation:** Planned activity was recorded but nothing new was submitted — see dropped rows below "
                    "(often illegal sells, limits, batch limits, or validation)."
                )
    edf = load_csv(EXEC_DROP_CSV_PATH, show_error=False)
    if edf is not None and not edf.empty:
        edf = sanitize_df_cols(edf)
        fcols = ["phase", "status", "reason_code", "symbol"]
        avail = [c for c in fcols if c in edf.columns]
        if avail:
            cfa, cfb, cfc, cfd = st.columns(4)
            phases = ["(all)"]
            if "phase" in edf.columns:
                phases.extend(sorted({str(x) for x in edf["phase"].dropna().unique()}))
            sts = ["(all)"]
            if "status" in edf.columns:
                sts.extend(sorted({str(x) for x in edf["status"].dropna().unique()}))
            rcs = ["(all)"]
            if "reason_code" in edf.columns:
                rcs.extend(sorted({str(x) for x in edf["reason_code"].dropna().unique()}))
            with cfa:
                pf = st.selectbox("Phase", phases, key="exec_dash_phase")
            with cfb:
                sf = st.selectbox("Status", sts, key="exec_dash_status")
            with cfc:
                rf = st.selectbox("Reason code", rcs, key="exec_dash_reason")
            with cfd:
                sym_q = st.text_input("Symbol contains", "", key="exec_dash_sym").strip().upper()
            filt = edf.copy()
            if pf != "(all)" and "phase" in filt.columns:
                filt = filt[filt["phase"].astype(str) == pf]
            if sf != "(all)" and "status" in filt.columns:
                filt = filt[filt["status"].astype(str) == sf]
            if rf != "(all)" and "reason_code" in filt.columns:
                filt = filt[filt["reason_code"].astype(str) == rf]
            if sym_q and "symbol" in filt.columns:
                filt = filt[filt["symbol"].astype(str).str.upper().str.contains(sym_q, na=False)]
            want_d = [
                "symbol",
                "stance",
                "phase",
                "status",
                "reason_code",
                "reason_detail",
                "planned_qty",
                "planned_notional",
                "source",
                "session",
            ]
            show_d = [c for c in want_d if c in filt.columns]
            st.dataframe(
                filt[show_d] if show_d else filt, use_container_width=True, hide_index=True
            )
        else:
            st.dataframe(edf.head(300), use_container_width=True, hide_index=True)
    elif edj:
        st.caption("No **execution_drop_diagnostics.csv** (optional detail file).")
    st.divider()

    # ── Section 4b — Signal pressure diagnostics (read-only funnel) ─────
    st.subheader("4b — Signal pressure diagnostics")
    st.caption(
        "Read-only funnel snapshot: raw signals → lifecycle replay → effective stance vs trade opportunities. "
        "Written by `services/signal_pressure_diagnostics.py` after signals / lifecycle / opportunities steps."
    )
    spd = load_json(SIGNAL_PRESSURE_DIAG_PATH, show_error=False)
    if not isinstance(spd, dict) or not spd.get("timestamp"):
        st.info(
            "No **signal_pressure_diagnostics.json** yet. Run signal generation, lifecycle, or `build_trade_opportunities` to refresh."
        )
    else:
        sp1, sp2, sp3, sp4 = st.columns(4)
        sp1.metric("Tickers", str(spd.get("ticker_count", "—")))
        rsc = spd.get("raw_signal_counts") if isinstance(spd.get("raw_signal_counts"), dict) else {}
        sp2.metric("Raw BUY-like", str(rsc.get("buy_like", "—")))
        sp3.metric("Raw SELL-like", str(rsc.get("sell_like", "—")))
        sp4.metric("Raw neutral-like", str(rsc.get("neutral_like", "—")))
        st.caption(f"**Timestamp:** {spd.get('timestamp')}")
        fc = spd.get("final_counts") if isinstance(spd.get("final_counts"), dict) else {}
        if fc:
            f1, f2, f3, f4, f5, f6, f7 = st.columns(7)
            f1.metric("buy", str(fc.get("buy", "—")))
            f2.metric("add", str(fc.get("add", "—")))
            f3.metric("trim", str(fc.get("trim", "—")))
            f4.metric("exit", str(fc.get("exit", "—")))
            f5.metric("hold", str(fc.get("hold", "—")))
            f6.metric("wait", str(fc.get("wait", "—")))
            f7.metric("actionable", str(fc.get("actionable_opportunities", "—")))
        filt = spd.get("filters") if isinstance(spd.get("filters"), dict) else {}
        if filt:
            st.markdown("**Lifecycle / filter counts (replay)**")
            top_f = sorted(
                filt.items(),
                key=lambda kv: (-int(kv[1]) if str(kv[1]).isdigit() else 0, str(kv[0])),
            )[:10]
            st.code(", ".join(f"{k}: {v}" for k, v in top_f))
        srt = spd.get("suppression_reasons_top")
        if isinstance(srt, list) and srt:
            st.markdown("**Top suppression reasons (per-ticker)**")
            st.code(
                ", ".join(
                    f"{x.get('reason', '')}: {x.get('count', '')}"
                    for x in srt[:12]
                    if isinstance(x, dict)
                )
            )
        nts = spd.get("notes")
        if isinstance(nts, list) and nts:
            st.caption("**Notes:** " + " · ".join(str(x) for x in nts[:6]))
    spd_csv = load_csv(SIGNAL_PRESSURE_DIAG_CSV_PATH, show_error=False)
    if spd_csv is not None and not spd_csv.empty:
        spd_csv = sanitize_df_cols(spd_csv)
        sub = spd_csv.copy()
        if "final_actionable" in sub.columns:
            try:
                sub = sub[
                    ~sub["final_actionable"].astype(str).str.lower().isin(("true", "1", "yes"))
                ]
            except Exception:
                pass
        elif "suppression_reason" in sub.columns:
            sub = sub[sub["suppression_reason"].astype(str).str.upper() != "ACTIONABLE"]
        want_sp = [
            "ticker",
            "raw_signal",
            "delta_pct",
            "lifecycle_stance",
            "effective_stance",
            "final_actionable",
            "suppression_reason",
        ]
        show_sp = [c for c in want_sp if c in sub.columns]
        st.markdown("**Suppressed / non-actionable tickers (sample)**")
        st.dataframe(
            sub[show_sp].head(25) if show_sp else sub.head(25),
            use_container_width=True,
            hide_index=True,
        )
    elif isinstance(spd, dict) and spd.get("timestamp"):
        st.caption("No **signal_pressure_diagnostics.csv** rows (optional per-ticker file).")
    st.divider()

    # ── Section 4c — Execution pressure (intent vs execution) ────────────
    st.subheader("4c — Execution pressure")
    st.caption(
        "Read-only: effective lifecycle intent vs trade opportunities vs planned orders vs fills. "
        "Written by `services/execution_pressure_diagnostics.py` (e.g. after scheduled paper cycle)."
    )
    ep = load_json(EXECUTION_PRESSURE_PATH, show_error=False)
    if not isinstance(ep, dict) or not ep.get("timestamp"):
        st.info(
            "No **execution_pressure.json** yet. Run the paper cycle or `python -m services.execution_pressure_diagnostics`."
        )
    else:
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Lifecycle actionable (effective)", str(ep.get("lifecycle_actionable", "—")))
        e2.metric("Lifecycle intent rows", str(ep.get("lifecycle_intent_actionable", "—")))
        e3.metric("Opportunities", str(ep.get("opportunities_created", "—")))
        e4.metric("Orders planned", str(ep.get("orders_planned", "—")))
        e5.metric("Orders executed (log)", str(ep.get("orders_executed", "—")))
        st.caption(
            f"**Timestamp:** {ep.get('timestamp')} · **blocked (drop diag):** {ep.get('blocked_orders', '—')}"
        )
        do = ep.get("drop_off") if isinstance(ep.get("drop_off"), dict) else {}
        if do:
            st.markdown("**Drop-off (stage gaps)**")
            st.code(
                "lifecycle→opportunity: "
                f"{do.get('lifecycle_to_opportunity', '—')} · "
                "opportunity→orders: "
                f"{do.get('opportunity_to_orders', '—')} · "
                "orders→execution: "
                f"{do.get('orders_to_execution', '—')}"
            )
        br = ep.get("block_reasons") if isinstance(ep.get("block_reasons"), dict) else {}
        if br:
            st.markdown("**Top block reason buckets** (from execution drop diagnostics)")
            top_b = sorted(
                br.items(), key=lambda kv: (-int(kv[1]) if str(kv[1]).isdigit() else 0, str(kv[0]))
            )[:12]
            st.code(", ".join(f"{k}: {v}" for k, v in top_b))
        n_ep = ep.get("notes")
        if isinstance(n_ep, list) and n_ep:
            st.caption("**Notes:** " + " · ".join(str(x) for x in n_ep[:5]))
    sf = load_json(SESSION_FILL_PRESSURE_PATH, show_error=False)
    if isinstance(sf, dict) and sf.get("timestamp"):
        st.markdown("**Session fill (current `execute_trades` session, log-scoped)**")
        st.caption(
            "From `session_fill_pressure.json` — counts only `live_orders_log.csv` rows matching "
            "`last_execution_session.json` session. Not mixed with historical fills."
        )
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Planned", str(sf.get("orders_planned", "—")))
        s2.metric("Submitted", str(sf.get("orders_submitted", "—")))
        s3.metric("Filled", str(sf.get("orders_filled", "—")))
        s4.metric("Open / in-flight", str(sf.get("orders_open", "—")))
        s5.metric("Canceled", str(sf.get("orders_canceled", "—")))
        s6.metric("Fill rate", str(sf.get("fill_rate", "—")))
        st.caption(
            f"**Session:** `{sf.get('session', '—')}` · **mode:** {sf.get('mode', '—')} · "
            f"**rejected:** {sf.get('orders_rejected', '—')}"
        )
        n_sf = sf.get("notes")
        if isinstance(n_sf, list) and n_sf:
            st.caption("**Session fill notes:** " + " · ".join(str(x) for x in n_sf[:4]))
    oop = load_json(OPEN_ORDER_PRESSURE_PATH, show_error=False)
    if isinstance(oop, dict) and oop.get("timestamp"):
        st.markdown(
            "**Open order pressure** (read-only; run `python -m services.manage_open_orders --mode paper`)"
        )
        o1, o2, o3, o4, o5, o6 = st.columns(6)
        o1.metric("Open total", str(oop.get("open_orders_total", "—")))
        o2.metric("Stale", str(oop.get("stale_orders_total", "—")))
        o3.metric("Fresh", str(oop.get("fresh_orders_total", "—")))
        o4.metric("Oldest (min)", str(oop.get("oldest_open_order_minutes", "—")))
        o5.metric("Stale threshold (m)", str(oop.get("stale_minutes", "—")))
        o6.metric("Dry-run last run", "yes" if oop.get("dry_run") else "no")
        blk = oop.get("symbols_blocking_execution")
        if isinstance(blk, list) and blk:
            st.caption("**Symbols with stale open orders:** " + ", ".join(str(x) for x in blk[:40]))
        st_caption = oop.get("notes")
        if isinstance(st_caption, list) and st_caption:
            st.caption("**Notes:** " + " · ".join(str(x) for x in st_caption[:4]))
        so = load_csv(STALE_OPEN_ORDERS_CSV_PATH, show_error=False)
        if so is not None and not so.empty:
            want_so = [
                c
                for c in (
                    "symbol",
                    "side",
                    "status",
                    "age_minutes",
                    "limit_price",
                    "order_id",
                    "stale_reason",
                )
                if c in so.columns
            ]
            st.dataframe(
                sanitize_df_cols(so[want_so].head(25) if want_so else so.head(25)),
                use_container_width=True,
                hide_index=True,
            )
    rp = load_json(REPRICE_OPEN_ORDERS_PATH, show_error=False)
    if isinstance(rp, dict) and rp.get("timestamp"):
        st.markdown(
            "**Reprice open orders** (read-only; `python -m services.reprice_open_orders --mode paper`)"
        )
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Stale buys seen", str(rp.get("stale_orders_seen", "—")))
        r2.metric("Eligible to reprice", str(rp.get("eligible_to_reprice", "—")))
        r3.metric("Replacements submitted", str(rp.get("replacement_orders_submitted", "—")))
        r4.metric("Dry-run", "yes" if rp.get("dry_run") else "no")
        r5.metric("Buffer (bps)", str(rp.get("buy_buffer_bps", "—")))
        sy = rp.get("symbols_repriced")
        if isinstance(sy, list) and sy:
            st.caption("**Symbols repriced:** " + ", ".join(str(x) for x in sy[:40]))
        if rp.get("replacement_session"):
            st.caption(f"**Replacement session:** `{rp.get('replacement_session')}`")
    rl = load_json(REPRICE_LADDER_RUN_PATH, show_error=False)
    if isinstance(rl, dict) and rl.get("timestamp"):
        st.markdown(
            "**Adaptive Repricing Ladder** (read-only; `python -m services.reprice_order_ladder --mode paper`)"
        )
        st.caption(
            "Multi-stage BUY limit repricing; paper `--execute` only. Does not run from the scheduler."
        )
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        l1.metric("Eligible (seen)", str(rl.get("eligible_orders_seen", "—")))
        l2.metric("Advanced", str(rl.get("orders_advanced", "—")))
        l3.metric("Replacements submitted", str(rl.get("replacement_orders_submitted", "—")))
        sc = rl.get("stage_counts") if isinstance(rl.get("stage_counts"), dict) else {}
        l4.metric(
            "Stage counts",
            (
                ", ".join(
                    f"{k}:{sc.get(k, 0)}" for k in ("stage_1", "stage_2", "stage_3", "stage_4")
                )
                if sc
                else "—"
            ),
        )
        l5.metric("Dry-run", "yes" if rl.get("dry_run") else "no")
        l6.metric("Mode", str(rl.get("mode", "—")))
        sy = rl.get("symbols_repriced")
        if isinstance(sy, list) and sy:
            st.caption("**Symbols repriced:** " + ", ".join(str(x) for x in sy[:40]))
        if rl.get("replacement_session"):
            st.caption(f"**Replacement session:** `{rl.get('replacement_session')}`")
    st.divider()

    # ── Section 5 — Lifecycle distribution ───────────────────────────────
    st.subheader("5 — Lifecycle distribution (effective)")
    le = load_csv(SIGNAL_LIFECYCLE_EFFECTIVE_PATH, show_error=False)
    if le is None or le.empty:
        st.info("No **signal_lifecycle_effective.csv** (or empty).")
    else:
        le = sanitize_df_cols(le)
        stance_col = None
        for c in ("effective_stance", "stance", "lifecycle_action"):
            if c in le.columns:
                stance_col = c
                break
        if stance_col:
            vc = le[stance_col].fillna("").astype(str).str.strip().str.upper().value_counts()
            st.markdown("**Stance counts**")
            c1, c2 = st.columns([1, 2])
            with c1:
                for k, v in vc.items():
                    st.metric(str(k) if k else "(blank)", int(v))
            with c2:
                try:
                    chart_df = pd.DataFrame({"count": vc})
                    st.bar_chart(chart_df)
                except Exception:
                    st.dataframe(vc.to_frame("count"), use_container_width=True)
        else:
            st.warning("No stance column found (expected effective_stance or stance).")
        pos_col = None
        for c in ("effective_position_state", "position_state"):
            if c in le.columns:
                pos_col = c
                break
        if pos_col:
            pv = le[pos_col].fillna("").astype(str).str.strip().str.upper().value_counts()
            st.markdown("**Effective position state**")
            n_pc = min(6, max(1, len(pv)))
            cols_pos = st.columns(n_pc)
            for i, (k, v) in enumerate(pv.items()):
                if i < len(cols_pos):
                    cols_pos[i].metric(str(k) if k else "(blank)", int(v))
    st.divider()

    # ── Section 6 — Broker / snapshots ───────────────────────────────────
    st.subheader("6 — Broker / snapshot overview")
    oos = load_csv(OPEN_ORDERS_SNAPSHOT_PATH, show_error=False)
    ror = load_csv(RECENT_ORDERS_PATH, show_error=False)
    pss = load_csv(POSITIONS_SNAPSHOT_PATH, show_error=False)
    lol = load_csv(LIVE_ORDERS_LOG_PATH, show_error=False)

    n_open = len(oos) if oos is not None else 0
    n_rec = len(ror) if ror is not None else 0
    n_pos = len(pss) if pss is not None else 0
    n_log = len(lol) if lol is not None else 0
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("open_orders_snapshot rows", n_open)
    s2.metric("recent_orders rows", n_rec)
    s3.metric("positions_snapshot rows", n_pos)
    s4.metric("live_orders_log rows", n_log)

    oc1, oc2 = st.columns(2)
    with oc1:
        st.caption("**Open orders (preview)**")
        if oos is not None and not oos.empty:
            o2 = sanitize_df_cols(oos)
            pref = ["symbol", "ticker", "side", "qty", "status", "order_id"]
            show_o = [c for c in pref if c in o2.columns]
            st.dataframe(
                o2[show_o].head(15) if show_o else o2.head(15),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Missing or empty **open_orders_snapshot.csv**.")
    with oc2:
        st.caption("**Positions (preview)**")
        if pss is not None and not pss.empty:
            p2 = sanitize_df_cols(pss)
            pref_p = ["symbol", "ticker", "qty", "quantity", "market_value", "side"]
            show_p = [c for c in pref_p if c in p2.columns]
            st.dataframe(
                p2[show_p].head(15) if show_p else p2.head(15),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Missing or empty **positions_snapshot.csv**.")
    st.divider()

    # ── Section 7 — Paper cycle log ──────────────────────────────────────
    st.subheader("7 — Recent paper cycle log")
    pcl = load_csv(PAPER_TRADE_CYCLE_LOG_PATH, show_error=False)
    if pcl is None or pcl.empty:
        st.info("No **paper_trade_cycle_log.csv** yet.")
    else:
        pcl = sanitize_df_cols(pcl)
        if "ts_utc" in pcl.columns:
            try:
                pcl = pcl.sort_values("ts_utc", ascending=False, na_position="last")
            except Exception:
                pass
        want_l = [
            "ts_utc",
            "ok",
            "blocked",
            "pipeline_ok",
            "execute_ok",
            "manage_ok",
            "snapshot_ok",
            "manage_execute",
            "notes",
        ]
        show_l = [c for c in want_l if c in pcl.columns]
        st.dataframe(
            pcl[show_l].head(40) if show_l else pcl.head(40),
            use_container_width=True,
            hide_index=True,
        )


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
    ("Signals", "🚀 Trade Opportunities"): page_trade_opportunities,
    ("Signals", "🔍 Reconciliation"): page_reconciliation,
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
    ("System", "⚙️ Execution Dashboard"): page_execution_dashboard,
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
        "🚀 Trade Opportunities",
        "🔍 Reconciliation",
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
        "⚙️ Execution Dashboard",
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
    render_heartbeat_freshness_banner()
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
