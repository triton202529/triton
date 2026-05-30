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
REPRICE_OPEN_ORDERS_CSV_PATH = RESULTS_DIR / "reprice_open_orders.csv"
EXECUTION_INTELLIGENCE_CSV_PATH = RESULTS_DIR / "execution_intelligence.csv"
FEEDBACK_LOOP_REPORT_CSV_PATH = RESULTS_DIR / "feedback_loop_report.csv"
FEEDBACK_RECOMMENDATIONS_CSV_PATH = RESULTS_DIR / "feedback_recommendations.csv"
FEEDBACK_LOOP_SUMMARY_JSON_PATH = RESULTS_DIR / "feedback_loop_summary.json"
ADAPTATION_PROPOSALS_CSV_PATH = RESULTS_DIR / "adaptation_proposals.csv"
ADAPTATION_REVIEW_QUEUE_CSV_PATH = RESULTS_DIR / "adaptation_review_queue.csv"
ADAPTATION_SUMMARY_JSON_PATH = RESULTS_DIR / "adaptation_summary.json"
APPLIED_ADJUSTMENTS_CSV_PATH = RESULTS_DIR / "applied_adjustments.csv"
APPLIED_ADJUSTMENTS_JSON_PATH = RESULTS_DIR / "applied_adjustments.json"
APPLY_LOG_CSV_PATH = RESULTS_DIR / "apply_log.csv"
APPLY_SUMMARY_JSON_PATH = RESULTS_DIR / "apply_summary.json"
ADAPTATION_SIMULATION_CSV_PATH = RESULTS_DIR / "adaptation_simulation.csv"
ADAPTATION_SIMULATION_SUMMARY_JSON_PATH = RESULTS_DIR / "adaptation_simulation_summary.json"
REPRICE_LADDER_RUN_PATH = RESULTS_DIR / "reprice_ladder_run.json"
SIGNAL_LIFECYCLE_EFFECTIVE_PATH = RESULTS_DIR / "signal_lifecycle_effective.csv"
OPEN_ORDERS_SNAPSHOT_PATH = RESULTS_DIR / "open_orders_snapshot.csv"
RECENT_ORDERS_PATH = RESULTS_DIR / "recent_orders.csv"
LIVE_ORDERS_LOG_PATH = RESULTS_DIR / "live_orders_log.csv"
PERFORMANCE_INTELLIGENCE_CSV_PATH = RESULTS_DIR / "performance_intelligence.csv"
PERFORMANCE_INTELLIGENCE_BY_SYMBOL_CSV_PATH = RESULTS_DIR / "performance_intelligence_by_symbol.csv"
PERFORMANCE_INTELLIGENCE_SUMMARY_JSON_PATH = RESULTS_DIR / "performance_intelligence_summary.json"
PERFORMANCE_RISK_OVERLAY_CSV_PATH = RESULTS_DIR / "performance_risk_overlay.csv"

# Nice-to-have (do not force FAIL)
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"

# Research / Intel (optional; dashboard-only)
NEWS_SENTIMENT_PATH = RESULTS_DIR / "news_sentiment.csv"
SMART_ALERTS_PATH = RESULTS_DIR / "smart_alerts.csv"
ECONOMIC_CALENDAR_PATH = RESULTS_DIR / "economic_calendar.csv"

# --- Dashboard CSV inventory (read-only; all loads go through read_csv_dashboard / load_csv) ---
# Core: portfolio_history.csv, trade_log.csv, positions_snapshot.csv, signals_with_rationale.csv,
#   signals.csv, signal_lifecycle.csv, target_weights.csv, stock_scores.csv, trade_opportunities.csv,
#   lifecycle_reconciliation.csv, live_orders.csv
# Diagnostics / execution observability: execution_plan.csv, manage_positions_plan.csv,
#   reallocation_plan.csv, execution_drop_diagnostics.csv, signal_pressure_diagnostics.csv,
#   stale_open_orders.csv, signal_lifecycle_effective.csv, open_orders_snapshot.csv, recent_orders.csv,
#   live_orders_log.csv, paper_trade_cycle_log.csv
# Research / Intelligence: news_sentiment.csv, smart_alerts.csv, economic_calendar.csv

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


def empty_table(columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Minimal empty DataFrame for placeholder tables (dashboard display only)."""
    if columns:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_csv_bytesafe(path_str: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Cached CSV read (no Streamlit side effects). Returns (df, err):
      - missing path      → (None, None)
      - empty (0 bytes)   → (empty_table(), "empty")
      - OS error          → (None, "os: …")
      - parse / other     → (None, short message)
      - success           → (df, None); 0 rows after parse → (empty_table(), "empty")
    """
    path = Path(path_str)
    try:
        if not path.exists():
            return None, None
    except OSError as e:
        return None, f"os: {_short_err(e)}"
    try:
        if path.stat().st_size == 0:
            return empty_table(), "empty"
    except OSError as e:
        return None, f"os: {_short_err(e)}"
    try:
        df = pd.read_csv(path)
    except OSError as e:
        return None, f"os: {_short_err(e)}"
    except Exception as e:
        return None, _short_err(e)
    try:
        df = sanitize_df_cols(df)
    except Exception as e:
        return None, _short_err(e)
    if df.empty:
        return empty_table(), "empty"
    return df, None


def load_csv(path: Path, *, show_error: bool = True) -> Optional[pd.DataFrame]:
    """
    CSV loader for validators and internal use.
      - show_error=True: emits st.error on parse/OS failure (not for missing or empty file)
      - show_error=False: silent
    Returns None on missing, parse failure, or OS failure; returns empty DataFrame when file is empty.
    """
    df, err = _read_csv_bytesafe(str(path))
    if err == "empty":
        return df
    if err and show_error:
        st.error(f"❌ Failed loading {path.name}: {err}")
    if err:
        return None
    return df


def load_first_nonempty_csv(
    paths: List[Path],
    *,
    label: str = "CSV",
    columns: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Try paths in order; return first non-empty readable DataFrame.
    Emits at most one warning if all candidates fail due to parse/OS errors (not missing/empty).
    `columns` is reserved for future empty placeholders; unused for now.
    """
    _ = columns
    hard_errors: List[str] = []
    for p in paths:
        df, err = _read_csv_bytesafe(str(p))
        if err is None and df is not None and not df.empty:
            return df
        if err and err != "empty":
            hard_errors.append(f"{p.name}: {err}")
    if hard_errors:
        st.warning(
            f"{label}: no usable file found. "
            + " · ".join(hard_errors[:3])
            + ("…" if len(hard_errors) > 3 else "")
        )
    return None


def read_csv_dashboard(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Dashboard CSV load. Returns (df, err):
      - (None, None)           — file missing
      - (empty_df, "empty")    — empty or zero-byte file
      - (None, "os: …")        — exists/stat/read OS error
      - (None, msg)            — pandas parse error or malformed content
      - (df, None)             — success with ≥1 row
    """
    return _read_csv_bytesafe(str(path))


def csv_usable_rows(path: Path) -> Optional[pd.DataFrame]:
    """Optional CSV panels: return a DataFrame only when rows exist; else None (silent)."""
    df, err = read_csv_dashboard(path)
    if err is None and df is not None and not df.empty:
        return df
    return None


@st.cache_data(show_spinner=False)
def _read_json_bytesafe(path_str: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Cached JSON loader with safe error return (no Streamlit side effects inside cache)."""
    path = Path(path_str)
    try:
        if not path.exists():
            return None, None
    except OSError as e:
        return None, f"os: {_short_err(e)}"
    try:
        if path.stat().st_size == 0:
            return None, None
    except OSError as e:
        return None, f"os: {_short_err(e)}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except OSError as e:
        return None, f"os: {_short_err(e)}"
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
            allow_empty=True,
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
            allow_empty=True,
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
            allow_empty=True,
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
            allow_empty=True,
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
        try:
            results = validate_all_contracts()
            st.session_state["contract_results"] = results
            st.session_state["contract_summary"] = contracts_summary(results)
            st.session_state.pop("contracts_validation_crash", None)
        except Exception as e:
            st.session_state["contract_results"] = []
            st.session_state["contract_summary"] = {
                "total": 0,
                "ok": 0,
                "failed": 1,
                "error_count": 1,
                "warn_count": 0,
                "info_count": 0,
            }
            st.session_state["contracts_validation_crash"] = _short_err(e)

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

    rationale_exists = False
    try:
        rationale_exists = SIGNALS_RATIONALE_PATH.is_file() and (
            SIGNALS_RATIONALE_PATH.stat().st_size > 0
        )
    except OSError:
        rationale_exists = False

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

    crash = st.session_state.get("contracts_validation_crash")
    if crash:
        st.warning(
            "Contract validator raised an unexpected error (dashboard stays usable). "
            f"Details: {crash}",
            icon="⚠️",
        )

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

    df, csv_err = read_csv_dashboard(PORTFOLIO_HISTORY_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read portfolio_history.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No portfolio history yet — **portfolio_history.csv** is missing or empty.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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

    df, snap_err = read_csv_dashboard(POSITIONS_SNAPSHOT_PATH)
    if snap_err and snap_err != "empty":
        st.warning(f"Could not read positions_snapshot.csv: {snap_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return

    if (df is None or df.empty) and PORTFOLIO_HISTORY_PATH.exists():
        legacy, leg_err = read_csv_dashboard(PORTFOLIO_HISTORY_PATH)
        if (
            (not leg_err or leg_err == "empty")
            and legacy is not None
            and not legacy.empty
            and (
                "ticker" in legacy.columns or "symbol" in legacy.columns or "sym" in legacy.columns
            )
        ):
            df = legacy

    if df is None or df.empty:
        st.info(
            "No positions snapshot yet. Add **data/results/positions_snapshot.csv** "
            "(recommended), or per-ticker rows in **portfolio_history.csv** (legacy)."
        )
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
    df, csv_err = read_csv_dashboard(TRADE_LOG_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read trade_log.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No trade log yet — **trade_log.csv** is missing or empty.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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

    df, csv_err = read_csv_dashboard(SIGNAL_LIFECYCLE_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read signal_lifecycle.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info(
            "No **signal_lifecycle.csv** yet (or it is empty). Generate it from your lifecycle step."
        )
        st.code(
            "python services/build_signal_lifecycle_state.py --results-dir data/results",
            language="bash",
        )
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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

    try:
        has_file = (
            TRADE_OPPORTUNITIES_PATH.is_file() and TRADE_OPPORTUNITIES_PATH.stat().st_size > 0
        )
    except OSError:
        has_file = False
    if not has_file:
        st.info("No **trade_opportunities.csv** yet (or file is empty).")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return

    df, csv_err = read_csv_dashboard(TRADE_OPPORTUNITIES_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read trade_opportunities.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None:
        st.info("No trade opportunities available.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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

    df, csv_err = read_csv_dashboard(LIFECYCLE_RECONCILIATION_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read lifecycle_reconciliation.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info(
            "No reconciliation data yet. Generate **data/results/lifecycle_reconciliation.csv** with `python -m services.reconcile_lifecycle_vs_positions`."
        )
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
        st.warning(f"lifecycle_reconciliation.csv missing expected columns: {missing}")
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
        lc_ok = False
        try:
            lc_ok = SIGNAL_LIFECYCLE_PATH.is_file() and SIGNAL_LIFECYCLE_PATH.stat().st_size > 0
        except OSError:
            lc_ok = False
        if lc_ok:
            df, _lc_err = read_csv_dashboard(SIGNAL_LIFECYCLE_PATH)
            if (_lc_err and _lc_err != "empty") or df is None or df.empty:
                df = None

    if df is None or df.empty:
        if stale:
            st.warning(
                "signal_lifecycle.csv is older than signals_with_rationale.csv / signals.csv. "
                "Showing rationale or raw signals until lifecycle is regenerated."
            )
        df = load_first_nonempty_csv(
            [SIGNALS_RATIONALE_PATH, SIGNALS_PATH],
            label="Signals (rationale or raw)",
        )

    if df is None or df.empty:
        st.info(
            "No signals to display yet — **signal_lifecycle.csv**, **signals_with_rationale.csv**, "
            "and **signals.csv** are missing, empty, or unreadable."
        )
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
    df, csv_err = read_csv_dashboard(SIGNALS_RATIONALE_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read signals_with_rationale.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No signals_with_rationale.csv found yet. Run your training step that writes it.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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

    df, csv_err = read_csv_dashboard(TARGET_WEIGHTS_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read target_weights.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info(
            "No target_weights.csv found yet. Generate it from your weighting step (e.g., allocator)."
        )
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
    df, csv_err = read_csv_dashboard(STOCK_SCORES_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read stock_scores.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No scores yet — **stock_scores.csv** is missing or empty.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
    df, csv_err = read_csv_dashboard(FEATURE_IMPORTANCE_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read feature_importance.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No feature_importance.csv yet. Your train step writes it when available.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
    df, csv_err = read_csv_dashboard(MODEL_COMPARISON_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read model_comparison.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No model_comparison.csv yet. Your train step writes it when available.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
    df, csv_err = read_csv_dashboard(TRADE_LOG_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read trade_log.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No trade log yet — **trade_log.csv** is missing or empty.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
    df, csv_err = read_csv_dashboard(NEWS_SENTIMENT_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not load news sentiment: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No news sentiment data available yet.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    st.dataframe(df, use_container_width=True)


def page_smart_alerts() -> None:
    st.markdown("### 🚨 Smart Alerts")
    df, csv_err = read_csv_dashboard(SMART_ALERTS_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not load smart alerts: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("Smart alerts file is missing or empty.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    st.dataframe(df, use_container_width=True)


def page_econ_calendar() -> None:
    st.markdown("### 📅 Economic Calendar")
    df, csv_err = read_csv_dashboard(ECONOMIC_CALENDAR_PATH)
    if csv_err and csv_err != "empty":
        st.warning("Economic calendar data could not be loaded.")
        st.caption(str(csv_err))
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None or df.empty:
        st.info("No economic calendar data available yet.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    st.dataframe(df, use_container_width=True)


def page_execution_health() -> None:
    st.markdown("### ⚙ Execution / Health")

    c1, c2 = st.columns(2)

    with c1:
        st.caption("Open orders log (live_orders.csv)")
        df, csv_err = read_csv_dashboard(LIVE_ORDERS_PATH)
        if csv_err and csv_err != "empty":
            st.warning(f"Could not read live_orders.csv: {csv_err}")
            st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        elif df is not None and not df.empty:
            st.dataframe(df.tail(50), use_container_width=True)
        else:
            st.info("No live_orders.csv yet.")
            st.dataframe(empty_table(), use_container_width=True, hide_index=True)

    with c2:
        st.caption("Guard snapshot (guard_snapshot.json)")
        guard = load_json(GUARD_SNAPSHOT_PATH, show_error=False)
        if guard:
            st.json(guard)
        else:
            st.info("No guard_snapshot.json yet (or file could not be read).")


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
    ep_csv = csv_usable_rows(EXECUTION_PLAN_CSV_PATH)
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
    mp_csv = csv_usable_rows(MANAGE_POSITIONS_PLAN_CSV_PATH)
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
    rp_csv = csv_usable_rows(REALLOCATION_PLAN_CSV_PATH)
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
    edf = csv_usable_rows(EXEC_DROP_CSV_PATH)
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
    spd_csv = csv_usable_rows(SIGNAL_PRESSURE_DIAG_CSV_PATH)
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
        so = csv_usable_rows(STALE_OPEN_ORDERS_CSV_PATH)
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
    le = csv_usable_rows(SIGNAL_LIFECYCLE_EFFECTIVE_PATH)
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
    oos = csv_usable_rows(OPEN_ORDERS_SNAPSHOT_PATH)
    ror = csv_usable_rows(RECENT_ORDERS_PATH)
    pss = csv_usable_rows(POSITIONS_SNAPSHOT_PATH)
    lol = csv_usable_rows(LIVE_ORDERS_LOG_PATH)

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
    pcl = csv_usable_rows(PAPER_TRADE_CYCLE_LOG_PATH)
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

    df, csv_err = read_csv_dashboard(LIVE_ORDERS_PATH)
    if csv_err and csv_err != "empty":
        st.warning(f"Could not read live_orders.csv: {csv_err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if df is None:
        st.info("**live_orders.csv** is not available yet.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return

    df = sanitize_df_cols(df)
    if df.empty:
        st.info("live_orders.csv loaded, but contains 0 rows.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
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
# Ticker Decision Timeline (observability)
# ──────────────────────────────
# Dashboard-only helpers. All side effects are read-only.

_TDT_HISTORY_CANDIDATES: Tuple[str, ...] = (
    "signals_lifecycle.csv",  # plural; often multi-day history if present
    "lifecycle_history.csv",
    "signal_lifecycle_history.csv",
)

_TDT_PREFERRED_COLUMNS: Tuple[str, ...] = (
    "date",
    "ticker",
    "signal",
    "decision_action",
    "state_transition",
    "held_state",
    "confidence",
    "prior_confidence",
    "confidence_change",
    "score",
    "prior_score",
    "score_change",
    "delta_pct",
    "close",
    "predicted_close",
    "rationale",
    "decision_reason",
    "stance",
    "lifecycle_action",
    "position_state",
    "last_action",
)


def _tdt_select_existing_columns(df: pd.DataFrame, preferred: Tuple[str, ...]) -> List[str]:
    """Return preferred columns that actually exist in df, in preferred order."""
    return [c for c in preferred if c in df.columns]


def _tdt_safe_metric(row: Optional[pd.Series], col: str, default: str = "N/A") -> str:
    """Read a scalar from a row safely, formatting floats compactly."""
    if row is None or col not in row.index:
        return default
    val = row.get(col)
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
    except Exception:
        return default
    if isinstance(val, float):
        return f"{val:.4f}"
    s = str(val).strip()
    return s if s else default


def _tdt_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: add a _ts datetime column from 'date' or 'as_of_date' if available."""
    df = df.copy()
    ts = None
    if "date" in df.columns:
        ts = pd.to_datetime(df["date"], errors="coerce")
    if (ts is None or ts.isna().all()) and "as_of_date" in df.columns:
        ts = pd.to_datetime(df["as_of_date"], errors="coerce")
    df["_ts"] = ts if ts is not None else pd.NaT
    return df


def _tdt_load_primary() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load signal_lifecycle.csv (current-state, one row per ticker)."""
    df, err = read_csv_dashboard(SIGNAL_LIFECYCLE_PATH)
    if err == "empty":
        return df, "empty"
    if err:
        return None, err
    if df is None or df.empty:
        return None, "empty"
    return sanitize_df_cols(df), None


def _tdt_load_history_optional() -> Optional[pd.DataFrame]:
    """Optional multi-day history source (best-effort, silent on missing)."""
    for name in _TDT_HISTORY_CANDIDATES:
        p = RESULTS_DIR / name
        if not p.is_file() or p.stat().st_size == 0:
            continue
        df, err = read_csv_dashboard(p)
        if err or df is None or df.empty:
            continue
        return sanitize_df_cols(df)
    return None


def _tdt_build_timeline_for_ticker(
    primary: pd.DataFrame, history: Optional[pd.DataFrame], ticker: str
) -> pd.DataFrame:
    """
    Combine primary (1 row per ticker) with optional history (many rows per ticker),
    keep only rows for `ticker`, sort ascending by date.
    """
    frames: List[pd.DataFrame] = []
    for src in (history, primary):
        if src is None or src.empty:
            continue
        if "ticker" not in src.columns:
            continue
        sel = src[src["ticker"].astype(str).str.upper() == str(ticker).upper()].copy()
        if not sel.empty:
            frames.append(sel)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    combined = _tdt_parse_dates(combined)

    # De-dupe: prefer rows that carry decision_action (from the newer primary) when a
    # history row and a primary row share the same date.
    if "_ts" in combined.columns:
        combined["_has_decision"] = (
            combined["decision_action"].notna().astype(int)
            if "decision_action" in combined.columns
            else 0
        )
        combined = combined.sort_values(
            ["_ts", "_has_decision"], ascending=[True, True], na_position="last"
        )
        combined = combined.drop_duplicates(subset=["_ts"], keep="last")
        combined = combined.drop(columns=["_has_decision"], errors="ignore")
        combined = combined.sort_values("_ts", ascending=True, na_position="last").reset_index(
            drop=True
        )
    return combined


def _tdt_compute_change_rows(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows that represent meaningful changes:
      - signal_changed is truthy, OR
      - decision_action differs from previous row, OR
      - state_transition is non-empty and not a pure hold-default, OR
      - |confidence_change| >= 0.04, OR
      - |score_change| >= 0.03
    """
    if df_ticker.empty:
        return df_ticker

    df = df_ticker.copy().reset_index(drop=True)
    mask = pd.Series(False, index=df.index)

    if "signal_changed" in df.columns:
        sc = df["signal_changed"]
        # Coerce both strings ("True"/"False") and booleans to a proper bool mask.
        sc_bool = sc.map(
            lambda v: (
                str(v).strip().lower() in ("true", "1", "yes")
                if v is not None and not (isinstance(v, float) and pd.isna(v))
                else False
            )
        )
        mask = mask | sc_bool.astype(bool)

    if "decision_action" in df.columns:
        prev = df["decision_action"].shift(1)
        mask = mask | (df["decision_action"].fillna("") != prev.fillna(""))
        # Don't flag the very first row solely on "differs from prev NaN"
        if len(mask):
            mask.iloc[0] = bool(mask.iloc[0]) and False

    if "state_transition" in df.columns:
        st_col = df["state_transition"].astype(str).fillna("")
        interesting = ~st_col.isin(["", "LONG_HOLD", "FLAT_WAIT", "nan", "NaN"])
        mask = mask | interesting

    if "confidence_change" in df.columns:
        mask = mask | (pd.to_numeric(df["confidence_change"], errors="coerce").abs() >= 0.04)

    if "score_change" in df.columns:
        mask = mask | (pd.to_numeric(df["score_change"], errors="coerce").abs() >= 0.03)

    # Always include the most recent row so the user sees the current state.
    if len(df):
        mask.iloc[-1] = True

    return df[mask].reset_index(drop=True)


def _tdt_display_decision_value(row: Optional[pd.Series]) -> str:
    """
    Display-only fallback for the summary "Decision" card.

    Strict priority (read-only, never writes back to files):
      1. decision_action if present and non-empty
      2. FLAT              -> "WAIT"
      3. LONG + BUY        -> "HOLD"
      4. LONG + HOLD       -> "HOLD"
      5. LONG + SELL       -> "EXIT"
      6. "N/A"
    """
    if row is None:
        return "N/A"

    raw = row.get("decision_action") if "decision_action" in row.index else None
    if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
        s = str(raw).strip()
        if s and s.lower() not in ("nan", "none", "nat"):
            return s.upper()

    held = str(row.get("held_state") or "").strip().upper() if "held_state" in row.index else ""
    signal = str(row.get("signal") or "").strip().upper() if "signal" in row.index else ""

    if held == "FLAT":
        return "WAIT"
    if held == "LONG":
        if signal == "BUY":
            return "HOLD"
        if signal == "HOLD":
            return "HOLD"
        if signal == "SELL":
            return "EXIT"
    return "N/A"


# Display-only fallbacks for missing values in the timeline/events tables.
# Text defaults use sentinel strings; numeric defaults use empty strings ("blank").
_TDT_TEXT_FILL_DEFAULTS: Dict[str, str] = {
    "decision_action": "INIT",
    "state_transition": "INIT",
    "held_state": "FLAT",
    "decision_reason": "",
    "prior_signal": "",
    "rationale": "",
    "signal": "",
    "stance": "",
    "lifecycle_action": "",
    "position_state": "",
    "last_action": "",
}
_TDT_NUMERIC_BLANK_COLS: Tuple[str, ...] = (
    "prior_confidence",
    "prior_score",
    "confidence_change",
    "score_change",
)


def _tdt_display_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with None/NaN/NaT replaced by display-friendly values.

    This only affects rendering — source CSVs are NEVER mutated.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    for col, default in _TDT_TEXT_FILL_DEFAULTS.items():
        if col not in out.columns:
            continue
        ser = out[col]
        # Map anything that looks like a missing value to the default.
        out[col] = ser.map(
            lambda v, _d=default: (
                _d
                if v is None
                or (isinstance(v, float) and pd.isna(v))
                or str(v).strip().lower() in ("nan", "none", "nat", "")
                else str(v)
            )
        )

    # Numeric columns: keep valid floats (formatted to 4dp), blank out everything else.
    for col in _TDT_NUMERIC_BLANK_COLS:
        if col not in out.columns:
            continue
        num = pd.to_numeric(out[col], errors="coerce")
        out[col] = num.map(lambda v: "" if pd.isna(v) else f"{float(v):+.4f}")

    return out


def _tdt_smoothed_series(df: pd.DataFrame, col: str, window: int = 5) -> Optional[pd.DataFrame]:
    """
    Build a chart-friendly frame with the raw series + a rolling-mean smoothed series.

    Returns a DataFrame with columns like `{col}` and `{col}_smooth`, or None if the
    input column is absent/empty. The smoothed series uses min_periods=1 so it renders
    immediately on short timelines and converges to a true 5-bar mean when enough rows
    exist. When the timeline is shorter than `window`, the smoothed line is identical
    to the raw series by design, so the chart still reads clearly.
    """
    if col not in df.columns:
        return None
    raw = pd.to_numeric(df[col], errors="coerce")
    if raw.notna().sum() == 0:
        return None

    smooth_name = f"{col}_smooth"
    out = pd.DataFrame({col: raw, smooth_name: raw.rolling(window=window, min_periods=1).mean()})
    # Chart is cleanest when the smoothed line is the primary visual; raw is a thin echo.
    return out[[smooth_name, col]]


def _tdt_describe_change(delta: Any, label: str) -> Optional[str]:
    """Compact English phrase for a signed change value; None if not material."""
    try:
        if delta is None or (isinstance(delta, float) and pd.isna(delta)):
            return None
        v = float(delta)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    mag = abs(v)
    # Threshold mirrors the "meaningful changes" detector (0.04 for confidence, 0.03 for score).
    min_mag = 0.04 if label == "confidence" else 0.03
    if mag < min_mag:
        return None
    direction = "rose" if v > 0 else "fell"
    return f"{label} {direction} by {mag:.2f}"


def _tdt_event_sentence(row: pd.Series) -> str:
    """
    Render one timeline row as a single human-readable sentence.

    Priority of sources: decision_action → state_transition → signal/rationale.
    When confidence_change or score_change is materially non-zero, it is woven into
    the sentence in plain English (no awkward slash phrases).
    """
    ts = row.get("_ts") if "_ts" in row.index else None
    if ts is None or (isinstance(ts, float) and pd.isna(ts)) or pd.isna(ts):
        ts = row.get("date") or row.get("as_of_date") or ""
    try:
        date_label = pd.to_datetime(ts).strftime("%Y-%m-%d")
    except Exception:
        date_label = str(ts) if ts else "(no date)"

    action = str(row.get("decision_action") or "").strip().upper()
    transition = str(row.get("state_transition") or "").strip().upper()
    signal = str(row.get("signal") or "").strip().upper()
    rationale = str(row.get("rationale") or "").strip()

    conf_change = row.get("confidence_change") if "confidence_change" in row.index else None
    score_change = row.get("score_change") if "score_change" in row.index else None
    conf_phrase = _tdt_describe_change(conf_change, "confidence")
    score_phrase = _tdt_describe_change(score_change, "score")
    # Prefer confidence phrasing; fall back to score if confidence isn't material.
    change_phrase = conf_phrase or score_phrase

    # Compose the core phrase (with inline change context when available).
    def _with_change(base: str, on_movement: str, fallback: str) -> str:
        if change_phrase:
            return f"{base} {on_movement} {change_phrase}."
        return f"{base}{fallback}"

    if action == "BUY" or transition == "FLAT_TO_LONG":
        phrase = _with_change(
            "New bullish entry triggered",
            "as",
            ".",
        )
    elif action == "ADD" or transition == "LONG_ADD":
        phrase = _with_change(
            "Conviction improved; added to the position",
            "as",
            ".",
        )
    elif action == "HOLD" or transition == "LONG_HOLD":
        # HOLD rarely needs a movement suffix; keep it calm unless a material change exists.
        if change_phrase:
            phrase = f"Bullish signal remained intact; held position ({change_phrase})."
        else:
            phrase = "Bullish signal remained intact; held position."
    elif action == "TRIM" or transition == "LONG_TRIM":
        phrase = _with_change(
            "Signal weakened; trimmed exposure",
            "as conviction softened —",
            " as conviction softened.",
        )
    elif action == "EXIT" or transition == "LONG_EXIT":
        phrase = _with_change(
            "Bearish deterioration triggered exit",
            "after",
            ".",
        )
    elif action == "WAIT" or transition == "FLAT_WAIT":
        phrase = "No actionable setup; waiting."
    elif signal in ("BUY", "SELL", "HOLD"):
        phrase = f"Signal: {signal}."
        if change_phrase:
            phrase = f"Signal: {signal} ({change_phrase})."
    elif rationale:
        phrase = rationale if rationale.endswith(".") else rationale + "."
    else:
        phrase = "Status update."

    return f"{date_label}: {phrase}"


def page_ticker_decision_timeline() -> None:
    """Observability page: per-ticker day-by-day decision timeline."""
    st.markdown("### 📜 Ticker Decision Timeline")
    st.caption(
        "Shows signal, decision_action, confidence, score, and lifecycle history for the "
        "selected ticker. Read-only observability view — no changes to signals, lifecycle, "
        "or execution."
    )

    # --- Load primary source safely ---
    primary, err = _tdt_load_primary()
    if err == "empty":
        st.info(
            "No **signal_lifecycle.csv** yet (or it is empty). "
            "Run the lifecycle step to populate it."
        )
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if err:
        st.warning(f"Could not read signal_lifecycle.csv: {err}")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return
    if primary is None or primary.empty:
        st.info("No lifecycle rows available.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return

    if "ticker" not in primary.columns:
        st.warning("signal_lifecycle.csv is missing the required `ticker` column.")
        st.dataframe(primary.head(50), use_container_width=True)
        return

    # --- Ticker selector ---
    tickers = sorted(
        [t for t in primary["ticker"].dropna().astype(str).str.upper().unique().tolist() if t]
    )
    if not tickers:
        st.info("No tickers found in signal_lifecycle.csv.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return

    default_idx = 0
    preferred_default = st.session_state.get("tdt_last_ticker")
    if preferred_default in tickers:
        default_idx = tickers.index(preferred_default)

    ticker = st.selectbox(
        "Select a ticker",
        options=tickers,
        index=default_idx,
        key="tdt_ticker",
        help="Type to filter. All tickers present in signal_lifecycle.csv.",
    )
    st.session_state["tdt_last_ticker"] = ticker

    # --- Optional history augment ---
    history = _tdt_load_history_optional()
    timeline = _tdt_build_timeline_for_ticker(primary, history, ticker)

    if timeline.empty:
        st.info(f"No timeline rows found for **{ticker}**.")
        st.dataframe(empty_table(), use_container_width=True, hide_index=True)
        return

    # --- Summary cards (latest row) ---
    latest = timeline.iloc[-1] if len(timeline) else None
    first_ts = timeline["_ts"].min() if "_ts" in timeline.columns else None
    last_ts = timeline["_ts"].max() if "_ts" in timeline.columns else None

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Signal", _tdt_safe_metric(latest, "signal"))
    with m2:
        # Display-only fallback: never show "N/A" when held_state + signal can imply a read.
        st.metric("Decision", _tdt_display_decision_value(latest))
    with m3:
        st.metric("Held", _tdt_safe_metric(latest, "held_state", default="FLAT"))
    with m4:
        st.metric("Confidence", _tdt_safe_metric(latest, "confidence"))
    with m5:
        st.metric("Score", _tdt_safe_metric(latest, "score"))
    with m6:
        st.metric("Rows", f"{len(timeline)}")

    def _fmt_ts(x: Any) -> str:
        try:
            if x is None or pd.isna(x):
                return "N/A"
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        except Exception:
            return "N/A"

    st.caption(
        f"Timeline span: **{_fmt_ts(first_ts)}** → **{_fmt_ts(last_ts)}**  ·  "
        f"Source: signal_lifecycle.csv"
        + (" (+ optional history file)" if history is not None else "")
    )

    # --- Full timeline table ---
    st.markdown("#### Full timeline")
    display_cols = _tdt_select_existing_columns(timeline, _TDT_PREFERRED_COLUMNS)
    if "date" not in display_cols and "_ts" in timeline.columns:
        # Expose a derived date column if user has no native date column
        timeline = timeline.assign(date=timeline["_ts"].dt.strftime("%Y-%m-%d"))
        display_cols = ["date"] + [c for c in display_cols if c != "date"]

    if not display_cols:
        st.warning("No recognizable columns to display.")
        st.dataframe(timeline.head(100), use_container_width=True, hide_index=True)
    else:
        # Display-only NaN/None/NaT cleanup; does not mutate source files.
        tl_display = _tdt_display_fillna(timeline[display_cols].reset_index(drop=True))
        st.dataframe(
            tl_display,
            use_container_width=True,
            hide_index=True,
        )

    # --- Meaningful changes ---
    st.markdown("#### Meaningful changes")
    st.caption(
        "Rows where the signal flipped, the decision_action changed, a non-trivial "
        "state transition fired, or confidence/score moved materially."
    )
    events = _tdt_compute_change_rows(timeline)
    if events.empty:
        st.info("No meaningful changes detected in the available history for this ticker.")
    else:
        ev_cols = _tdt_select_existing_columns(events, _TDT_PREFERRED_COLUMNS)
        if "date" not in ev_cols and "_ts" in events.columns:
            events = events.assign(date=events["_ts"].dt.strftime("%Y-%m-%d"))
            ev_cols = ["date"] + [c for c in ev_cols if c != "date"]
        # Display-only NaN/None/NaT cleanup; does not mutate source files.
        ev_display = _tdt_display_fillna(
            events[ev_cols].reset_index(drop=True) if ev_cols else events
        )
        st.dataframe(
            ev_display,
            use_container_width=True,
            hide_index=True,
        )

    # --- Charts (Streamlit native; each gated on column availability) ---
    st.markdown("#### Charts")
    chart_df = timeline.copy()
    if "_ts" in chart_df.columns:
        chart_df = chart_df.dropna(subset=["_ts"]).set_index("_ts")

    rendered_any = False

    # --- Confidence: prefer a 5-row rolling mean for readability; show raw as thin echo. ---
    conf_frame = _tdt_smoothed_series(chart_df, "confidence", window=5)
    if conf_frame is not None:
        try:
            n_real = int(pd.to_numeric(chart_df["confidence"], errors="coerce").notna().sum())
            label = (
                "**Confidence over time** (5-row smoothed, raw overlay)"
                if n_real >= 5
                else "**Confidence over time**"
            )
            st.markdown(label)
            st.line_chart(conf_frame)
            rendered_any = True
        except Exception as e:
            # Graceful fallback to raw series if smoothing/plotting fails.
            try:
                st.markdown("**Confidence over time**")
                st.line_chart(pd.to_numeric(chart_df["confidence"], errors="coerce"))
                rendered_any = True
            except Exception:
                st.caption(f"Confidence chart unavailable: {_short_err(e)}")

    # --- Score: same smoothing treatment when the column is present. ---
    score_frame = _tdt_smoothed_series(chart_df, "score", window=5)
    if score_frame is not None:
        try:
            n_real = int(pd.to_numeric(chart_df["score"], errors="coerce").notna().sum())
            label = (
                "**Score over time** (5-row smoothed, raw overlay)"
                if n_real >= 5
                else "**Score over time**"
            )
            st.markdown(label)
            st.line_chart(score_frame)
            rendered_any = True
        except Exception as e:
            try:
                st.markdown("**Score over time**")
                st.line_chart(pd.to_numeric(chart_df["score"], errors="coerce"))
                rendered_any = True
            except Exception:
                st.caption(f"Score chart unavailable: {_short_err(e)}")

    if "close" in chart_df.columns and chart_df["close"].notna().any():
        try:
            st.markdown("**Close over time**")
            st.line_chart(pd.to_numeric(chart_df["close"], errors="coerce"))
            rendered_any = True
        except Exception as e:
            st.caption(f"Close chart unavailable: {_short_err(e)}")

    if not rendered_any:
        st.info("No numeric series available to chart for this ticker.")

    # --- Plain-English event feed ---
    st.markdown("#### Decision feed")
    if events.empty:
        st.caption("Nothing to narrate yet.")
    else:
        for _, r in events.iterrows():
            st.write(f"• {_tdt_event_sentence(r)}")


# ──────────────────────────────
# EXECUTION INTELLIGENCE PAGE
# (dashboard-only; reads execution_intelligence.csv,
#  execution_plan.csv, reprice_open_orders.csv)
# ──────────────────────────────


def _ei_select_existing_columns(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    """Return preferred columns that actually exist in df, preserving order."""
    if df is None or df.empty:
        return []
    cols = list(df.columns)
    return [c for c in preferred if c in cols]


def _ei_safe_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Return df[col] if present, else None. Never raises."""
    try:
        if df is None or df.empty or col not in df.columns:
            return None
        return df[col]
    except Exception:
        return None


def _ei_safe_numeric(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Return df[col] coerced to numeric (NaNs dropped), else None."""
    s = _ei_safe_col(df, col)
    if s is None:
        return None
    try:
        out = pd.to_numeric(s, errors="coerce").dropna()
        return out if not out.empty else None
    except Exception:
        return None


def _ei_safe_metric(
    df: pd.DataFrame, col: str, agg: str = "mean", fmt: str = "{:.4f}", default: str = "N/A"
) -> str:
    """Compute a single aggregate metric on a numeric column with safe fallback."""
    s = _ei_safe_numeric(df, col)
    if s is None:
        return default
    try:
        if agg == "mean":
            v = float(s.mean())
        elif agg == "median":
            v = float(s.median())
        elif agg == "min":
            v = float(s.min())
        elif agg == "max":
            v = float(s.max())
        elif agg == "sum":
            v = float(s.sum())
        elif agg == "count":
            v = int(s.shape[0])
            return f"{v:,}"
        else:
            return default
        if not np.isfinite(v):
            return default
        return fmt.format(v)
    except Exception:
        return default


def _ei_value_counts_table(
    df: pd.DataFrame, col: str, order: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """Return a tidy value-counts DataFrame with stable column names."""
    s = _ei_safe_col(df, col)
    if s is None:
        return None
    try:
        s = s.fillna("").astype(str).str.strip().replace({"": "(missing)"})
        vc = s.value_counts(dropna=False)
        if order:
            present = [v for v in order if v in vc.index]
            tail = [v for v in vc.index if v not in present]
            vc = vc.reindex(present + tail).fillna(0).astype(int)
        out = vc.reset_index()
        out.columns = [col, "count"]
        total = int(out["count"].sum())
        if total > 0:
            out["pct"] = (out["count"] / total * 100.0).round(1)
        return out
    except Exception:
        return None


def _ei_top_reason_table(df: pd.DataFrame, col: str, top_n: int = 10) -> Optional[pd.DataFrame]:
    """Top-N value counts for a free-form text column (e.g. reasons)."""
    vc = _ei_value_counts_table(df, col)
    if vc is None or vc.empty:
        return None
    return vc.head(top_n).reset_index(drop=True)


def _ei_normalize_symbol_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'symbol' column if only 'ticker' is present (non-destructive copy)."""
    if df is None or df.empty:
        return df
    if "symbol" in df.columns:
        return df
    if "ticker" in df.columns:
        out = df.copy()
        out["symbol"] = out["ticker"]
        return out
    return df


def _ei_pick_time_col(df: pd.DataFrame) -> Optional[str]:
    """Pick the best timestamp-like column we know how to render."""
    if df is None or df.empty:
        return None
    for c in (
        "timestamp",
        "submitted_at",
        "created_at",
        "generated_at",
        "ts",
        "time",
        "snapshot_ts",
    ):
        if c in df.columns:
            return c
    return None


def _ei_truthy_count(df: pd.DataFrame, col: str) -> Optional[int]:
    """Count True-ish values in a column. Returns None if column missing."""
    s = _ei_safe_col(df, col)
    if s is None:
        return None
    try:
        s2 = s.astype(str).str.strip().str.lower()
        return int(s2.isin(["true", "1", "yes", "t"]).sum())
    except Exception:
        return None


def _ei_pct_of(numer: Optional[int], denom: int) -> str:
    if numer is None or denom <= 0:
        return "N/A"
    return f"{(numer / denom * 100.0):.1f}%"


def _ei_render_table(
    df: Optional[pd.DataFrame],
    *,
    height: Optional[int] = None,
    empty_msg: str = "No rows to display.",
) -> None:
    """Render a DataFrame with a friendly fallback when empty/None."""
    if df is None or (hasattr(df, "empty") and df.empty):
        st.caption(empty_msg)
        return
    try:
        if height is not None:
            st.dataframe(df, use_container_width=True, height=height)
        else:
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.caption(f"Table unavailable: {_short_err(e)}")


def _ei_load(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Wrap dashboard CSV loader and emit one st.warning on hard errors."""
    df, err = read_csv_dashboard(path)
    if err is None:
        return df
    if err == "empty":
        return df  # empty df renders cleanly downstream
    st.warning(f"{label}: could not read {path.name} ({err}).")
    return None


def page_execution_intelligence() -> None:
    """⚙️ Execution Intelligence — observability of EI annotations."""
    st.title("⚙️ Execution Intelligence")
    st.caption(
        "Execution quality, quote conditions, liquidity pressure, slippage, "
        "and partial-fill follow-up diagnostics."
    )

    ei_df = _ei_load(EXECUTION_INTELLIGENCE_CSV_PATH, "Execution Intelligence log")
    plan_df = _ei_load(EXECUTION_PLAN_CSV_PATH, "Execution Plan")
    reprice_df = _ei_load(REPRICE_OPEN_ORDERS_CSV_PATH, "Reprice / Partial Fill follow-up")

    ei_df = _ei_normalize_symbol_col(ei_df) if ei_df is not None else ei_df
    plan_df = _ei_normalize_symbol_col(plan_df) if plan_df is not None else plan_df
    reprice_df = _ei_normalize_symbol_col(reprice_df) if reprice_df is not None else reprice_df

    # Pick the "primary" enriched orders source: prefer the EI sidecar (per-submit
    # rows), fall back to the planning CSV when the sidecar is empty.
    primary: Optional[pd.DataFrame] = None
    primary_label = ""
    if ei_df is not None and not ei_df.empty:
        primary, primary_label = ei_df, "execution_intelligence.csv"
    elif plan_df is not None and not plan_df.empty:
        primary, primary_label = plan_df, "execution_plan.csv"

    if primary is None or primary.empty:
        st.info(
            "No execution-intelligence rows found yet. "
            "Run a planning or placement cycle to populate "
            "`data/results/execution_intelligence.csv` or "
            "`data/results/execution_plan.csv`."
        )

    # ── 1) TOP SUMMARY METRICS ────────────────────────────────────────
    st.markdown("### Summary")
    n_rows = int(primary.shape[0]) if primary is not None and not primary.empty else 0
    avg_q = (
        _ei_safe_metric(primary, "execution_quality_score", "mean", "{:.3f}")
        if primary is not None
        else "N/A"
    )

    risk_counts: Dict[str, int] = {}
    if primary is not None and "execution_risk_flag" in primary.columns:
        try:
            tmp = primary["execution_risk_flag"].fillna("").astype(str).str.strip().str.upper()
            risk_counts = tmp.value_counts(dropna=False).to_dict()
        except Exception:
            risk_counts = {}
    pct_low = _ei_pct_of(risk_counts.get("LOW", 0), n_rows) if n_rows else "N/A"
    pct_high = _ei_pct_of(risk_counts.get("HIGH", 0), n_rows) if n_rows else "N/A"

    stale_count = _ei_truthy_count(primary, "quote_is_stale") if primary is not None else None
    skip_flag_count = (
        _ei_truthy_count(primary, "execution_skip_flag") if primary is not None else None
    )

    defer_skip_count: Optional[int] = None
    if primary is not None and "execution_style" in primary.columns:
        try:
            es = primary["execution_style"].fillna("").astype(str).str.strip().str.upper()
            defer_skip_count = int(es.isin(["DEFER", "SKIP"]).sum())
        except Exception:
            defer_skip_count = None

    partial_action_count: Optional[int] = None
    if (
        reprice_df is not None
        and not reprice_df.empty
        and "partial_fill_action" in reprice_df.columns
    ):
        try:
            pa = reprice_df["partial_fill_action"].fillna("").astype(str).str.strip().str.upper()
            partial_action_count = int((pa != "").sum())
        except Exception:
            partial_action_count = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (primary)", f"{n_rows:,}" if n_rows else "0")
    c2.metric("Avg quality score", avg_q)
    c3.metric("% LOW risk", pct_low)
    c4.metric("% HIGH risk", pct_high)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Stale quotes", f"{stale_count:,}" if isinstance(stale_count, int) else "N/A")
    c6.metric(
        "DEFER+SKIP styles", f"{defer_skip_count:,}" if isinstance(defer_skip_count, int) else "N/A"
    )
    c7.metric(
        "Skip flag count", f"{skip_flag_count:,}" if isinstance(skip_flag_count, int) else "N/A"
    )
    c8.metric(
        "Partial-fill actions",
        f"{partial_action_count:,}" if isinstance(partial_action_count, int) else "N/A",
    )

    if primary is not None and not primary.empty:
        st.caption(f"Primary source: `{primary_label}` ({n_rows:,} rows)")

    # ── 2) EXECUTION RISK DISTRIBUTION ────────────────────────────────
    st.markdown("### Execution risk distribution")
    risk_table = (
        _ei_value_counts_table(
            primary,
            "execution_risk_flag",
            order=["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
        )
        if primary is not None
        else None
    )
    if risk_table is None or risk_table.empty:
        st.caption("No `execution_risk_flag` column available.")
    else:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            _ei_render_table(risk_table)
        with col_b:
            try:
                chart_df = risk_table.set_index("execution_risk_flag")[["count"]]
                st.bar_chart(chart_df)
            except Exception as e:
                st.caption(f"Risk chart unavailable: {_short_err(e)}")

    # ── 3) EXECUTION QUALITY DISTRIBUTION ─────────────────────────────
    st.markdown("### Execution quality score")
    q_series = _ei_safe_numeric(primary, "execution_quality_score") if primary is not None else None
    if q_series is None or q_series.empty:
        st.caption("No `execution_quality_score` data available.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean", f"{q_series.mean():.3f}")
        m2.metric("Median", f"{q_series.median():.3f}")
        m3.metric("Min", f"{q_series.min():.3f}")
        m4.metric("Max", f"{q_series.max():.3f}")

        try:
            bins = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.000001]
            labels = [
                "0.00–0.15",
                "0.15–0.30",
                "0.30–0.45",
                "0.45–0.60",
                "0.60–0.75",
                "0.75–0.90",
                "0.90–1.00",
            ]
            binned = pd.cut(q_series, bins=bins, labels=labels, include_lowest=True, right=False)
            hist_df = binned.value_counts().reindex(labels).fillna(0).astype(int)
            hist_df = hist_df.rename_axis("score_bin").reset_index(name="count")
            col_h1, col_h2 = st.columns([1, 2])
            with col_h1:
                _ei_render_table(hist_df)
            with col_h2:
                st.bar_chart(hist_df.set_index("score_bin"))
        except Exception as e:
            st.caption(f"Histogram unavailable: {_short_err(e)}")

        # Optional: line chart of score by order sequence
        if primary is not None:
            tcol = _ei_pick_time_col(primary)
            if tcol is not None and "execution_quality_score" in primary.columns:
                try:
                    trend = primary[[tcol, "execution_quality_score"]].copy()
                    trend["execution_quality_score"] = pd.to_numeric(
                        trend["execution_quality_score"], errors="coerce"
                    )
                    trend = trend.dropna(subset=["execution_quality_score"])
                    if not trend.empty:
                        try:
                            trend[tcol] = pd.to_datetime(trend[tcol], errors="coerce", utc=True)
                            trend = trend.dropna(subset=[tcol])
                        except Exception:
                            pass
                        if not trend.empty:
                            trend = trend.sort_values(tcol).reset_index(drop=True)
                            st.line_chart(trend.set_index(tcol)[["execution_quality_score"]])
                except Exception as e:
                    st.caption(f"Quality trend chart unavailable: {_short_err(e)}")

    # ── 4) SPREAD / QUOTE QUALITY ─────────────────────────────────────
    st.markdown("### Spread & quote quality")
    spread_table = (
        _ei_value_counts_table(
            primary,
            "spread_bucket",
            order=["TIGHT", "NORMAL", "WIDE", "TOO_WIDE", "UNKNOWN"],
        )
        if primary is not None
        else None
    )
    stale_table = _ei_value_counts_table(primary, "quote_is_stale") if primary is not None else None

    sp_a, sp_b, sp_c = st.columns(3)
    sp_a.metric(
        "Avg spread (bps)",
        _ei_safe_metric(primary, "spread_bps", "mean", "{:.2f}") if primary is not None else "N/A",
    )
    sp_b.metric(
        "Avg quote age (s)",
        (
            _ei_safe_metric(primary, "quote_age_sec", "mean", "{:.1f}")
            if primary is not None
            else "N/A"
        ),
    )
    sp_c.metric("Stale quote rows", f"{stale_count:,}" if isinstance(stale_count, int) else "N/A")

    sb_col, st_col = st.columns(2)
    with sb_col:
        st.markdown("**Spread bucket counts**")
        _ei_render_table(spread_table, empty_msg="No `spread_bucket` data.")
    with st_col:
        st.markdown("**Quote-is-stale counts**")
        _ei_render_table(stale_table, empty_msg="No `quote_is_stale` data.")

    # ── 5) EXECUTION STYLE ───────────────────────────────────────────
    st.markdown("### Execution style")
    style_table = (
        _ei_value_counts_table(
            primary,
            "execution_style",
            order=["AGGRESSIVE_LIMIT", "NORMAL_LIMIT", "PASSIVE_LIMIT", "DEFER", "SKIP"],
        )
        if primary is not None
        else None
    )
    skipflag_table = (
        _ei_value_counts_table(primary, "execution_skip_flag") if primary is not None else None
    )

    es_a, es_b = st.columns(2)
    with es_a:
        st.markdown("**Style counts**")
        _ei_render_table(style_table, empty_msg="No `execution_style` data.")
    with es_b:
        st.markdown("**Skip flag counts**")
        _ei_render_table(skipflag_table, empty_msg="No `execution_skip_flag` data.")

    rs_a, rs_b = st.columns(2)
    with rs_a:
        st.markdown("**Top execution_reason values**")
        _ei_render_table(
            _ei_top_reason_table(primary, "execution_reason", 10),
            empty_msg="No `execution_reason` data.",
        )
    with rs_b:
        st.markdown("**Top execution_skip_reason values**")
        _ei_render_table(
            _ei_top_reason_table(primary, "execution_skip_reason", 10),
            empty_msg="No `execution_skip_reason` data.",
        )

    # ── 6) LIQUIDITY PRESSURE ────────────────────────────────────────
    st.markdown("### Liquidity pressure")
    if primary is None or primary.empty:
        st.caption("No primary EI source available.")
    else:
        l1, l2, l3 = st.columns(3)
        l1.metric(
            "Avg order_notional", _ei_safe_metric(primary, "order_notional", "mean", "{:,.0f}")
        )
        l2.metric(
            "Avg liquidity_proxy", _ei_safe_metric(primary, "liquidity_proxy", "mean", "{:,.0f}")
        )
        l3.metric(
            "Avg notional / liquidity",
            _ei_safe_metric(primary, "notional_vs_liquidity", "mean", "{:.4f}"),
        )

        if "notional_vs_liquidity" in primary.columns:
            try:
                worst = primary.copy()
                worst["__nvl"] = pd.to_numeric(worst["notional_vs_liquidity"], errors="coerce")
                worst = worst.dropna(subset=["__nvl"]).sort_values("__nvl", ascending=False)
                cols_top = _ei_select_existing_columns(
                    worst,
                    [
                        "symbol",
                        "side",
                        "qty",
                        "order_notional",
                        "liquidity_proxy",
                        "notional_vs_liquidity",
                        "spread_bucket",
                        "execution_style",
                        "execution_quality_score",
                        "execution_risk_flag",
                    ],
                )
                if cols_top:
                    st.markdown("**Top notional vs liquidity**")
                    _ei_render_table(
                        worst[cols_top].head(10), height=320, empty_msg="No rows to rank."
                    )
            except Exception as e:
                st.caption(f"Liquidity ranking unavailable: {_short_err(e)}")

    # ── 7) PARTIAL FILL INTELLIGENCE ─────────────────────────────────
    st.markdown("### Partial-fill follow-up (reprice_open_orders.csv)")
    if reprice_df is None:
        st.info("`data/results/reprice_open_orders.csv` not found.")
    elif reprice_df.empty:
        st.caption("No reprice / partial-fill rows yet.")
    else:
        pf_a, pf_b = st.columns(2)
        with pf_a:
            st.markdown("**partial_fill_action counts**")
            _ei_render_table(
                _ei_value_counts_table(
                    reprice_df,
                    "partial_fill_action",
                    order=["KEEP_WORKING", "REPRICE", "CANCEL", "DEFER_REPRICE"],
                ),
                empty_msg="No `partial_fill_action` column.",
            )
        with pf_b:
            st.markdown("**Top partial_fill_reason values**")
            _ei_render_table(
                _ei_top_reason_table(reprice_df, "partial_fill_reason", 10),
                empty_msg="No `partial_fill_reason` column.",
            )

        cols_pf = _ei_select_existing_columns(
            reprice_df,
            [
                "symbol",
                "side",
                "status",
                "fill_pct",
                "partial_fill_action",
                "partial_fill_reason",
                "spread_bucket",
                "spread_bps",
                "quote_is_stale",
                "quote_age_sec",
                "execution_quality_score",
                "execution_risk_flag",
            ],
        )
        if cols_pf:
            st.markdown("**Reprice rows (curated columns)**")
            _ei_render_table(reprice_df[cols_pf], height=360, empty_msg="No rows.")

    # ── 8) SLIPPAGE DIAGNOSTICS ──────────────────────────────────────
    st.markdown("### Slippage diagnostics")

    slip_sources: List[Tuple[str, Optional[pd.DataFrame]]] = [
        ("execution_intelligence.csv", ei_df),
        ("reprice_open_orders.csv", reprice_df),
        ("execution_plan.csv", plan_df),
    ]
    rendered_any_slip = False
    for src_label, src_df in slip_sources:
        if src_df is None or src_df.empty:
            continue
        if not any(c in src_df.columns for c in ("expected_slippage_bps", "realized_slippage_bps")):
            continue
        rendered_any_slip = True
        st.markdown(f"**Source:** `{src_label}`")
        s1, s2 = st.columns(2)
        s1.metric(
            "Avg expected slippage (bps)",
            _ei_safe_metric(src_df, "expected_slippage_bps", "mean", "{:.2f}"),
        )
        s2.metric(
            "Avg realized slippage (bps)",
            _ei_safe_metric(src_df, "realized_slippage_bps", "mean", "{:.2f}"),
        )

        if "realized_slippage_bps" in src_df.columns:
            try:
                w = src_df.copy()
                w["__rs"] = pd.to_numeric(w["realized_slippage_bps"], errors="coerce")
                w = w.dropna(subset=["__rs"]).sort_values("__rs", ascending=False)
                cols_slip = _ei_select_existing_columns(
                    w,
                    [
                        "symbol",
                        "side",
                        "qty",
                        "intended_price",
                        "submitted_limit_price",
                        "decision_mid_price",
                        "fill_price",
                        "expected_slippage_bps",
                        "realized_slippage_bps",
                        "execution_quality_score",
                        "execution_risk_flag",
                    ],
                )
                if cols_slip and not w.empty:
                    st.markdown("Top realized slippage rows")
                    _ei_render_table(
                        w[cols_slip].head(10), height=320, empty_msg="No realized slippage rows."
                    )
                    try:
                        rs_series = pd.to_numeric(
                            src_df["realized_slippage_bps"], errors="coerce"
                        ).dropna()
                        if not rs_series.empty:
                            st.markdown("Realized slippage distribution (bps)")
                            bins = [-100, -50, -20, -5, 0, 5, 20, 50, 100, 1e9]
                            labels = [
                                "<-50",
                                "-50..-20",
                                "-20..-5",
                                "-5..0",
                                "0..5",
                                "5..20",
                                "20..50",
                                "50..100",
                                ">100",
                            ]
                            binned = pd.cut(
                                rs_series,
                                bins=bins,
                                labels=labels,
                                include_lowest=True,
                                right=False,
                            )
                            sd = binned.value_counts().reindex(labels).fillna(0).astype(int)
                            sd = sd.rename_axis("bin").reset_index(name="count")
                            st.bar_chart(sd.set_index("bin"))
                    except Exception as e:
                        st.caption(f"Slippage chart unavailable: {_short_err(e)}")
            except Exception as e:
                st.caption(f"Slippage ranking unavailable: {_short_err(e)}")

    if not rendered_any_slip:
        st.caption("No slippage columns found in any EI source yet.")

    # ── 9) DETAILED RAW TABLES ───────────────────────────────────────
    st.markdown("### Detailed tables")

    curated_orders = [
        "timestamp",
        "submitted_at",
        "created_at",
        "generated_at",
        "session",
        "symbol",
        "action",
        "side",
        "qty",
        "status",
        "execution_quality_score",
        "execution_risk_flag",
        "execution_quality_reason",
        "execution_style",
        "execution_aggressiveness",
        "execution_reason",
        "execution_skip_flag",
        "execution_skip_reason",
        "spread_bucket",
        "spread_bps",
        "quote_is_stale",
        "quote_age_sec",
        "liquidity_proxy",
        "order_notional",
        "notional_vs_liquidity",
        "expected_slippage_bps",
        "realized_slippage_bps",
    ]
    curated_reprice = [
        "timestamp",
        "symbol",
        "side",
        "status",
        "fill_pct",
        "partial_fill_action",
        "partial_fill_reason",
        "spread_bucket",
        "spread_bps",
        "quote_is_stale",
        "quote_age_sec",
        "decision_mid_price",
        "expected_slippage_bps",
        "realized_slippage_bps",
        "execution_quality_score",
        "execution_risk_flag",
    ]

    with st.expander("Planned orders (execution_plan.csv)", expanded=False):
        if plan_df is None:
            st.info("`data/results/execution_plan.csv` not found.")
        elif plan_df.empty:
            st.caption("No planned orders in the current cycle.")
        else:
            cols_p = _ei_select_existing_columns(plan_df, curated_orders) or list(plan_df.columns)
            _ei_render_table(plan_df[cols_p], height=360)
            with st.expander("Show full raw planned-orders table", expanded=False):
                _ei_render_table(plan_df, height=360)

    with st.expander("Submitted orders (execution_intelligence.csv)", expanded=False):
        if ei_df is None:
            st.info("`data/results/execution_intelligence.csv` not found.")
        elif ei_df.empty:
            st.caption("No submitted-order EI rows yet.")
        else:
            cols_e = _ei_select_existing_columns(ei_df, curated_orders) or list(ei_df.columns)
            _ei_render_table(ei_df[cols_e], height=360)
            with st.expander("Show full raw submitted-orders table", expanded=False):
                _ei_render_table(ei_df, height=360)

    with st.expander("Reprice / Partial-fill follow-up (reprice_open_orders.csv)", expanded=False):
        if reprice_df is None:
            st.info("`data/results/reprice_open_orders.csv` not found.")
        elif reprice_df.empty:
            st.caption("No reprice rows in the current cycle.")
        else:
            cols_r = _ei_select_existing_columns(reprice_df, curated_reprice) or list(
                reprice_df.columns
            )
            _ei_render_table(reprice_df[cols_r], height=360)
            with st.expander("Show full raw reprice table", expanded=False):
                _ei_render_table(reprice_df, height=360)


# ──────────────────────────────
# FEEDBACK INTELLIGENCE PAGE
# (dashboard-only; reads feedback_loop_report.csv,
#  feedback_recommendations.csv, feedback_loop_summary.json)
# ──────────────────────────────

# Friendly labels for the aggregate-performance group keys emitted by
# services/feedback_loop.py — render only the ones that exist.
_FB_GROUP_LABELS: List[Tuple[str, str]] = [
    ("by_execution_risk_flag", "By execution risk flag"),
    ("by_spread_bucket", "By spread bucket"),
    ("by_execution_style", "By execution style"),
    ("by_sizing_bucket", "By sizing bucket"),
    ("by_signal", "By signal"),
    ("by_decision_action", "By decision action"),
    ("by_quote_is_stale", "By quote-is-stale"),
    ("by_liquidity_pressure_bucket", "By liquidity pressure bucket"),
    ("by_partial_fill_action", "By partial-fill action"),
    ("by_action", "By action"),
]


def _fb_load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Wrap dashboard CSV loader with a friendly per-file warning on hard errors."""
    df, err = read_csv_dashboard(path)
    if err is None:
        return df
    if err == "empty":
        return df
    st.warning(f"{label}: could not read {path.name} ({err}).")
    return None


def _fb_load_json(path: Path, label: str) -> Optional[Dict[str, Any]]:
    """Read a JSON file safely. Returns None on missing/error, dict otherwise."""
    try:
        if not path.exists():
            return None
        try:
            if path.stat().st_size == 0:
                return None
        except OSError:
            pass
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {"value": obj}
    except Exception as e:
        st.warning(f"{label}: could not read {path.name} ({_short_err(e)}).")
        return None


def _fb_select_existing_columns(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    return _ei_select_existing_columns(df, preferred)


def _fb_value_counts_table(
    df: pd.DataFrame, col: str, order: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    return _ei_value_counts_table(df, col, order=order)


def _fb_safe_metric(
    df: pd.DataFrame, col: str, agg: str = "mean", fmt: str = "{:.3f}", default: str = "N/A"
) -> str:
    return _ei_safe_metric(df, col, agg=agg, fmt=fmt, default=default)


def _fb_top_recommendations(df: pd.DataFrame, top_n: int = 10) -> Optional[pd.DataFrame]:
    """Return the top-N recommendations sorted by confidence then evidence count."""
    if df is None or df.empty:
        return None
    out = df.copy()
    if "recommendation_confidence" in out.columns:
        out["recommendation_confidence"] = pd.to_numeric(
            out["recommendation_confidence"], errors="coerce"
        )
    if "evidence_count" in out.columns:
        out["evidence_count"] = pd.to_numeric(out["evidence_count"], errors="coerce")
    sort_cols: List[str] = []
    if "recommendation_confidence" in out.columns:
        sort_cols.append("recommendation_confidence")
    if "evidence_count" in out.columns:
        sort_cols.append("evidence_count")
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False, na_position="last")
    return out.head(top_n).reset_index(drop=True)


def _fb_safe_json_table(obj: Optional[Dict[str, Any]], key: str) -> Optional[pd.DataFrame]:
    """
    Convert a nested dict-of-dicts (e.g. summary['aggregate_performance'][group])
    into a tidy table indexed by category name. Returns None if missing/empty.
    """
    if not obj or not isinstance(obj, dict):
        return None
    sub = obj.get(key)
    if not sub or not isinstance(sub, dict):
        return None
    try:
        df = pd.DataFrame(sub).T
        df.index.name = key.replace("by_", "")
        df = df.reset_index()
        for c in df.columns:
            if c == key.replace("by_", ""):
                continue
            try:
                coerced = pd.to_numeric(df[c], errors="coerce")
                # Only adopt the numeric cast if it didn't wipe every value;
                # otherwise leave the column as-is (e.g. for true-string fields).
                if coerced.notna().any():
                    df[c] = coerced
            except Exception:
                pass
        return df if not df.empty else None
    except Exception:
        return None


def _fb_summary_table_sources(summary: Dict[str, Any]) -> Optional[pd.DataFrame]:
    sa = summary.get("source_availability") if isinstance(summary, dict) else None
    if not sa or not isinstance(sa, dict):
        return None
    rows: List[Dict[str, Any]] = []
    for name, meta in sa.items():
        m = meta if isinstance(meta, dict) else {}
        rows.append(
            {
                "source": name,
                "status": str(m.get("status", "")),
                "rows": int(m.get("rows", 0) or 0),
                "path": str(m.get("path", "")),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["status", "source"], ascending=[True, True])
    return df.reset_index(drop=True)


def _fb_best_worst_from_groups(
    summary: Dict[str, Any],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Build flat best / worst tables across all aggregate-performance groups.

    Each row is one (group, category) pair. Returns (best_by_pnl, worst_by_pnl)
    sorted respectively. Returns (None, None) when no usable data exists.
    """
    if not summary or not isinstance(summary, dict):
        return None, None
    agg = summary.get("aggregate_performance") or {}
    if not isinstance(agg, dict) or not agg:
        return None, None

    rows: List[Dict[str, Any]] = []
    for group, table in agg.items():
        if not isinstance(table, dict):
            continue
        for category, stats in table.items():
            if not isinstance(stats, dict):
                continue
            rows.append(
                {
                    "group": str(group).replace("by_", ""),
                    "category": str(category),
                    "count": stats.get("count"),
                    "avg_pnl": stats.get("avg_pnl"),
                    "win_rate": stats.get("win_rate"),
                    "avg_realized_slippage_bps": stats.get("avg_realized_slippage_bps"),
                    "fill_rate": stats.get("fill_rate"),
                }
            )
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    for c in ("count", "avg_pnl", "win_rate", "avg_realized_slippage_bps", "fill_rate"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    pnl_df = df.dropna(subset=["avg_pnl"]).copy()
    if pnl_df.empty:
        return None, None
    best = pnl_df.sort_values("avg_pnl", ascending=False).head(10).reset_index(drop=True)
    worst = pnl_df.sort_values("avg_pnl", ascending=True).head(10).reset_index(drop=True)
    return best, worst


def page_feedback_intelligence() -> None:
    """🧠 Feedback Intelligence — observability of the feedback-loop layer."""
    st.title("🧠 Feedback Intelligence")
    st.caption(
        "Observed trade outcomes, execution conditions, evidence strength, "
        "and advisory recommendations from Triton's feedback loop."
    )

    report_df = _fb_load_csv(FEEDBACK_LOOP_REPORT_CSV_PATH, "Feedback report")
    rec_df = _fb_load_csv(FEEDBACK_RECOMMENDATIONS_CSV_PATH, "Feedback recommendations")
    summary = _fb_load_json(FEEDBACK_LOOP_SUMMARY_JSON_PATH, "Feedback summary")

    no_report = report_df is None or report_df.empty
    no_recs = rec_df is None or rec_df.empty
    no_summary = summary is None or not isinstance(summary, dict) or not summary

    if no_report and no_recs and no_summary:
        st.info(
            "No feedback-loop outputs found yet. "
            "Run `python -m services.feedback_loop` (or `python services/feedback_loop.py`) "
            "to populate `data/results/feedback_loop_report.csv`, "
            "`data/results/feedback_recommendations.csv`, and "
            "`data/results/feedback_loop_summary.json`."
        )

    # ── 1) TOP SUMMARY METRICS ───────────────────────────────────────
    st.markdown("### Summary")
    n_records = 0 if no_report else int(report_df.shape[0])
    n_recs = 0 if no_recs else int(rec_df.shape[0])

    high_evid_count = 0
    if not no_recs and "evidence_strength" in rec_df.columns:
        try:
            high_evid_count = int(
                rec_df["evidence_strength"].astype(str).str.upper().eq("HIGH").sum()
            )
        except Exception:
            high_evid_count = 0

    avg_conf = _fb_safe_metric(
        rec_df if not no_recs else pd.DataFrame(), "recommendation_confidence", "mean", "{:.2f}"
    )

    sources_avail = "N/A"
    missing_count: Optional[int] = None
    if isinstance(summary, dict):
        sa = summary.get("source_availability") or {}
        if isinstance(sa, dict) and sa:
            ok = sum(1 for v in sa.values() if isinstance(v, dict) and str(v.get("status")) == "ok")
            sources_avail = f"{ok}/{len(sa)}"
        missing = summary.get("missing_inputs")
        if isinstance(missing, list):
            missing_count = len(missing)

    fb_quality_high = 0
    if not no_report and "feedback_quality" in report_df.columns:
        try:
            fb_quality_high = int(
                report_df["feedback_quality"].astype(str).str.upper().eq("HIGH").sum()
            )
        except Exception:
            fb_quality_high = 0

    advisory_only = "N/A"
    if isinstance(summary, dict) and "advisory_only" in summary:
        advisory_only = "Yes" if bool(summary.get("advisory_only")) else "No"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feedback records", f"{n_records:,}")
    c2.metric("Recommendations", f"{n_recs:,}")
    c3.metric("HIGH evidence recs", f"{high_evid_count:,}")
    c4.metric("Avg recommendation conf.", avg_conf)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sources available", sources_avail)
    c6.metric(
        "Missing inputs",
        f"{missing_count:,}" if isinstance(missing_count, int) else "N/A",
    )
    c7.metric("HIGH-quality records", f"{fb_quality_high:,}")
    c8.metric("Advisory-only", advisory_only)

    if isinstance(summary, dict):
        gen_at = summary.get("generated_at_utc")
        spine = (
            (summary.get("record_counts") or {}).get("spine_source")
            if isinstance(summary.get("record_counts"), dict)
            else None
        )
        bits = []
        if gen_at:
            bits.append(f"generated_at_utc=`{gen_at}`")
        if spine:
            bits.append(f"spine=`{spine}`")
        if advisory_only != "N/A":
            bits.append(f"advisory_only=`{advisory_only}`")
        if bits:
            st.caption(" • ".join(bits))

    # ── 2) SOURCE AVAILABILITY ───────────────────────────────────────
    st.markdown("### Source availability")
    if no_summary:
        st.caption("No `feedback_loop_summary.json` available.")
    else:
        src_table = _fb_summary_table_sources(summary)
        if src_table is None or src_table.empty:
            st.caption("Summary JSON has no `source_availability` block.")
        else:
            _ei_render_table(src_table)
        missing = summary.get("missing_inputs") if isinstance(summary, dict) else None
        if isinstance(missing, list) and missing:
            st.warning("Missing or unreadable inputs: " + ", ".join(str(m) for m in missing))
        notes = summary.get("notes") if isinstance(summary, dict) else None
        if isinstance(notes, list) and notes:
            with st.expander("Summary notes", expanded=False):
                for n in notes:
                    st.write(f"• {n}")

    # ── 3) RECOMMENDATION OVERVIEW ───────────────────────────────────
    st.markdown("### Recommendation overview")
    if no_recs:
        st.info(
            "No recommendations have been emitted yet — `feedback_recommendations.csv` is empty."
        )
    else:
        type_table = _fb_value_counts_table(rec_df, "recommendation_type")
        evid_table = _fb_value_counts_table(
            rec_df,
            "evidence_strength",
            order=["LOW", "MEDIUM", "HIGH"],
        )

        col_t, col_e = st.columns(2)
        with col_t:
            st.markdown("**Counts by recommendation type**")
            _ei_render_table(type_table, empty_msg="No `recommendation_type` data.")
        with col_e:
            st.markdown("**Counts by evidence strength**")
            _ei_render_table(evid_table, empty_msg="No `evidence_strength` data.")

        # Avg confidence by recommendation_type
        if (
            "recommendation_type" in rec_df.columns
            and "recommendation_confidence" in rec_df.columns
        ):
            try:
                tmp = rec_df[["recommendation_type", "recommendation_confidence"]].copy()
                tmp["recommendation_confidence"] = pd.to_numeric(
                    tmp["recommendation_confidence"], errors="coerce"
                )
                conf_by_type = (
                    tmp.dropna(subset=["recommendation_confidence"])
                    .groupby("recommendation_type")["recommendation_confidence"]
                    .agg(["mean", "median", "count"])
                    .round(3)
                    .reset_index()
                    .rename(
                        columns={
                            "mean": "avg_confidence",
                            "median": "median_confidence",
                            "count": "count",
                        }
                    )
                    .sort_values("avg_confidence", ascending=False)
                    .reset_index(drop=True)
                )
                st.markdown("**Average confidence by recommendation type**")
                _ei_render_table(conf_by_type)
            except Exception as e:
                st.caption(f"Confidence-by-type unavailable: {_short_err(e)}")

        # Top recommendations
        st.markdown("**Top recommendations**")
        top_recs = _fb_top_recommendations(rec_df, top_n=10)
        curated = _fb_select_existing_columns(
            top_recs,
            [
                "recommendation_type",
                "recommendation_text",
                "evidence_count",
                "evidence_strength",
                "recommendation_confidence",
                "related_bucket",
                "related_flag",
                "related_style",
                "metric_snapshot",
            ],
        )
        view = top_recs[curated] if (top_recs is not None and curated) else top_recs
        _ei_render_table(view, height=320, empty_msg="No recommendations to rank.")

    # ── 4) EVIDENCE STRENGTH / CONFIDENCE ────────────────────────────
    st.markdown("### Evidence strength & confidence")
    if no_recs:
        st.caption("No recommendations to summarize.")
    else:
        evid_counts = _fb_value_counts_table(
            rec_df,
            "evidence_strength",
            order=["LOW", "MEDIUM", "HIGH"],
        )
        col_a, col_b = st.columns([1, 2])
        with col_a:
            _ei_render_table(evid_counts, empty_msg="No `evidence_strength` data.")
        with col_b:
            try:
                if evid_counts is not None and not evid_counts.empty:
                    chart_df = evid_counts.set_index("evidence_strength")[["count"]]
                    st.bar_chart(chart_df)
            except Exception as e:
                st.caption(f"Evidence chart unavailable: {_short_err(e)}")

        conf_series = _ei_safe_numeric(rec_df, "recommendation_confidence")
        if conf_series is None or conf_series.empty:
            st.caption("No `recommendation_confidence` data available.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean", f"{conf_series.mean():.3f}")
            m2.metric("Median", f"{conf_series.median():.3f}")
            m3.metric("Min", f"{conf_series.min():.3f}")
            m4.metric("Max", f"{conf_series.max():.3f}")

            try:
                bins = [0.0, 0.20, 0.40, 0.60, 0.80, 1.000001]
                labels = ["0.00–0.20", "0.20–0.40", "0.40–0.60", "0.60–0.80", "0.80–1.00"]
                binned = pd.cut(
                    conf_series, bins=bins, labels=labels, include_lowest=True, right=False
                )
                hist_df = (
                    binned.value_counts()
                    .reindex(labels)
                    .fillna(0)
                    .astype(int)
                    .rename_axis("confidence_bin")
                    .reset_index(name="count")
                )
                col_h1, col_h2 = st.columns([1, 2])
                with col_h1:
                    _ei_render_table(hist_df)
                with col_h2:
                    st.bar_chart(hist_df.set_index("confidence_bin"))
            except Exception as e:
                st.caption(f"Confidence histogram unavailable: {_short_err(e)}")

    # ── 5) AGGREGATE PERFORMANCE ─────────────────────────────────────
    st.markdown("### Aggregate performance")
    if no_summary:
        st.caption("No `feedback_loop_summary.json` available.")
    else:
        agg = summary.get("aggregate_performance") if isinstance(summary, dict) else None
        if not agg or not isinstance(agg, dict):
            st.caption("Summary JSON has no `aggregate_performance` block.")
        else:
            rendered_any = False
            for key, label in _FB_GROUP_LABELS:
                tbl = _fb_safe_json_table(agg, key)
                if tbl is None or tbl.empty:
                    continue
                rendered_any = True
                st.markdown(f"**{label}**")
                _ei_render_table(tbl)
                if "avg_pnl" in tbl.columns:
                    try:
                        cat_col = tbl.columns[0]
                        chart = tbl[[cat_col, "avg_pnl"]].copy()
                        chart["avg_pnl"] = pd.to_numeric(chart["avg_pnl"], errors="coerce")
                        chart = chart.dropna(subset=["avg_pnl"])
                        if not chart.empty:
                            st.bar_chart(chart.set_index(cat_col)[["avg_pnl"]])
                    except Exception as e:
                        st.caption(f"Chart unavailable for {label}: {_short_err(e)}")
            if not rendered_any:
                st.caption("Aggregate-performance block was present but empty.")

    # ── 6) BEST / WORST CONDITIONS ───────────────────────────────────
    st.markdown("### Best & worst observed conditions")
    if no_summary:
        st.caption("No summary JSON; cannot rank best/worst conditions.")
    else:
        best, worst = _fb_best_worst_from_groups(summary)
        if best is None and worst is None:
            st.caption("Not enough PnL data across groups to rank best/worst conditions.")
        else:
            col_b, col_w = st.columns(2)
            with col_b:
                st.markdown("**Best (highest avg PnL)**")
                _ei_render_table(best, empty_msg="No best-condition rows.")
            with col_w:
                st.markdown("**Worst (lowest avg PnL)**")
                _ei_render_table(worst, empty_msg="No worst-condition rows.")

        if not no_recs and "recommendation_type" in rec_df.columns:
            POS = {"EDGE_VALIDATION", "SIGNAL_TRUST_BOOST"}
            try:
                tmp = rec_df.copy()
                tmp["recommendation_confidence"] = pd.to_numeric(
                    tmp.get("recommendation_confidence"), errors="coerce"
                )
                tmp["evidence_count"] = pd.to_numeric(tmp.get("evidence_count"), errors="coerce")
                pos = tmp[tmp["recommendation_type"].astype(str).isin(POS)]
                neg = tmp[~tmp["recommendation_type"].astype(str).isin(POS)]
                if not pos.empty:
                    sp = pos.sort_values(
                        ["recommendation_confidence", "evidence_count"],
                        ascending=False,
                        na_position="last",
                    ).iloc[0]
                    st.success(
                        f"**Strongest positive recommendation** "
                        f"({sp.get('recommendation_type','?')} · "
                        f"conf={sp.get('recommendation_confidence','?')} · "
                        f"n={sp.get('evidence_count','?')}): "
                        f"{sp.get('recommendation_text','')}"
                    )
                if not neg.empty:
                    sn = neg.sort_values(
                        ["recommendation_confidence", "evidence_count"],
                        ascending=False,
                        na_position="last",
                    ).iloc[0]
                    st.warning(
                        f"**Strongest caution recommendation** "
                        f"({sn.get('recommendation_type','?')} · "
                        f"conf={sn.get('recommendation_confidence','?')} · "
                        f"n={sn.get('evidence_count','?')}): "
                        f"{sn.get('recommendation_text','')}"
                    )
            except Exception as e:
                st.caption(f"Strongest-recommendation rendering unavailable: {_short_err(e)}")

    # ── 7) FEEDBACK QUALITY ──────────────────────────────────────────
    st.markdown("### Feedback quality")
    if no_report:
        st.info("No `feedback_loop_report.csv` rows to summarize quality.")
    else:
        col_q, col_s = st.columns(2)
        with col_q:
            st.markdown("**Counts by feedback_quality**")
            _ei_render_table(
                _fb_value_counts_table(
                    report_df,
                    "feedback_quality",
                    order=["LOW", "MEDIUM", "HIGH"],
                ),
                empty_msg="No `feedback_quality` column.",
            )
        with col_s:
            st.markdown("**Counts by spine_source**")
            _ei_render_table(
                _fb_value_counts_table(report_df, "spine_source"),
                empty_msg="No `spine_source` column.",
            )

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(
            "Avg matched_sources_count",
            _fb_safe_metric(report_df, "matched_sources_count", "mean", "{:.2f}"),
        )
        col_m2.metric(
            "Median matched_sources_count",
            _fb_safe_metric(report_df, "matched_sources_count", "median", "{:.1f}"),
        )
        col_m3.metric("Records", f"{int(report_df.shape[0]):,}")

        st.markdown("**Recent feedback rows**")
        curated_report = _fb_select_existing_columns(
            report_df,
            [
                "symbol",
                "date",
                "session",
                "side",
                "action",
                "signal",
                "decision_action",
                "execution_style",
                "execution_risk_flag",
                "execution_quality_score",
                "spread_bucket",
                "quote_is_stale",
                "partial_fill_action",
                "pnl",
                "pnl_pct",
                "feedback_quality",
                "matched_sources_count",
                "spine_source",
            ],
        )
        view = report_df[curated_report] if curated_report else report_df
        try:
            if "date" in view.columns:
                view = view.copy()
                view["__sort"] = pd.to_datetime(view["date"], errors="coerce", utc=True)
                view = view.sort_values("__sort", ascending=False, na_position="last")
                view = view.drop(columns=["__sort"])
        except Exception:
            pass
        _ei_render_table(view.head(50), height=380, empty_msg="No report rows available.")

    # ── 8) RECOMMENDATIONS TABLE (curated + raw) ─────────────────────
    st.markdown("### Recommendations table")
    if no_recs:
        st.caption("No recommendations to display.")
    else:
        curated_recs = _fb_select_existing_columns(
            rec_df,
            [
                "recommendation_type",
                "recommendation_text",
                "evidence_count",
                "evidence_strength",
                "recommendation_confidence",
                "related_bucket",
                "related_flag",
                "related_style",
                "metric_snapshot",
            ],
        )
        view = rec_df[curated_recs] if curated_recs else rec_df
        try:
            view = view.copy()
            if "recommendation_confidence" in view.columns:
                view["recommendation_confidence"] = pd.to_numeric(
                    view["recommendation_confidence"], errors="coerce"
                )
            if "evidence_count" in view.columns:
                view["evidence_count"] = pd.to_numeric(view["evidence_count"], errors="coerce")
            sort_cols = [
                c for c in ("recommendation_confidence", "evidence_count") if c in view.columns
            ]
            if sort_cols:
                view = view.sort_values(sort_cols, ascending=False, na_position="last")
        except Exception:
            pass
        _ei_render_table(view, height=420)
        with st.expander("Show full raw recommendations table", expanded=False):
            _ei_render_table(rec_df, height=360)

    # ── 9) DETAILED DATA EXPANDERS ───────────────────────────────────
    st.markdown("### Raw data")
    with st.expander("Feedback Report (raw)", expanded=False):
        if no_report:
            st.caption("`feedback_loop_report.csv` is empty or missing.")
        else:
            st.caption(f"{report_df.shape[0]:,} rows × {report_df.shape[1]:,} columns")
            _ei_render_table(report_df, height=420)

    with st.expander("Feedback Recommendations (raw)", expanded=False):
        if no_recs:
            st.caption("`feedback_recommendations.csv` is empty or missing.")
        else:
            st.caption(f"{rec_df.shape[0]:,} rows × {rec_df.shape[1]:,} columns")
            _ei_render_table(rec_df, height=360)

    with st.expander("Feedback Summary (key sections)", expanded=False):
        if no_summary:
            st.caption("`feedback_loop_summary.json` is empty or missing.")
        else:
            try:
                meta = {
                    k: summary.get(k)
                    for k in (
                        "generated_at_utc",
                        "schema_version",
                        "advisory_only",
                        "missing_inputs",
                        "record_counts",
                        "recommendation_counts_by_type",
                        "notes",
                    )
                    if k in summary
                }
                st.json(meta)
                with st.expander("Full summary JSON", expanded=False):
                    st.json(summary)
            except Exception as e:
                st.caption(f"Summary rendering unavailable: {_short_err(e)}")


# ──────────────────────────────
# ADAPTATION INTELLIGENCE PAGE
# (dashboard-only; reads adaptation_proposals.csv,
#  adaptation_review_queue.csv, adaptation_summary.json)
# ──────────────────────────────


def _ad_load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Wrap dashboard CSV loader with a friendly per-file warning on hard errors."""
    df, err = read_csv_dashboard(path)
    if err is None:
        return df
    if err == "empty":
        return df
    st.warning(f"{label}: could not read {path.name} ({err}).")
    return None


def _ad_load_json(path: Path, label: str) -> Optional[Dict[str, Any]]:
    """Read a JSON file safely. Returns None on missing/error, dict otherwise."""
    try:
        if not path.exists():
            return None
        try:
            if path.stat().st_size == 0:
                return None
        except OSError:
            pass
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {"value": obj}
    except Exception as e:
        st.warning(f"{label}: could not read {path.name} ({_short_err(e)}).")
        return None


# Re-use the well-tested EI helpers — same contracts (df-shape-agnostic,
# column-existence-aware, never raises). Thin wrappers keep call-sites readable.
def _ad_select_existing_columns(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    return _ei_select_existing_columns(df, preferred)


def _ad_value_counts_table(
    df: pd.DataFrame, col: str, order: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    return _ei_value_counts_table(df, col, order=order)


def _ad_safe_metric(
    df: pd.DataFrame, col: str, agg: str = "mean", fmt: str = "{:.3f}", default: str = "N/A"
) -> str:
    return _ei_safe_metric(df, col, agg=agg, fmt=fmt, default=default)


def _ad_truthy_count(df: pd.DataFrame, col: str) -> Optional[int]:
    return _ei_truthy_count(df, col)


def _ad_safe_json_table(obj: Optional[Dict[str, Any]], key: str) -> Optional[pd.DataFrame]:
    """
    Generic JSON-section flattener. Handles the two shapes the adaptation
    summary actually emits:

      - {<category>: {<metric>: value, ...}}        → multi-column tidy table
      - {<category>: <int>}                         → 2-column count table
    """
    if not obj or not isinstance(obj, dict):
        return None
    sub = obj.get(key)
    if sub is None:
        return None
    try:
        if isinstance(sub, dict):
            if not sub:
                return None
            sample = next(iter(sub.values()))
            if isinstance(sample, dict):
                df = pd.DataFrame(sub).T
                df.index.name = key.replace("by_", "")
                df = df.reset_index()
                for c in df.columns:
                    if c == key.replace("by_", ""):
                        continue
                    try:
                        coerced = pd.to_numeric(df[c], errors="coerce")
                        if coerced.notna().any():
                            df[c] = coerced
                    except Exception:
                        pass
                return df if not df.empty else None
            # Flat dict-of-scalars (e.g. count-by-target)
            df = pd.Series(sub, name="count").rename_axis(key.replace("by_", "")).reset_index()
            try:
                df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
            except Exception:
                pass
            return df if not df.empty else None
        if isinstance(sub, list):
            df = pd.DataFrame(sub)
            return df if not df.empty else None
    except Exception:
        return None
    return None


def _ad_top_proposals(df: pd.DataFrame, top_n: int = 10) -> Optional[pd.DataFrame]:
    """Top-N proposals: priority desc → confidence desc → evidence desc, thin-data last."""
    if df is None or df.empty:
        return None
    out = df.copy()
    pri_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "": 3}
    if "review_priority" in out.columns:
        out["__pri"] = (
            out["review_priority"].astype(str).str.upper().map(lambda p: pri_order.get(p, 3))
        )
    else:
        out["__pri"] = 3
    if "thin_data_flag" in out.columns:
        out["__thin"] = (
            out["thin_data_flag"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "t"])
            .map(lambda b: 1 if b else 0)
        )
    else:
        out["__thin"] = 0
    if "proposal_confidence" in out.columns:
        out["__conf"] = pd.to_numeric(out["proposal_confidence"], errors="coerce").fillna(0.0)
    else:
        out["__conf"] = 0.0
    if "evidence_count" in out.columns:
        out["__evid"] = pd.to_numeric(out["evidence_count"], errors="coerce").fillna(0)
    else:
        out["__evid"] = 0
    out = out.sort_values(
        ["__thin", "__pri", "__conf", "__evid"],
        ascending=[True, True, False, False],
    )
    out = out.drop(columns=["__pri", "__thin", "__conf", "__evid"])
    return out.head(top_n).reset_index(drop=True)


def _ad_summary_table_sources(summary: Dict[str, Any]) -> Optional[pd.DataFrame]:
    sa = summary.get("source_availability") if isinstance(summary, dict) else None
    if not sa or not isinstance(sa, dict):
        return None
    rows: List[Dict[str, Any]] = []
    for name, meta in sa.items():
        m = meta if isinstance(meta, dict) else {}
        rows.append(
            {
                "source": name,
                "status": str(m.get("status", "")),
                "rows": int(m.get("rows", 0) or 0),
                "path": str(m.get("path", "")),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["status", "source"], ascending=[True, True])
    return df.reset_index(drop=True)


# Friendly labels for the adaptation summary's `*_by_*` count blocks.
_AD_GROUP_LABELS: List[Tuple[str, str]] = [
    ("proposal_count_by_target", "By adaptation target"),
    ("proposal_count_by_type", "By proposal type"),
    ("proposal_count_by_priority", "By review priority"),
    ("proposal_count_by_evidence_strength", "By evidence strength"),
]

# Direction-set heuristics for "best/strongest" splits.
_AD_CAUTION_DIRECTIONS: set = {"DECREASE"}
_AD_CAUTION_TYPES: set = {
    "DECREASE_TRUST",
    "DECREASE_AGGRESSIVENESS",
    "INCREASE_PENALTY",
    "INCREASE_CAUTION",
    "ADJUST_BUCKET",
}
_AD_POSITIVE_TYPES: set = {
    "MAINTAIN_OR_SLIGHTLY_INCREASE",
    "ADJUST_SIGNAL_TRUST",  # split further by direction below
}


def page_adaptation_intelligence() -> None:
    """🛠️ Adaptation Intelligence — observability of advisory adaptation proposals."""
    st.title("🛠️ Adaptation Intelligence")
    st.caption(
        "Advisory-only adaptation proposals, confidence, evidence, guardrails, "
        "and review priority from Triton's controlled adaptation layer."
    )

    proposals = _ad_load_csv(ADAPTATION_PROPOSALS_CSV_PATH, "Adaptation proposals")
    review_q = _ad_load_csv(ADAPTATION_REVIEW_QUEUE_CSV_PATH, "Adaptation review queue")
    summary = _ad_load_json(ADAPTATION_SUMMARY_JSON_PATH, "Adaptation summary")

    no_props = proposals is None or proposals.empty
    no_rq = review_q is None or review_q.empty
    no_summary = summary is None or not isinstance(summary, dict) or not summary

    if no_props and no_rq and no_summary:
        st.info(
            "No adaptation-layer outputs found yet. "
            "Run `python -m services.adaptation_layer` (or `python services/adaptation_layer.py`) "
            "to populate `data/results/adaptation_proposals.csv`, "
            "`data/results/adaptation_review_queue.csv`, and "
            "`data/results/adaptation_summary.json`."
        )

    # ── 1) TOP SUMMARY METRICS ───────────────────────────────────────
    st.markdown("### Summary")
    n_props = 0 if no_props else int(proposals.shape[0])

    high_pri_count = 0
    if not no_props and "review_priority" in proposals.columns:
        try:
            high_pri_count = int(
                proposals["review_priority"].astype(str).str.upper().eq("HIGH").sum()
            )
        except Exception:
            high_pri_count = 0

    high_evid_count = 0
    if not no_props and "evidence_strength" in proposals.columns:
        try:
            high_evid_count = int(
                proposals["evidence_strength"].astype(str).str.upper().eq("HIGH").sum()
            )
        except Exception:
            high_evid_count = 0

    avg_conf = _ad_safe_metric(
        proposals if not no_props else pd.DataFrame(),
        "proposal_confidence",
        "mean",
        "{:.2f}",
    )

    thin_count = _ad_truthy_count(proposals, "thin_data_flag") if not no_props else None
    advisory_only = "N/A"
    auto_apply = "N/A"
    phase_str = "N/A"
    if isinstance(summary, dict):
        if "advisory_only" in summary:
            advisory_only = "Yes" if bool(summary.get("advisory_only")) else "No"
        if "auto_apply_allowed" in summary:
            auto_apply = "Yes" if bool(summary.get("auto_apply_allowed")) else "No"
        ph = summary.get("phase")
        if ph:
            phase_str = str(ph)

    sources_avail = "N/A"
    if isinstance(summary, dict):
        sa = summary.get("source_availability") or {}
        if isinstance(sa, dict) and sa:
            ok = sum(1 for v in sa.values() if isinstance(v, dict) and str(v.get("status")) == "ok")
            sources_avail = f"{ok}/{len(sa)}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Proposals", f"{n_props:,}")
    c2.metric("HIGH priority", f"{high_pri_count:,}")
    c3.metric("HIGH evidence", f"{high_evid_count:,}")
    c4.metric("Avg confidence", avg_conf)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Thin-data proposals", f"{thin_count:,}" if isinstance(thin_count, int) else "N/A")
    c6.metric("Advisory-only", advisory_only)
    c7.metric("Auto-apply allowed", auto_apply)
    c8.metric("Sources available", sources_avail)

    if isinstance(summary, dict):
        gen_at = summary.get("generated_at_utc")
        bits = []
        if gen_at:
            bits.append(f"generated_at_utc=`{gen_at}`")
        if phase_str != "N/A":
            bits.append(f"phase=`{phase_str}`")
        if bits:
            st.caption(" • ".join(bits))

    # ── 2) GOVERNANCE STATUS ─────────────────────────────────────────
    st.markdown("### Governance status")
    if no_summary:
        st.caption("No `adaptation_summary.json` available — governance status unknown.")
    else:
        # Always make the review-only nature explicit, even when fields missing.
        gov_msgs: List[str] = []
        if advisory_only == "Yes":
            gov_msgs.append("`advisory_only = True` — proposals are review-only.")
        if auto_apply == "No":
            gov_msgs.append(
                "`auto_apply_allowed = False` — no proposal will be applied automatically."
            )
        if phase_str != "N/A":
            gov_msgs.append(f"Phase: `{phase_str}`.")
        if gov_msgs:
            st.success(" ".join(gov_msgs))
        else:
            st.info("Governance flags missing from summary — treat all proposals as review-only.")

        missing = summary.get("missing_inputs") if isinstance(summary, dict) else None
        if isinstance(missing, list) and missing:
            st.warning(
                f"Missing or unreadable inputs ({len(missing)}): "
                + ", ".join(str(m) for m in missing)
            )
        notes = summary.get("notes") if isinstance(summary, dict) else None
        if isinstance(notes, list) and notes:
            with st.expander("Summary notes", expanded=False):
                for n in notes:
                    st.write(f"• {n}")

    # ── 3) SOURCE AVAILABILITY ───────────────────────────────────────
    st.markdown("### Source availability")
    if no_summary:
        st.caption("No `adaptation_summary.json` available.")
    else:
        src_table = _ad_summary_table_sources(summary)
        if src_table is None or src_table.empty:
            st.caption("Summary JSON has no `source_availability` block.")
        else:
            _ei_render_table(src_table)

    # ── 4) PROPOSAL OVERVIEW ─────────────────────────────────────────
    st.markdown("### Proposal overview")
    if no_props:
        st.info("No proposals to summarize — `adaptation_proposals.csv` is empty.")
    else:
        col_t, col_p = st.columns(2)
        with col_t:
            st.markdown("**Counts by adaptation target**")
            _ei_render_table(
                _ad_value_counts_table(proposals, "adaptation_target"),
                empty_msg="No `adaptation_target` data.",
            )
        with col_p:
            st.markdown("**Counts by proposal type**")
            _ei_render_table(
                _ad_value_counts_table(proposals, "proposal_type"),
                empty_msg="No `proposal_type` data.",
            )

        col_r, col_e = st.columns(2)
        with col_r:
            st.markdown("**Counts by review priority**")
            _ei_render_table(
                _ad_value_counts_table(
                    proposals,
                    "review_priority",
                    order=["HIGH", "MEDIUM", "LOW"],
                ),
                empty_msg="No `review_priority` data.",
            )
        with col_e:
            st.markdown("**Counts by evidence strength**")
            _ei_render_table(
                _ad_value_counts_table(
                    proposals,
                    "evidence_strength",
                    order=["LOW", "MEDIUM", "HIGH"],
                ),
                empty_msg="No `evidence_strength` data.",
            )

        if "adaptation_target" in proposals.columns and "proposal_confidence" in proposals.columns:
            try:
                tmp = proposals[["adaptation_target", "proposal_confidence"]].copy()
                tmp["proposal_confidence"] = pd.to_numeric(
                    tmp["proposal_confidence"], errors="coerce"
                )
                conf_by_target = (
                    tmp.dropna(subset=["proposal_confidence"])
                    .groupby("adaptation_target")["proposal_confidence"]
                    .agg(["mean", "median", "count"])
                    .round(3)
                    .reset_index()
                    .rename(columns={"mean": "avg_confidence", "median": "median_confidence"})
                    .sort_values("avg_confidence", ascending=False)
                    .reset_index(drop=True)
                )
                st.markdown("**Average confidence by adaptation target**")
                _ei_render_table(conf_by_target)
            except Exception as e:
                st.caption(f"Confidence-by-target unavailable: {_short_err(e)}")

    # ── 5) PROPOSAL CONFIDENCE & EVIDENCE ────────────────────────────
    st.markdown("### Proposal confidence & evidence")
    if no_props:
        st.caption("No proposals to summarize.")
    else:
        evid_counts = _ad_value_counts_table(
            proposals,
            "evidence_strength",
            order=["LOW", "MEDIUM", "HIGH"],
        )
        col_a, col_b = st.columns([1, 2])
        with col_a:
            _ei_render_table(evid_counts, empty_msg="No `evidence_strength` data.")
        with col_b:
            try:
                if evid_counts is not None and not evid_counts.empty:
                    st.bar_chart(evid_counts.set_index("evidence_strength")[["count"]])
            except Exception as e:
                st.caption(f"Evidence chart unavailable: {_short_err(e)}")

        conf_series = _ei_safe_numeric(proposals, "proposal_confidence")
        if conf_series is None or conf_series.empty:
            st.caption("No `proposal_confidence` data available.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean", f"{conf_series.mean():.3f}")
            m2.metric("Median", f"{conf_series.median():.3f}")
            m3.metric("Min", f"{conf_series.min():.3f}")
            m4.metric("Max", f"{conf_series.max():.3f}")

            try:
                bins = [0.0, 0.20, 0.40, 0.60, 0.80, 1.000001]
                labels = ["0.00–0.20", "0.20–0.40", "0.40–0.60", "0.60–0.80", "0.80–1.00"]
                binned = pd.cut(
                    conf_series, bins=bins, labels=labels, include_lowest=True, right=False
                )
                hist_df = (
                    binned.value_counts()
                    .reindex(labels)
                    .fillna(0)
                    .astype(int)
                    .rename_axis("confidence_bin")
                    .reset_index(name="count")
                )
                col_h1, col_h2 = st.columns([1, 2])
                with col_h1:
                    _ei_render_table(hist_df)
                with col_h2:
                    st.bar_chart(hist_df.set_index("confidence_bin"))
            except Exception as e:
                st.caption(f"Confidence histogram unavailable: {_short_err(e)}")

        thin_table = _ad_value_counts_table(proposals, "thin_data_flag")
        st.markdown("**Thin-data flag counts**")
        _ei_render_table(thin_table, empty_msg="No `thin_data_flag` data.")

    # ── 6) GUARDRAILS ────────────────────────────────────────────────
    st.markdown("### Guardrails")
    if no_props:
        st.caption("No proposals to inspect for guardrails.")
    else:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("**bounded_change_applied counts**")
            _ei_render_table(
                _ad_value_counts_table(proposals, "bounded_change_applied"),
                empty_msg="No `bounded_change_applied` column.",
            )
            st.markdown("**requires_manual_review counts**")
            _ei_render_table(
                _ad_value_counts_table(proposals, "requires_manual_review"),
                empty_msg="No `requires_manual_review` column.",
            )
        with col_g2:
            st.markdown("**auto_apply_allowed counts**")
            _ei_render_table(
                _ad_value_counts_table(proposals, "auto_apply_allowed"),
                empty_msg="No `auto_apply_allowed` column.",
            )
            st.markdown("**proposal_direction counts**")
            _ei_render_table(
                _ad_value_counts_table(proposals, "proposal_direction"),
                empty_msg="No `proposal_direction` column.",
            )

        # Bounded-vs-unbounded summary metrics
        bounded_n = _ad_truthy_count(proposals, "bounded_change_applied")
        review_n = _ad_truthy_count(proposals, "requires_manual_review")
        autoapply_n = _ad_truthy_count(proposals, "auto_apply_allowed")
        gm1, gm2, gm3 = st.columns(3)
        gm1.metric("Bounded changes", f"{bounded_n:,}" if isinstance(bounded_n, int) else "N/A")
        gm2.metric("Require manual review", f"{review_n:,}" if isinstance(review_n, int) else "N/A")
        gm3.metric("Auto-apply rows", f"{autoapply_n:,}" if isinstance(autoapply_n, int) else "N/A")

        st.markdown("**Curated guardrail view**")
        guard_cols = _ad_select_existing_columns(
            proposals,
            [
                "adaptation_target",
                "proposal_type",
                "proposed_delta",
                "min_allowed_value",
                "max_allowed_value",
                "bounded_change_applied",
                "requires_manual_review",
                "auto_apply_allowed",
                "advisory_only",
            ],
        )
        view = proposals[guard_cols] if guard_cols else proposals
        _ei_render_table(view, height=320, empty_msg="No proposal rows to display.")

    # ── 7) REVIEW QUEUE ──────────────────────────────────────────────
    st.markdown("### Review queue")
    if no_rq:
        st.info(
            "No `adaptation_review_queue.csv` rows. The review queue is "
            "regenerated each time the adaptation layer runs."
        )
    else:
        st.markdown("**Counts by review priority**")
        _ei_render_table(
            _ad_value_counts_table(
                review_q,
                "review_priority",
                order=["HIGH", "MEDIUM", "LOW"],
            ),
            empty_msg="No `review_priority` data.",
        )

        st.markdown("**Top of the review queue**")
        rq_curated = _ad_select_existing_columns(
            review_q,
            [
                "proposal_id",
                "adaptation_target",
                "proposal_type",
                "proposal_direction",
                "proposal_strength",
                "proposal_confidence",
                "evidence_count",
                "evidence_strength",
                "review_priority",
                "thin_data_flag",
                "proposal_reason",
                "related_bucket",
                "related_flag",
                "related_style",
            ],
        )
        view = review_q[rq_curated] if rq_curated else review_q
        sorted_view = _ad_top_proposals(view, top_n=25)
        if sorted_view is None or sorted_view.empty:
            sorted_view = view
        _ei_render_table(sorted_view, height=420, empty_msg="No queue entries to display.")

    # ── 8) PROPOSAL DETAIL / EXPLANATION ─────────────────────────────
    st.markdown("### Proposal detail & explanation")
    if no_props:
        st.caption("No proposals to detail.")
    else:
        detail_cols = _ad_select_existing_columns(
            proposals,
            [
                "adaptation_target",
                "proposal_type",
                "recommendation_type",
                "source_recommendation_text",
                "proposal_reason",
                "proposal_note",
                "observed_group",
                "observed_metric",
                "observed_value",
                "baseline_value",
                "effect_direction",
                "current_value",
                "proposed_value",
                "proposed_delta",
            ],
        )
        view = proposals[detail_cols] if detail_cols else proposals
        _ei_render_table(view, height=420, empty_msg="No proposal detail rows.")

    # ── 9) BEST / STRONGEST PROPOSALS ────────────────────────────────
    st.markdown("### Highest-confidence & strongest proposals")
    if no_props:
        st.caption("No proposals to rank.")
    else:
        # Highest-confidence
        try:
            hc = proposals.copy()
            hc["proposal_confidence"] = pd.to_numeric(
                hc.get("proposal_confidence"), errors="coerce"
            )
            hc = hc.dropna(subset=["proposal_confidence"]).sort_values(
                "proposal_confidence", ascending=False
            )
            hc_curated = _ad_select_existing_columns(
                hc,
                [
                    "adaptation_target",
                    "proposal_type",
                    "proposal_direction",
                    "proposal_confidence",
                    "evidence_strength",
                    "evidence_count",
                    "review_priority",
                    "thin_data_flag",
                    "related_bucket",
                    "related_flag",
                    "related_style",
                ],
            )
            view = hc[hc_curated] if hc_curated else hc
            st.markdown("**Top by proposal confidence**")
            _ei_render_table(view.head(10), empty_msg="No ranked proposals to show.")
        except Exception as e:
            st.caption(f"Highest-confidence ranking unavailable: {_short_err(e)}")

        # Strongest caution / strongest positive
        try:
            tmp = proposals.copy()
            ptype = tmp.get("proposal_type", pd.Series([], dtype=object))
            pdir = tmp.get("proposal_direction", pd.Series([], dtype=object))
            pdelta = pd.to_numeric(tmp.get("proposed_delta"), errors="coerce")
            pconf = pd.to_numeric(tmp.get("proposal_confidence"), errors="coerce")

            type_upper = ptype.astype(str).str.upper()
            dir_upper = pdir.astype(str).str.upper()

            caution_mask = type_upper.isin(_AD_CAUTION_TYPES) | dir_upper.isin(
                _AD_CAUTION_DIRECTIONS
            )
            positive_mask = type_upper.eq("MAINTAIN_OR_SLIGHTLY_INCREASE") | (
                type_upper.eq("ADJUST_SIGNAL_TRUST") & (pdelta.fillna(0) > 0)
            )

            curated_strong = _ad_select_existing_columns(
                tmp,
                [
                    "adaptation_target",
                    "proposal_type",
                    "proposal_direction",
                    "proposed_delta",
                    "proposal_confidence",
                    "evidence_strength",
                    "evidence_count",
                    "review_priority",
                    "thin_data_flag",
                    "related_bucket",
                    "related_flag",
                    "related_style",
                ],
            )

            col_caut, col_pos = st.columns(2)
            with col_caut:
                st.markdown("**Strongest caution proposals**")
                caut = tmp[caution_mask].copy()
                if "proposal_confidence" in caut.columns:
                    caut["__conf"] = pd.to_numeric(
                        caut["proposal_confidence"], errors="coerce"
                    ).fillna(0.0)
                    caut["__abs_delta"] = (
                        pd.to_numeric(caut.get("proposed_delta"), errors="coerce").abs().fillna(0.0)
                    )
                    caut = caut.sort_values(["__conf", "__abs_delta"], ascending=False).drop(
                        columns=["__conf", "__abs_delta"]
                    )
                view = caut[curated_strong] if curated_strong else caut
                _ei_render_table(view.head(8), empty_msg="No caution-type proposals.")
            with col_pos:
                st.markdown("**Strongest positive proposals**")
                pos = tmp[positive_mask].copy()
                if "proposal_confidence" in pos.columns:
                    pos["__conf"] = pd.to_numeric(
                        pos["proposal_confidence"], errors="coerce"
                    ).fillna(0.0)
                    pos["__delta"] = pd.to_numeric(
                        pos.get("proposed_delta"), errors="coerce"
                    ).fillna(0.0)
                    pos = pos.sort_values(["__conf", "__delta"], ascending=False).drop(
                        columns=["__conf", "__delta"]
                    )
                view = pos[curated_strong] if curated_strong else pos
                _ei_render_table(view.head(8), empty_msg="No positive-type proposals.")
        except Exception as e:
            st.caption(f"Caution/positive split unavailable: {_short_err(e)}")

    # ── 10) DETAILED TABLES ──────────────────────────────────────────
    st.markdown("### Raw data")
    with st.expander("Adaptation Proposals (raw)", expanded=False):
        if no_props:
            st.caption("`adaptation_proposals.csv` is empty or missing.")
        else:
            st.caption(f"{proposals.shape[0]:,} rows × {proposals.shape[1]:,} columns")
            _ei_render_table(proposals, height=420)

    with st.expander("Adaptation Review Queue (raw)", expanded=False):
        if no_rq:
            st.caption("`adaptation_review_queue.csv` is empty or missing.")
        else:
            st.caption(f"{review_q.shape[0]:,} rows × {review_q.shape[1]:,} columns")
            _ei_render_table(review_q, height=360)

    with st.expander("Adaptation Summary (key sections)", expanded=False):
        if no_summary:
            st.caption("`adaptation_summary.json` is empty or missing.")
        else:
            try:
                meta = {
                    k: summary.get(k)
                    for k in (
                        "generated_at_utc",
                        "schema_version",
                        "advisory_only",
                        "auto_apply_allowed",
                        "phase",
                        "missing_inputs",
                        "proposal_count",
                        "thin_data_proposal_count",
                        "proposal_count_by_target",
                        "proposal_count_by_type",
                        "proposal_count_by_priority",
                        "proposal_count_by_evidence_strength",
                        "notes",
                    )
                    if k in summary
                }
                st.json(meta)
                with st.expander("Adaptation targets registry", expanded=False):
                    targets = summary.get("adaptation_targets")
                    if isinstance(targets, list) and targets:
                        try:
                            _ei_render_table(pd.DataFrame(targets))
                        except Exception:
                            st.json(targets)
                    else:
                        st.caption("No `adaptation_targets` block in summary.")
                with st.expander("Top proposals (from summary)", expanded=False):
                    top = summary.get("top_proposals")
                    if isinstance(top, list) and top:
                        try:
                            _ei_render_table(pd.DataFrame(top))
                        except Exception:
                            st.json(top)
                    else:
                        st.caption("No `top_proposals` block in summary.")
                with st.expander("Full summary JSON", expanded=False):
                    st.json(summary)
            except Exception as e:
                st.caption(f"Summary rendering unavailable: {_short_err(e)}")


# ──────────────────────────────
# APPLIED ADJUSTMENTS PAGE
# (dashboard-only; reads applied_adjustments.csv,
#  applied_adjustments.json, apply_log.csv, apply_summary.json)
# ──────────────────────────────


def _ap_load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Wrap dashboard CSV loader with a friendly per-file warning on hard errors."""
    df, err = read_csv_dashboard(path)
    if err is None:
        return df
    if err == "empty":
        return df
    st.warning(f"{label}: could not read {path.name} ({err}).")
    return None


def _ap_load_json(path: Path, label: str) -> Optional[Dict[str, Any]]:
    """Read a JSON file safely. Returns None on missing/error, dict otherwise."""
    try:
        if not path.exists():
            return None
        try:
            if path.stat().st_size == 0:
                return None
        except OSError:
            pass
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {"value": obj}
    except Exception as e:
        st.warning(f"{label}: could not read {path.name} ({_short_err(e)}).")
        return None


# Reuse the battle-tested EI helpers; thin wrappers keep call-sites consistent.
def _ap_select_existing_columns(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    return _ei_select_existing_columns(df, preferred)


def _ap_value_counts_table(
    df: pd.DataFrame, col: str, order: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    return _ei_value_counts_table(df, col, order=order)


def _ap_safe_metric(
    df: pd.DataFrame, col: str, agg: str = "mean", fmt: str = "{:.3f}", default: str = "N/A"
) -> str:
    return _ei_safe_metric(df, col, agg=agg, fmt=fmt, default=default)


def _ap_truthy_count(df: pd.DataFrame, col: str) -> Optional[int]:
    return _ei_truthy_count(df, col)


def _ap_safe_json_table(obj: Optional[Dict[str, Any]], key: str) -> Optional[pd.DataFrame]:
    """Generic JSON-section flattener (same contract as _ad_safe_json_table)."""
    return _ad_safe_json_table(obj, key)


def _ap_top_rows(
    df: pd.DataFrame, by_cols: List[str], top_n: int = 10, ascending: Optional[List[bool]] = None
) -> Optional[pd.DataFrame]:
    """Sort by the given columns (desc by default) and return the top N rows."""
    if df is None or df.empty or not by_cols:
        return df
    present = [c for c in by_cols if c in df.columns]
    if not present:
        return df.head(top_n).reset_index(drop=True)
    asc = ascending if ascending is not None else [False] * len(present)
    if len(asc) != len(present):
        asc = [False] * len(present)
    out = df.copy()
    try:
        out = out.sort_values(present, ascending=asc, kind="mergesort")
    except Exception:
        pass
    return out.head(top_n).reset_index(drop=True)


def _ap_summary_table_sources(summary: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Normalize apply_summary.json `source_availability` into a tidy table."""
    sa = summary.get("source_availability") if isinstance(summary, dict) else None
    if not sa or not isinstance(sa, dict):
        return None
    rows: List[Dict[str, Any]] = []
    for name, meta in sa.items():
        m = meta if isinstance(meta, dict) else {}
        rows.append(
            {
                "source": name,
                "status": str(m.get("status", "")),
                "rows": int(m.get("rows", 0) or 0),
                "path": str(m.get("path", "")),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["status", "source"], ascending=[True, True])
    return df.reset_index(drop=True)


def _ap_active_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Boolean mask of active rows, using _ei_truthy_count's truthiness set."""
    if df is None or df.empty or "active_flag" not in df.columns:
        return None
    try:
        truthy = {"true", "1", "yes", "t", "y"}
        return df["active_flag"].astype(str).str.strip().str.lower().isin(truthy)
    except Exception:
        return None


# Canonical status order used whenever we render applied-row status counts.
_AP_STATUS_ORDER: List[str] = ["APPLIED", "INACTIVE", "ROLLED_BACK"]

_AP_EVENT_ORDER: List[str] = ["APPLY", "SKIP", "ROLLBACK", "SUPERSEDE", "NOOP"]


def page_applied_adjustments() -> None:
    """✅ Applied Adjustments — observability of the applied-state registry."""
    st.title("✅ Applied Adjustments")
    st.caption(
        "Applied adjustment registry, active state, rollback history, "
        "supersession history, and audit log from Triton's controlled apply layer."
    )

    applied = _ap_load_csv(APPLIED_ADJUSTMENTS_CSV_PATH, "Applied adjustments")
    applied_json = _ap_load_json(APPLIED_ADJUSTMENTS_JSON_PATH, "Applied adjustments JSON")
    apply_log = _ap_load_csv(APPLY_LOG_CSV_PATH, "Apply log")
    summary = _ap_load_json(APPLY_SUMMARY_JSON_PATH, "Apply summary")

    no_applied = applied is None or applied.empty
    no_log = apply_log is None or apply_log.empty
    no_json = applied_json is None or not isinstance(applied_json, dict)
    no_summary = summary is None or not isinstance(summary, dict) or not summary

    if no_applied and no_log and no_json and no_summary:
        st.info(
            "No apply-layer outputs found yet. "
            "Run `python -m services.apply_layer` "
            "(or `python services/apply_layer.py`) to populate "
            "`data/results/applied_adjustments.csv`, "
            "`data/results/applied_adjustments.json`, "
            "`data/results/apply_log.csv`, and "
            "`data/results/apply_summary.json`."
        )

    # ── 1) TOP SUMMARY METRICS ───────────────────────────────────────
    st.markdown("### Summary")

    active_mask = _ap_active_mask(applied) if not no_applied else None
    active_count = int(active_mask.sum()) if active_mask is not None else 0
    inactive_count = int((~active_mask).sum()) if active_mask is not None else 0

    proposals_applied = summary.get("proposals_applied") if isinstance(summary, dict) else None
    proposals_rolled_back = (
        summary.get("proposals_rolled_back") if isinstance(summary, dict) else None
    )
    proposals_skipped = summary.get("proposals_skipped") if isinstance(summary, dict) else None
    supersessions = summary.get("supersessions") if isinstance(summary, dict) else None

    advisory_only = "N/A"
    auto_apply = "N/A"
    approvals_source = "N/A"
    if isinstance(summary, dict):
        if "advisory_only_source" in summary:
            advisory_only = "Yes" if bool(summary.get("advisory_only_source")) else "No"
        if "auto_apply_allowed" in summary:
            auto_apply = "Yes" if bool(summary.get("auto_apply_allowed")) else "No"
        if "approvals_source" in summary:
            approvals_source = str(summary.get("approvals_source") or "N/A")

    sources_avail = "N/A"
    if isinstance(summary, dict):
        sa = summary.get("source_availability") or {}
        if isinstance(sa, dict) and sa:
            ok = sum(1 for v in sa.values() if isinstance(v, dict) and str(v.get("status")) == "ok")
            sources_avail = f"{ok}/{len(sa)}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active adjustments", f"{active_count:,}")
    c2.metric("Inactive adjustments", f"{inactive_count:,}")
    c3.metric(
        "Proposals applied",
        f"{int(proposals_applied):,}" if isinstance(proposals_applied, (int, float)) else "N/A",
    )
    c4.metric(
        "Rolled back",
        (
            f"{int(proposals_rolled_back):,}"
            if isinstance(proposals_rolled_back, (int, float))
            else "N/A"
        ),
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Skipped",
        f"{int(proposals_skipped):,}" if isinstance(proposals_skipped, (int, float)) else "N/A",
    )
    c6.metric(
        "Supersessions",
        f"{int(supersessions):,}" if isinstance(supersessions, (int, float)) else "N/A",
    )
    c7.metric("Advisory-only source", advisory_only)
    c8.metric("Sources available", sources_avail)

    bits: List[str] = []
    if isinstance(summary, dict):
        gen_at = summary.get("generated_at_utc")
        phase = summary.get("phase")
        if gen_at:
            bits.append(f"generated_at_utc=`{gen_at}`")
        if phase:
            bits.append(f"phase=`{phase}`")
        if approvals_source != "N/A":
            bits.append(f"approvals_source=`{approvals_source}`")
    if bits:
        st.caption(" • ".join(bits))

    # ── 2) GOVERNANCE / STATUS ───────────────────────────────────────
    st.markdown("### Governance status")
    if no_summary:
        st.caption("No `apply_summary.json` available — governance status unknown.")
    else:
        gov_msgs: List[str] = []
        if advisory_only == "Yes":
            gov_msgs.append("`advisory_only_source = True` — proposals originated as advisory.")
        if auto_apply == "No":
            gov_msgs.append(
                "`auto_apply_allowed = False` — nothing is being applied automatically."
            )
        phase_s = summary.get("phase")
        if phase_s:
            gov_msgs.append(f"Phase: `{phase_s}`.")
        if gov_msgs:
            st.success(" ".join(gov_msgs))
        else:
            st.info("Governance flags missing from summary — treat this registry as advisory only.")

        st.info(
            "This page is an **audit view** of the applied-state registry. "
            "Applied rows are written to `data/results/applied_adjustments.*` and "
            "are **not yet consumed by live trading code** unless a future component "
            "explicitly opts in."
        )

        # Compact status grid from the summary itself
        st.markdown("**Apply counters (from summary)**")
        counter_rows = [
            ("proposals_seen", summary.get("proposals_seen")),
            ("proposals_approved", summary.get("proposals_approved")),
            ("proposals_applied", summary.get("proposals_applied")),
            ("proposals_skipped", summary.get("proposals_skipped")),
            ("proposals_rolled_back", summary.get("proposals_rolled_back")),
            ("supersessions", summary.get("supersessions")),
            ("active_adjustments_count", summary.get("active_adjustments_count")),
            ("inactive_adjustments_count", summary.get("inactive_adjustments_count")),
        ]
        counter_rows = [(k, v) for k, v in counter_rows if v is not None]
        if counter_rows:
            cdf = pd.DataFrame(counter_rows, columns=["metric", "value"])
            try:
                cdf["value"] = pd.to_numeric(cdf["value"], errors="coerce").fillna(0).astype(int)
            except Exception:
                pass
            _ei_render_table(cdf)

        missing = summary.get("missing_inputs")
        if isinstance(missing, list) and missing:
            st.warning(
                f"Missing or unreadable inputs ({len(missing)}): "
                + ", ".join(str(m) for m in missing)
            )
        notes = summary.get("notes")
        if isinstance(notes, list) and notes:
            with st.expander("Summary notes", expanded=False):
                for n in notes:
                    st.write(f"• {n}")

    # ── 3) SOURCE AVAILABILITY ───────────────────────────────────────
    st.markdown("### Source availability")
    if no_summary:
        st.caption("No `apply_summary.json` available.")
    else:
        src_table = _ap_summary_table_sources(summary)
        if src_table is None or src_table.empty:
            st.caption("Summary JSON has no `source_availability` block.")
        else:
            _ei_render_table(src_table)

    # ── 4) ACTIVE ADJUSTMENTS ────────────────────────────────────────
    st.markdown("### Active adjustments")
    if no_applied:
        st.info("No `applied_adjustments.csv` rows yet.")
    elif active_mask is None:
        st.caption("`active_flag` column missing from `applied_adjustments.csv`.")
    elif not active_mask.any():
        st.caption("No currently-active applied adjustments in the registry.")
    else:
        st.caption(f"{active_count:,} active row(s)")
        active_df = applied[active_mask].copy()
        active_cols = _ap_select_existing_columns(
            active_df,
            [
                "application_id",
                "proposal_id",
                "applied_at_utc",
                "adaptation_target",
                "proposal_type",
                "proposal_direction",
                "proposal_strength",
                "proposal_confidence",
                "effective_delta",
                "min_allowed_value",
                "max_allowed_value",
                "active_flag",
                "status",
                "applied_by",
                "apply_reason",
                "related_bucket",
                "related_flag",
                "related_style",
            ],
        )
        view = active_df[active_cols] if active_cols else active_df
        sorted_view = _ap_top_rows(view, ["proposal_confidence", "applied_at_utc"], top_n=50)
        _ei_render_table(sorted_view, height=380, empty_msg="No active rows to display.")

    # ── 5) REGISTRY STATUS BREAKDOWN ─────────────────────────────────
    st.markdown("### Registry status breakdown")
    if no_applied:
        st.caption("No registry rows to summarize.")
    else:
        col_s, col_af = st.columns(2)
        with col_s:
            st.markdown("**Count by status**")
            _ei_render_table(
                _ap_value_counts_table(applied, "status", order=_AP_STATUS_ORDER),
                empty_msg="No `status` data.",
            )
        with col_af:
            st.markdown("**Count by active_flag**")
            _ei_render_table(
                _ap_value_counts_table(applied, "active_flag"),
                empty_msg="No `active_flag` data.",
            )

        col_r, col_t = st.columns(2)
        with col_r:
            st.markdown("**Count by rollback_eligible**")
            _ei_render_table(
                _ap_value_counts_table(applied, "rollback_eligible"),
                empty_msg="No `rollback_eligible` data.",
            )
        with col_t:
            st.markdown("**Count by adaptation_target**")
            _ei_render_table(
                _ap_value_counts_table(applied, "adaptation_target"),
                empty_msg="No `adaptation_target` data.",
            )

        if "adaptation_target" in applied.columns and "proposal_confidence" in applied.columns:
            try:
                tmp = applied[["adaptation_target", "proposal_confidence"]].copy()
                tmp["proposal_confidence"] = pd.to_numeric(
                    tmp["proposal_confidence"], errors="coerce"
                )
                conf_by_target = (
                    tmp.dropna(subset=["proposal_confidence"])
                    .groupby("adaptation_target")["proposal_confidence"]
                    .agg(["mean", "median", "count"])
                    .round(3)
                    .reset_index()
                    .rename(columns={"mean": "avg_confidence", "median": "median_confidence"})
                    .sort_values("avg_confidence", ascending=False)
                    .reset_index(drop=True)
                )
                st.markdown("**Average confidence by adaptation target**")
                _ei_render_table(conf_by_target)
            except Exception as e:
                st.caption(f"Confidence-by-target unavailable: {_short_err(e)}")

    # ── 6) ROLLBACK / SUPERSESSION HISTORY ───────────────────────────
    st.markdown("### Rollback & supersession history")
    if no_applied:
        st.caption("No registry rows to inspect.")
    else:
        # Rollback history: status == ROLLED_BACK
        rb_mask = None
        if "status" in applied.columns:
            try:
                rb_mask = applied["status"].astype(str).str.upper() == "ROLLED_BACK"
            except Exception:
                rb_mask = None

        st.markdown("**Rollback history**")
        if rb_mask is None or not rb_mask.any():
            st.caption("No rolled-back rows.")
        else:
            rb_df = applied[rb_mask].copy()
            rb_cols = _ap_select_existing_columns(
                rb_df,
                [
                    "application_id",
                    "proposal_id",
                    "adaptation_target",
                    "status",
                    "rollback_parent_application_id",
                    "applied_at_utc",
                    "apply_reason",
                    "apply_note",
                ],
            )
            view = rb_df[rb_cols] if rb_cols else rb_df
            _ei_render_table(
                _ap_top_rows(view, ["applied_at_utc"], top_n=50),
                height=280,
                empty_msg="No rollback rows to display.",
            )

        # Supersession history: superseded_by_application_id populated
        sup_mask = None
        if "superseded_by_application_id" in applied.columns:
            try:
                sup_series = applied["superseded_by_application_id"].astype(str).str.strip()
                sup_mask = sup_series.ne("") & ~sup_series.str.lower().isin(["nan", "none"])
            except Exception:
                sup_mask = None

        st.markdown("**Supersession history**")
        if sup_mask is None or not sup_mask.any():
            st.caption("No superseded rows.")
        else:
            sup_df = applied[sup_mask].copy()
            sup_cols = _ap_select_existing_columns(
                sup_df,
                [
                    "application_id",
                    "proposal_id",
                    "adaptation_target",
                    "status",
                    "superseded_by_application_id",
                    "applied_at_utc",
                    "proposal_confidence",
                    "effective_delta",
                ],
            )
            view = sup_df[sup_cols] if sup_cols else sup_df
            _ei_render_table(
                _ap_top_rows(view, ["applied_at_utc"], top_n=50),
                height=280,
                empty_msg="No supersession rows to display.",
            )

    # ── 7) APPLY LOG ─────────────────────────────────────────────────
    st.markdown("### Apply log")
    if no_log:
        st.info("No `apply_log.csv` events yet.")
    else:
        col_e, col_er = st.columns(2)
        with col_e:
            st.markdown("**Count by event_type**")
            _ei_render_table(
                _ap_value_counts_table(apply_log, "event_type", order=_AP_EVENT_ORDER),
                empty_msg="No `event_type` data.",
            )
        with col_er:
            st.markdown("**Top reasons**")
            reason_tbl = _ap_value_counts_table(apply_log, "reason")
            if reason_tbl is not None and not reason_tbl.empty:
                _ei_render_table(reason_tbl.head(12))
            else:
                st.caption("No `reason` data.")

        st.markdown("**Recent apply log rows**")
        log_cols = _ap_select_existing_columns(
            apply_log,
            [
                "event_time_utc",
                "event_type",
                "application_id",
                "proposal_id",
                "adaptation_target",
                "result",
                "reason",
                "note",
            ],
        )
        view = apply_log[log_cols] if log_cols else apply_log
        # Log is append-only → last rows are most recent.
        try:
            recent = view.tail(50).iloc[::-1].reset_index(drop=True)
        except Exception:
            recent = view
        _ei_render_table(recent, height=360, empty_msg="No apply-log rows to display.")

    # ── 8) APPLIED JSON SNAPSHOT ─────────────────────────────────────
    st.markdown("### Applied JSON snapshot")
    if no_json:
        st.caption("No `applied_adjustments.json` available.")
    else:
        active_list = applied_json.get("active_adjustments") or []
        all_list = applied_json.get("all_adjustments") or []
        notes_list = applied_json.get("notes") or []
        gen_at = applied_json.get("generated_at_utc")

        jm1, jm2 = st.columns(2)
        jm1.metric("active_adjustments", f"{len(active_list):,}")
        jm2.metric("all_adjustments", f"{len(all_list):,}")
        if gen_at:
            st.caption(f"snapshot generated_at_utc=`{gen_at}`")

        if isinstance(active_list, list) and active_list:
            try:
                active_snap = pd.DataFrame(active_list)
                snap_cols = _ap_select_existing_columns(
                    active_snap,
                    [
                        "application_id",
                        "proposal_id",
                        "adaptation_target",
                        "proposal_type",
                        "proposal_direction",
                        "proposal_strength",
                        "proposal_confidence",
                        "effective_delta",
                        "applied_at_utc",
                        "related_bucket",
                        "related_flag",
                        "related_style",
                    ],
                )
                view = active_snap[snap_cols] if snap_cols else active_snap
                _ei_render_table(view, height=300, empty_msg="No active-snapshot rows.")
            except Exception as e:
                st.caption(f"Active snapshot table unavailable: {_short_err(e)}")
        if isinstance(notes_list, list) and notes_list:
            with st.expander("Snapshot notes", expanded=False):
                for n in notes_list:
                    st.write(f"• {n}")

    # ── 9) CURRENT APPLIED STATE OVERVIEW (heuristic audit) ──────────
    st.markdown("### Current applied-state overview")
    if no_applied:
        st.caption("No registry rows to audit.")
    else:
        # Which targets currently have active rows?
        if active_mask is not None and active_mask.any():
            active_df = applied[active_mask].copy()
            if "adaptation_target" in active_df.columns:
                st.markdown("**Adaptation targets with active rows**")
                try:
                    by_target = (
                        active_df.groupby("adaptation_target")
                        .size()
                        .rename("active_rows")
                        .reset_index()
                        .sort_values("active_rows", ascending=False)
                        .reset_index(drop=True)
                    )
                    _ei_render_table(by_target)
                except Exception as e:
                    st.caption(f"Target summary unavailable: {_short_err(e)}")

            # Duplicate active surfaces — same (target, bucket, flag, style)
            key_cols = [
                c
                for c in (
                    "adaptation_target",
                    "related_bucket",
                    "related_flag",
                    "related_style",
                )
                if c in active_df.columns
            ]
            dup_df = None
            if key_cols:
                try:
                    dup_counts = (
                        active_df.assign(
                            **{c: active_df[c].astype(str).fillna("") for c in key_cols}
                        )
                        .groupby(key_cols)
                        .size()
                        .rename("active_rows")
                        .reset_index()
                    )
                    dup_df = dup_counts[dup_counts["active_rows"] > 1]
                except Exception:
                    dup_df = None
            if dup_df is not None and not dup_df.empty:
                st.warning(
                    f"{len(dup_df):,} duplicate active surface(s) detected — "
                    "multiple active rows share the same "
                    "(adaptation_target, related_bucket, related_flag, related_style)."
                )
                _ei_render_table(dup_df)
            else:
                st.caption("No duplicate active surfaces detected.")

            # Anomaly: active rows whose status is not APPLIED
            if "status" in active_df.columns:
                try:
                    bad_status = active_df[active_df["status"].astype(str).str.upper() != "APPLIED"]
                except Exception:
                    bad_status = active_df.iloc[0:0]
                if not bad_status.empty:
                    st.warning(
                        f"{len(bad_status):,} active row(s) carry a status other than APPLIED."
                    )
                    anom_cols = _ap_select_existing_columns(
                        bad_status,
                        [
                            "application_id",
                            "proposal_id",
                            "adaptation_target",
                            "status",
                            "active_flag",
                            "applied_at_utc",
                        ],
                    )
                    view = bad_status[anom_cols] if anom_cols else bad_status
                    _ei_render_table(view, height=220)
        else:
            st.caption("No active rows; nothing to audit in the overview.")

    # ── 10) RAW DETAILED TABLES ──────────────────────────────────────
    st.markdown("### Raw data")
    with st.expander("Applied Adjustments Registry (raw)", expanded=False):
        if no_applied:
            st.caption("`applied_adjustments.csv` is empty or missing.")
        else:
            st.caption(f"{applied.shape[0]:,} rows × {applied.shape[1]:,} columns")
            _ei_render_table(applied, height=420)

    with st.expander("Apply Log (raw, most recent first)", expanded=False):
        if no_log:
            st.caption("`apply_log.csv` is empty or missing.")
        else:
            try:
                log_view = apply_log.iloc[::-1].reset_index(drop=True)
            except Exception:
                log_view = apply_log
            st.caption(f"{apply_log.shape[0]:,} rows × {apply_log.shape[1]:,} columns")
            _ei_render_table(log_view, height=360)

    with st.expander("Applied JSON Snapshot (raw)", expanded=False):
        if no_json:
            st.caption("`applied_adjustments.json` is empty or missing.")
        else:
            try:
                meta = {
                    k: applied_json.get(k)
                    for k in (
                        "generated_at_utc",
                        "schema_version",
                        "phase",
                        "notes",
                    )
                    if k in applied_json
                }
                st.json(meta)
                with st.expander("Full snapshot JSON", expanded=False):
                    st.json(applied_json)
            except Exception as e:
                st.caption(f"Snapshot rendering unavailable: {_short_err(e)}")

    with st.expander("Apply Summary (key sections)", expanded=False):
        if no_summary:
            st.caption("`apply_summary.json` is empty or missing.")
        else:
            try:
                meta = {
                    k: summary.get(k)
                    for k in (
                        "generated_at_utc",
                        "schema_version",
                        "phase",
                        "advisory_only_source",
                        "auto_apply_allowed",
                        "approvals_source",
                        "missing_inputs",
                        "proposals_seen",
                        "proposals_approved",
                        "proposals_applied",
                        "proposals_skipped",
                        "proposals_rolled_back",
                        "supersessions",
                        "active_adjustments_count",
                        "inactive_adjustments_count",
                        "notes",
                    )
                    if k in summary
                }
                st.json(meta)
                with st.expander("Top active adjustments (from summary)", expanded=False):
                    top = summary.get("top_active_adjustments")
                    if isinstance(top, list) and top:
                        try:
                            _ei_render_table(pd.DataFrame(top))
                        except Exception:
                            st.json(top)
                    else:
                        st.caption("No `top_active_adjustments` block in summary.")
                with st.expander("Full summary JSON", expanded=False):
                    st.json(summary)
            except Exception as e:
                st.caption(f"Summary rendering unavailable: {_short_err(e)}")


# ─────────────────────────────────────────────────────────────
# 🧪 ADAPTATION SIMULATION — Phase-2 observability page.
# Reads data/results/adaptation_simulation.csv + _summary.json.
# The simulation itself is produced by services/adaptation_simulation.py
# which is read-only w.r.t. live trading logic.
# ─────────────────────────────────────────────────────────────


_AS_DECISION_ORDER: List[str] = [
    "UNCHANGED_ACCEPT",
    "NEWLY_REJECTED",
    "NEWLY_ACCEPTED",
    "UNCHANGED_REJECT",
]


def _as_load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Mirror of the _ad_/_ap_ loaders — tolerant of missing/empty/malformed."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"{label}: could not read `{path.name}` — {_short_err(e)}")
        return pd.DataFrame()
    return df


def _as_load_json(path: Path, label: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception as e:
        st.warning(f"{label}: could not read `{path.name}` — {_short_err(e)}")
        return {}
    txt = (txt or "").strip()
    if not txt:
        return {}
    try:
        obj = json.loads(txt)
    except Exception as e:
        st.warning(f"{label}: `{path.name}` is not valid JSON — {_short_err(e)}")
        return {}
    return obj if isinstance(obj, dict) else {}


def _as_select_existing_columns(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    if df is None or df.empty:
        return []
    return [c for c in preferred if c in df.columns]


def _as_value_counts_table(
    df: pd.DataFrame, col: str, order: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    return _ad_value_counts_table(df, col, order=order)


def _as_truthy_count(df: pd.DataFrame, col: str) -> Optional[int]:
    return _ad_truthy_count(df, col)


def _as_summary_table_sources(summary: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Render source_availability block from simulation summary JSON."""
    return _ad_summary_table_sources(summary)


def _as_acceptance_bar_chart(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Build the tiny baseline-vs-simulated table for st.bar_chart."""
    if df is None or df.empty:
        return None
    try:
        b_acc = int(
            df.get("baseline_accepted", pd.Series([], dtype=object))
            .apply(lambda v: str(v).strip().lower() in ("true", "1"))
            .sum()
        )
        b_rej = len(df) - b_acc
        s_acc = int(
            df.get("simulated_accepted", pd.Series([], dtype=object))
            .apply(lambda v: str(v).strip().lower() in ("true", "1"))
            .sum()
        )
        s_rej = len(df) - s_acc
        return pd.DataFrame(
            {
                "accepted": [b_acc, s_acc],
                "rejected": [b_rej, s_rej],
            },
            index=["baseline", "simulated"],
        )
    except Exception:
        return None


def page_adaptation_simulation() -> None:
    """🧪 Adaptation Simulation — what-if preview for active applied adjustments."""
    st.title("🧪 Adaptation Simulation")
    st.caption(
        "Phase-2 what-if preview. Takes the ACTIVE rows from "
        "`applied_adjustments.csv` and simulates — in memory only — what "
        "would change in acceptance, thresholds, and opportunity flow. "
        "**No broker, execution, lifecycle, or risk state is modified.**"
    )

    sim = _as_load_csv(ADAPTATION_SIMULATION_CSV_PATH, "Adaptation simulation CSV")
    summary = _as_load_json(
        ADAPTATION_SIMULATION_SUMMARY_JSON_PATH, "Adaptation simulation summary"
    )

    no_sim = sim is None or sim.empty
    no_summary = summary is None or not isinstance(summary, dict) or not summary

    if no_sim and no_summary:
        st.info(
            "No simulation outputs found yet. Run "
            "`python -m services.adaptation_simulation` to populate "
            "`data/results/adaptation_simulation.csv` and "
            "`data/results/adaptation_simulation_summary.json`."
        )

    # ── 1) SUMMARY COMPARISON ─────────────────────────────────────────
    st.markdown("### Summary — baseline vs simulated")

    opps_seen = summary.get("opportunities_seen") if isinstance(summary, dict) else None
    active_n = summary.get("active_adjustments_count") if isinstance(summary, dict) else None
    active_used = (
        summary.get("active_adjustments_used_count") if isinstance(summary, dict) else None
    )

    baseline_counts = (summary or {}).get("baseline_counts") or {}
    simulated_counts = (summary or {}).get("simulated_counts") or {}
    decision_counts = (summary or {}).get("decision_change_counts") or {}

    def _int_or_na(v: Any) -> str:
        try:
            if v is None:
                return "N/A"
            return f"{int(v):,}"
        except Exception:
            return "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opportunities seen", _int_or_na(opps_seen))
    c2.metric("Active adjustments", _int_or_na(active_n))
    c3.metric("Active used", _int_or_na(active_used))
    c4.metric("Newly rejected", _int_or_na(decision_counts.get("NEWLY_REJECTED")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Baseline accepted", _int_or_na(baseline_counts.get("accepted")))
    c6.metric("Simulated accepted", _int_or_na(simulated_counts.get("accepted")))
    c7.metric("Baseline rejected", _int_or_na(baseline_counts.get("rejected")))
    c8.metric("Newly accepted", _int_or_na(decision_counts.get("NEWLY_ACCEPTED")))

    bits: List[str] = []
    if isinstance(summary, dict):
        gen_at = summary.get("generated_at_utc")
        phase = summary.get("phase")
        score_floor = summary.get("score_floor")
        sim_only = summary.get("simulation_only")
        adv_only = summary.get("advisory_only")
        if gen_at:
            bits.append(f"generated_at_utc=`{gen_at}`")
        if phase:
            bits.append(f"phase=`{phase}`")
        if score_floor is not None:
            bits.append(f"score_floor=`{score_floor}`")
        if sim_only is True:
            bits.append("simulation_only=`True`")
        if adv_only is True:
            bits.append("advisory_only=`True`")
    if bits:
        st.caption(" • ".join(bits))

    if not no_sim:
        try:
            chart_df = _as_acceptance_bar_chart(sim)
            if chart_df is not None:
                col_a, col_b = st.columns([2, 3])
                with col_a:
                    st.markdown("**Acceptance comparison**")
                    _ei_render_table(chart_df.reset_index().rename(columns={"index": "series"}))
                with col_b:
                    try:
                        st.bar_chart(chart_df)
                    except Exception as e:
                        st.caption(f"Bar chart unavailable: {_short_err(e)}")
        except Exception as e:
            st.caption(f"Acceptance summary unavailable: {_short_err(e)}")

        dc_counts = _as_value_counts_table(sim, "decision_change", order=_AS_DECISION_ORDER)
        st.markdown("**Decision-change counts**")
        _ei_render_table(dc_counts, empty_msg="No `decision_change` column.")

    # ── 2) THRESHOLD IMPACT ──────────────────────────────────────────
    st.markdown("### Threshold impact")
    if no_summary:
        st.caption("No summary JSON — threshold utilisation unavailable.")
    else:
        tu = summary.get("threshold_utilization") or {}
        exposure = summary.get("exposure_delta_estimate") or {}

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric(
            "ADD-threshold matches",
            _int_or_na(tu.get("add_score_threshold_matches")),
        )
        tc2.metric(
            "Trim-threshold rows",
            _int_or_na(tu.get("trim_profit_threshold_rows")),
        )
        tc3.metric(
            "Cooldown-bias rows",
            _int_or_na(tu.get("position_cooldown_bias_rows")),
        )

        if isinstance(exposure, dict) and exposure:
            st.markdown("**Exposure delta (heuristic proxy — not a real risk figure)**")
            try:
                ex_df = pd.DataFrame(
                    [
                        (
                            "newly_rejected_exposure_proxy",
                            exposure.get("newly_rejected_exposure_proxy"),
                        ),
                        (
                            "newly_accepted_exposure_proxy",
                            exposure.get("newly_accepted_exposure_proxy"),
                        ),
                        ("net_exposure_change_proxy", exposure.get("net_exposure_change_proxy")),
                    ],
                    columns=["metric", "value"],
                )
                _ei_render_table(ex_df)
                note = exposure.get("note")
                if note:
                    st.caption(str(note))
            except Exception as e:
                st.caption(f"Exposure summary unavailable: {_short_err(e)}")

    # ── 3) SIGNAL ACCEPTANCE CHANGES ─────────────────────────────────
    st.markdown("### Signal acceptance changes")
    if no_sim:
        st.caption("No per-row simulation rows available.")
    else:
        curated_cols = _as_select_existing_columns(
            sim,
            [
                "row_id",
                "symbol",
                "opportunity_type",
                "effective_stance",
                "sizing_bucket",
                "confidence",
                "edge_score",
                "baseline_score",
                "baseline_threshold",
                "simulated_score",
                "simulated_threshold",
                "simulated_penalty_total",
                "simulated_boost_total",
                "score_delta",
                "decision_change",
                "baseline_reject_reason",
                "simulated_reject_reason",
                "adjustments_applied",
                "adjustment_details",
                "spread_bucket",
                "quote_is_stale",
                "execution_risk_flag",
            ],
        )

        def _filter_decision(df: pd.DataFrame, decision: str) -> pd.DataFrame:
            if "decision_change" not in df.columns:
                return df.iloc[0:0]
            return df[df["decision_change"].astype(str).str.upper() == decision].copy()

        newly_rejected = _filter_decision(sim, "NEWLY_REJECTED")
        newly_accepted = _filter_decision(sim, "NEWLY_ACCEPTED")

        col_nr, col_na = st.columns(2)
        with col_nr:
            st.markdown(f"**Newly rejected ({len(newly_rejected):,})**")
            if newly_rejected.empty:
                st.caption("No opportunities flip accept→reject.")
            else:
                view = newly_rejected[curated_cols] if curated_cols else newly_rejected
                _ei_render_table(view.head(50), height=340)
        with col_na:
            st.markdown(f"**Newly accepted ({len(newly_accepted):,})**")
            if newly_accepted.empty:
                st.caption("No opportunities flip reject→accept.")
            else:
                view = newly_accepted[curated_cols] if curated_cols else newly_accepted
                _ei_render_table(view.head(50), height=340)

        if "baseline_reject_reason" in sim.columns:
            try:
                rr = (
                    sim[sim["baseline_reject_reason"].astype(str).str.strip() != ""][
                        "baseline_reject_reason"
                    ]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                rr.columns = ["baseline_reject_reason", "count"]
                if not rr.empty:
                    st.markdown("**Baseline reject reasons**")
                    _ei_render_table(rr.head(15))
            except Exception as e:
                st.caption(f"Reject-reason summary unavailable: {_short_err(e)}")

        if "simulated_reject_reason" in sim.columns:
            try:
                rr = (
                    sim[sim["simulated_reject_reason"].astype(str).str.strip() != ""][
                        "simulated_reject_reason"
                    ]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                rr.columns = ["simulated_reject_reason", "count"]
                if not rr.empty:
                    st.markdown("**Simulated reject reasons**")
                    _ei_render_table(rr.head(15))
            except Exception as e:
                st.caption(f"Simulated-reject summary unavailable: {_short_err(e)}")

    # ── 4) OPPORTUNITY CHANGES ───────────────────────────────────────
    st.markdown("### Opportunity changes by type")
    if no_sim:
        st.caption("No simulation rows to break down.")
    else:
        try:
            if {"opportunity_type", "decision_change"}.issubset(set(sim.columns)):
                crosstab = (
                    sim.groupby(["opportunity_type", "decision_change"])
                    .size()
                    .unstack(fill_value=0)
                )
                # Keep our canonical column order when present
                existing_order = [c for c in _AS_DECISION_ORDER if c in crosstab.columns]
                leftover = [c for c in crosstab.columns if c not in _AS_DECISION_ORDER]
                crosstab = crosstab[existing_order + leftover]
                crosstab = crosstab.reset_index()
                _ei_render_table(crosstab)
            else:
                st.caption("`opportunity_type` or `decision_change` column missing.")
        except Exception as e:
            st.caption(f"Opportunity breakdown unavailable: {_short_err(e)}")

    # ── 5) RISK IMPACT (COUNTS ONLY) ─────────────────────────────────
    st.markdown("### Risk impact (counts only)")
    st.caption(
        "Counts of rows that matched risk-sensitive conditions during "
        "simulation. No real risk state is modified."
    )
    if no_summary:
        st.caption("No summary JSON — risk-impact counts unavailable.")
    else:
        rc = summary.get("risk_impact_counts") or {}
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric(
            "Wide / too-wide spread rows", _int_or_na(rc.get("wide_or_too_wide_spread_rows"))
        )
        rc2.metric("Stale-quote rows", _int_or_na(rc.get("stale_quote_rows")))
        rc3.metric("HIGH exec-risk rows", _int_or_na(rc.get("high_execution_risk_rows")))
        note = rc.get("note") if isinstance(rc, dict) else None
        if note:
            st.caption(str(note))

    # ── 6) ACTIVE ADJUSTMENTS USED ───────────────────────────────────
    st.markdown("### Active adjustments used")
    adj_rows: List[Dict[str, Any]] = []
    if isinstance(summary, dict):
        raw = summary.get("active_adjustments") or []
        if isinstance(raw, list):
            adj_rows = [r for r in raw if isinstance(r, dict)]
    if not adj_rows:
        st.caption("No active adjustments were present in the last simulation.")
    else:
        try:
            adj_df = pd.DataFrame(adj_rows)
            preferred = _as_select_existing_columns(
                adj_df,
                [
                    "adaptation_target",
                    "effect",
                    "effective_delta",
                    "related_bucket",
                    "related_flag",
                    "related_style",
                    "rows_matched",
                    "application_id",
                    "proposal_id",
                    "description",
                ],
            )
            view = adj_df[preferred] if preferred else adj_df
            if "rows_matched" in view.columns:
                try:
                    view = view.sort_values("rows_matched", ascending=False)
                except Exception:
                    pass
            _ei_render_table(view, height=280)
        except Exception as e:
            st.caption(f"Active-adjustment table unavailable: {_short_err(e)}")

    # ── 7) SOURCE AVAILABILITY ───────────────────────────────────────
    st.markdown("### Source availability")
    if no_summary:
        st.caption("No summary JSON available.")
    else:
        src_table = _as_summary_table_sources(summary)
        if src_table is None or src_table.empty:
            st.caption("Summary JSON has no `source_availability` block.")
        else:
            _ei_render_table(src_table)
        missing = summary.get("missing_inputs") if isinstance(summary, dict) else None
        if isinstance(missing, list) and missing:
            st.warning(
                f"Missing or unreadable inputs ({len(missing)}): "
                + ", ".join(str(m) for m in missing)
            )

    # ── 8) EXPANDABLE RAW TABLES ─────────────────────────────────────
    st.markdown("### Raw data")
    with st.expander("Simulation per-row results (raw)", expanded=False):
        if no_sim:
            st.caption("`adaptation_simulation.csv` is empty or missing.")
        else:
            st.caption(f"{sim.shape[0]:,} rows × {sim.shape[1]:,} columns")
            _ei_render_table(sim, height=420)

    with st.expander("Simulation summary JSON (key sections)", expanded=False):
        if no_summary:
            st.caption("`adaptation_simulation_summary.json` is empty or missing.")
        else:
            try:
                meta_keys = [
                    "generated_at_utc",
                    "schema_version",
                    "phase",
                    "simulation_only",
                    "advisory_only",
                    "auto_apply_allowed",
                    "score_floor",
                    "missing_inputs",
                    "opportunities_seen",
                    "active_adjustments_count",
                    "active_adjustments_used_count",
                    "baseline_counts",
                    "simulated_counts",
                    "decision_change_counts",
                    "exposure_delta_estimate",
                    "threshold_utilization",
                    "risk_impact_counts",
                    "notes",
                ]
                meta = {k: summary.get(k) for k in meta_keys if k in summary}
                st.json(meta)
                with st.expander("Full summary JSON", expanded=False):
                    st.json(summary)
            except Exception as e:
                st.caption(f"Summary rendering unavailable: {_short_err(e)}")


# ──────────────────────────────
# PERFORMANCE INTELLIGENCE (read-only analytics)
# ──────────────────────────────
def _pi_format_money(v: Any) -> str:
    try:
        f = float(v)
    except Exception:
        return "N/A"
    if not np.isfinite(f):
        return "N/A"
    sign = "-" if f < 0 else ""
    return f"{sign}${abs(f):,.2f}"


def _pi_color_pl(val: Any) -> str:
    try:
        f = float(val)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    if f > 0:
        return "color: #16a34a; font-weight: 600;"
    if f < 0:
        return "color: #dc2626; font-weight: 600;"
    return ""


_PI_RISK_FLAG_RANK: Dict[str, int] = {
    "FORCE_EXIT": 0,
    "TRIM_PRIORITY": 1,
    "BLOCK_NEW_BUY": 2,
    "OK": 3,
}

# Visual styling per risk flag (text colour + light background). FORCE_EXIT is
# the loudest red, TRIM_PRIORITY amber, BLOCK_NEW_BUY blue, OK muted green.
_PI_RISK_FLAG_STYLE: Dict[str, str] = {
    "FORCE_EXIT": "background-color: #fee2e2; color: #b91c1c; font-weight: 700;",
    "TRIM_PRIORITY": "background-color: #fef3c7; color: #b45309; font-weight: 700;",
    "BLOCK_NEW_BUY": "background-color: #dbeafe; color: #1d4ed8; font-weight: 700;",
    "OK": "background-color: #dcfce7; color: #166534; font-weight: 600;",
}

# HTML badge colours for the legend / metric annotations (matching cell styles).
_PI_RISK_FLAG_BADGE_HTML: Dict[str, str] = {
    "FORCE_EXIT": (
        '<span style="background-color:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;'
        'padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.85rem;">'
        "🛑 FORCE_EXIT</span>"
    ),
    "TRIM_PRIORITY": (
        '<span style="background-color:#fef3c7;color:#b45309;border:1px solid #fcd34d;'
        'padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.85rem;">'
        "✂️ TRIM_PRIORITY</span>"
    ),
    "BLOCK_NEW_BUY": (
        '<span style="background-color:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;'
        'padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.85rem;">'
        "🚫 BLOCK_NEW_BUY</span>"
    ),
    "OK": (
        '<span style="background-color:#dcfce7;color:#166534;border:1px solid #86efac;'
        'padding:2px 8px;border-radius:6px;font-weight:600;font-size:0.85rem;">'
        "✅ OK</span>"
    ),
}


def _pi_risk_flag_priority(flag: Any) -> int:
    """Return numeric severity (lower = more urgent) for sort ordering."""
    s = str(flag or "").strip().upper()
    if not s:
        return _PI_RISK_FLAG_RANK["OK"]
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return _PI_RISK_FLAG_RANK["OK"]
    return min(_PI_RISK_FLAG_RANK.get(p, 99) for p in parts)


def _pi_risk_flag_cell_style(flag: Any) -> str:
    """Cell CSS for the risk_flag column. Returns '' for unknown values."""
    s = str(flag or "").strip().upper()
    if not s:
        return ""
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return ""
    # Use the highest-severity component for cell colour.
    primary = min(parts, key=lambda p: _PI_RISK_FLAG_RANK.get(p, 99))
    return _PI_RISK_FLAG_STYLE.get(primary, "")


def _pi_drag_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a boolean mask of drag rows when an inferable column exists."""
    if df is None or df.empty:
        return None
    if "drag_flag" in df.columns:
        try:
            return df["drag_flag"].astype(bool)
        except Exception:
            try:
                s = df["drag_flag"].astype(str).str.strip().str.lower()
                return s.isin({"true", "1", "yes", "y"})
            except Exception:
                return None
    if "performance_bucket" in df.columns:
        try:
            return df["performance_bucket"].astype(str).str.upper() == "HIGH_DRAG"
        except Exception:
            return None
    if "severity_bucket" in df.columns and "total_pl" in df.columns:
        try:
            sev = df["severity_bucket"].astype(str).str.upper() == "HIGH"
            pl = pd.to_numeric(df["total_pl"], errors="coerce")
            return sev & (pl < 0)
        except Exception:
            return None
    return None


def page_performance_intelligence() -> None:
    st.markdown("### 📊 Performance Intelligence")
    st.caption(
        "Read-only analytics. Aggregates `performance_intelligence_*` artifacts produced by "
        "`services/build_performance_intelligence.py`. Does not influence trading."
    )

    summary = load_json(PERFORMANCE_INTELLIGENCE_SUMMARY_JSON_PATH, show_error=False) or {}
    system_df = csv_usable_rows(PERFORMANCE_INTELLIGENCE_CSV_PATH)
    by_symbol_df = csv_usable_rows(PERFORMANCE_INTELLIGENCE_BY_SYMBOL_CSV_PATH)

    if (
        not summary
        and (system_df is None or system_df.empty)
        and (by_symbol_df is None or by_symbol_df.empty)
    ):
        st.info(
            "No performance intelligence artifacts found yet. "
            "Run `python -m services.build_performance_intelligence` to populate this page.",
            icon="ℹ️",
        )
        with st.expander("Expected files", expanded=False):
            st.code(
                "\n".join(
                    [
                        str(PERFORMANCE_INTELLIGENCE_CSV_PATH),
                        str(PERFORMANCE_INTELLIGENCE_BY_SYMBOL_CSV_PATH),
                        str(PERFORMANCE_INTELLIGENCE_SUMMARY_JSON_PATH),
                    ]
                )
            )
        return

    # ── 1) SYSTEM PERFORMANCE ────────────────────────────────────────
    st.markdown("#### System Performance")

    # Prefer summary JSON; fall back to one-row CSV when JSON is missing.
    sys_view: Dict[str, Any] = {}
    if summary:
        sys_view = dict(summary)
    elif system_df is not None and not system_df.empty:
        try:
            sys_view = {k: system_df.iloc[-1].get(k) for k in system_df.columns}
        except Exception:
            sys_view = {}

    if not sys_view:
        st.warning("System summary unavailable.", icon="⚠️")
    else:
        gen_at = sys_view.get("generated_at_utc")
        if gen_at:
            st.caption(f"Generated at: `{gen_at}` (UTC)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total symbols", f"{int(sys_view.get('total_symbols') or 0):,}")
        m2.metric("Winners", f"{int(sys_view.get('winners') or 0):,}")
        m3.metric("Losers", f"{int(sys_view.get('losers') or 0):,}")
        m4.metric("High-drag symbols", f"{int(sys_view.get('high_drag_symbols') or 0):,}")

        n1, n2, n3, n4 = st.columns(4)
        best_sym = sys_view.get("best_symbol") or "—"
        worst_sym = sys_view.get("worst_symbol") or "—"
        n1.metric("Best symbol", str(best_sym))
        n2.metric("Worst symbol", str(worst_sym))
        n3.metric("Total combined P/L", _pi_format_money(sys_view.get("total_combined_pl")))
        n4.metric("Open positions", f"{int(sys_view.get('open_positions') or 0):,}")

        with st.expander("Realized vs unrealized split", expanded=False):
            r1, r2 = st.columns(2)
            r1.metric("Total realized P/L", _pi_format_money(sys_view.get("total_realized_pl")))
            r2.metric("Total unrealized P/L", _pi_format_money(sys_view.get("total_unrealized_pl")))

        if isinstance(summary, dict):
            with st.expander("Full summary JSON", expanded=False):
                st.json(summary)

    # ── 2) SYMBOL PERFORMANCE ────────────────────────────────────────
    st.markdown("#### Symbol Performance")

    if by_symbol_df is None or by_symbol_df.empty:
        st.info("`performance_intelligence_by_symbol.csv` is missing or empty.", icon="ℹ️")
    else:
        df = by_symbol_df.copy()

        sym_col: Optional[str] = None
        for cand in ("ticker", "symbol"):
            if cand in df.columns:
                sym_col = cand
                break

        # Coerce numeric columns where applicable.
        numeric_candidates = [
            "total_pl",
            "realized_pl",
            "unrealized_pl",
            "win_rate",
            "trade_count",
            "trade_rows",
            "open_qty",
            "recent_order_count",
            "buy_count",
            "sell_count",
            "filled_count",
            "open_order_count",
            "add_count",
            "exit_count",
            "trim_count",
        ]
        for c in numeric_candidates:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Build display table — only include columns that exist.
        preferred_cols = [
            sym_col,
            "total_pl",
            "realized_pl",
            "unrealized_pl",
            "win_rate",
            "trade_count",
            "trade_rows",
            "performance_bucket",
            "severity_bucket",
            "open_qty",
            "drag_flag",
        ]
        cols_to_show = [c for c in preferred_cols if c and c in df.columns]
        view = df[cols_to_show].copy() if cols_to_show else df.copy()

        if "total_pl" in view.columns:
            view = view.sort_values("total_pl", ascending=False, na_position="last").reset_index(
                drop=True
            )
        elif sym_col is not None:
            view = view.sort_values(sym_col, ascending=True, na_position="last").reset_index(
                drop=True
            )

        # Highlight P/L columns green/red.
        pl_cols = [c for c in ("total_pl", "realized_pl", "unrealized_pl") if c in view.columns]
        try:
            sty = view.style
            if pl_cols:
                sty = sty.applymap(_pi_color_pl, subset=pl_cols)
                sty = sty.format({c: "{:,.2f}" for c in pl_cols})
            st.dataframe(sty, use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"Styled table unavailable: {_short_err(e)}")
            st.dataframe(view, use_container_width=True, hide_index=True)

        # ── 3) VISUALS ────────────────────────────────────────────────
        st.markdown("#### Top winners & losers")
        if sym_col is None or "total_pl" not in df.columns:
            st.caption("Need `total_pl` and a symbol column to render top/worst charts.")
        else:
            try:
                ranked = df[[sym_col, "total_pl"]].dropna(subset=["total_pl"]).copy()
                if ranked.empty:
                    st.caption("No P/L rows to chart.")
                else:
                    top10 = ranked.sort_values("total_pl", ascending=False).head(10)
                    bot10 = ranked.sort_values("total_pl", ascending=True).head(10)
                    cL, cR = st.columns(2)
                    with cL:
                        st.caption("Top 10 winners (by total P/L)")
                        st.bar_chart(top10.set_index(sym_col)[["total_pl"]])
                    with cR:
                        st.caption("Top 10 worst (by total P/L)")
                        st.bar_chart(bot10.set_index(sym_col)[["total_pl"]])
            except Exception as e:
                st.caption(f"Top/worst chart unavailable: {_short_err(e)}")

        st.markdown("#### Distribution of total P/L across symbols")
        if "total_pl" not in df.columns:
            st.caption("`total_pl` column not present — cannot render distribution.")
        else:
            try:
                pl_series = pd.to_numeric(df["total_pl"], errors="coerce").dropna()
                if pl_series.empty:
                    st.caption("No numeric `total_pl` values to chart.")
                else:
                    plotted = False
                    try:
                        import plotly.express as px  # type: ignore

                        fig = px.histogram(
                            pl_series.rename("total_pl").to_frame(),
                            x="total_pl",
                            nbins=30,
                            title="Symbol P/L distribution",
                        )
                        fig.update_layout(bargap=0.05)
                        st.plotly_chart(fig, use_container_width=True)
                        plotted = True
                    except Exception:
                        plotted = False

                    if not plotted:
                        # Streamlit fallback: bucketed bar chart.
                        try:
                            lo = float(pl_series.min())
                            hi = float(pl_series.max())
                            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                                st.caption(
                                    "Distribution not chartable (all values equal or non-finite)."
                                )
                            else:
                                bins = np.linspace(lo, hi, 21)
                                cuts = pd.cut(pl_series, bins=bins, include_lowest=True)
                                counts = cuts.value_counts().sort_index()
                                labels = [
                                    f"{interval.left:,.0f} → {interval.right:,.0f}"
                                    for interval in counts.index
                                ]
                                hist_df = pd.DataFrame({"bucket": labels, "count": counts.values})
                                st.bar_chart(hist_df.set_index("bucket"))
                        except Exception as e:
                            st.caption(f"Histogram unavailable: {_short_err(e)}")
            except Exception as e:
                st.caption(f"Distribution chart unavailable: {_short_err(e)}")

        # ── 4) DRAG ANALYSIS ─────────────────────────────────────────
        st.markdown("#### Drag Analysis")
        drag_mask = _pi_drag_mask(df)
        if drag_mask is None:
            st.caption(
                "No `drag_flag` / `performance_bucket` / `severity_bucket` column found — "
                "drag analysis skipped."
            )
        else:
            drag_df = df[drag_mask.fillna(False)]
            if drag_df.empty:
                st.success("No high-drag symbols flagged. ✅")
            else:
                st.warning(
                    f"⚠️ High Drag Symbols ({len(drag_df):,}) — review for capital protection.",
                    icon="⚠️",
                )
                drag_cols = [
                    c
                    for c in (
                        sym_col,
                        "total_pl",
                        "realized_pl",
                        "unrealized_pl",
                        "performance_bucket",
                        "severity_bucket",
                        "loss_source",
                        "current_lifecycle_stance",
                        "effective_position_state",
                        "open_qty",
                        "drag_flag",
                    )
                    if c and c in drag_df.columns
                ]
                drag_view = drag_df[drag_cols] if drag_cols else drag_df
                if "total_pl" in drag_view.columns:
                    drag_view = drag_view.sort_values(
                        "total_pl", ascending=True, na_position="last"
                    )
                drag_view = drag_view.reset_index(drop=True)
                try:
                    sty = drag_view.style
                    pl_cols_d = [
                        c
                        for c in ("total_pl", "realized_pl", "unrealized_pl")
                        if c in drag_view.columns
                    ]
                    if pl_cols_d:
                        sty = sty.applymap(_pi_color_pl, subset=pl_cols_d)
                        sty = sty.format({c: "{:,.2f}" for c in pl_cols_d})
                    st.dataframe(sty, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.caption(f"Drag table styling unavailable: {_short_err(e)}")
                    st.dataframe(drag_view, use_container_width=True, hide_index=True)

    # ── 5) RISK OVERLAY ──────────────────────────────────────────────
    st.markdown("#### Risk Overlay")
    st.caption("ℹ️ This is read-only intelligence. It does not place, cancel, trim, or exit trades.")

    overlay_df = csv_usable_rows(PERFORMANCE_RISK_OVERLAY_CSV_PATH)
    if overlay_df is None or overlay_df.empty:
        st.info(
            "`performance_risk_overlay.csv` not found or empty. "
            "Run `python -m services.performance_risk_overlay` to generate it.",
            icon="ℹ️",
        )
    else:
        try:
            ov = overlay_df.copy()

            sym_col_o: Optional[str] = None
            for cand in ("ticker", "symbol"):
                if cand in ov.columns:
                    sym_col_o = cand
                    break

            for c in ("total_pl", "unrealized_pl"):
                if c in ov.columns:
                    ov[c] = pd.to_numeric(ov[c], errors="coerce")

            if "drag_flag" in ov.columns:
                try:
                    ov["drag_flag"] = ov["drag_flag"].apply(
                        lambda v: (
                            bool(v)
                            if isinstance(v, (bool, int, float))
                            else str(v).strip().lower() in {"true", "1", "yes", "y"}
                        )
                    )
                except Exception:
                    pass

            flag_series = (
                ov["risk_flag"].fillna("OK").astype(str).str.upper()
                if "risk_flag" in ov.columns
                else pd.Series(["OK"] * len(ov), index=ov.index)
            )

            def _has_flag(label: str) -> int:
                return int(flag_series.str.contains(label, regex=False).sum())

            n_total = int(len(ov))
            n_force = _has_flag("FORCE_EXIT")
            n_trim = _has_flag("TRIM_PRIORITY")
            n_block = _has_flag("BLOCK_NEW_BUY")
            n_ok = int((flag_series == "OK").sum())

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Total symbols", f"{n_total:,}")
            r2.metric("🛑 FORCE_EXIT", f"{n_force:,}")
            r3.metric("✂️ TRIM_PRIORITY", f"{n_trim:,}")
            r4.metric("🚫 BLOCK_NEW_BUY", f"{n_block:,}")
            r5.metric("✅ OK", f"{n_ok:,}")

            legend_html = "&nbsp;&nbsp;".join(
                _PI_RISK_FLAG_BADGE_HTML[k]
                for k in ("FORCE_EXIT", "TRIM_PRIORITY", "BLOCK_NEW_BUY", "OK")
            )
            st.markdown(
                f"<div style='margin:0.35rem 0 0.5rem 0;'>{legend_html}</div>",
                unsafe_allow_html=True,
            )

            preferred_cols_o = [
                sym_col_o,
                "total_pl",
                "unrealized_pl",
                "drag_flag",
                "risk_flag",
            ]
            cols_show_o = [c for c in preferred_cols_o if c and c in ov.columns]
            view_o = ov[cols_show_o].copy() if cols_show_o else ov.copy()

            if "risk_flag" in view_o.columns:
                view_o["__rank__"] = view_o["risk_flag"].apply(_pi_risk_flag_priority)
            else:
                view_o["__rank__"] = _PI_RISK_FLAG_RANK["OK"]

            sort_keys = ["__rank__"]
            sort_asc = [True]
            if "total_pl" in view_o.columns:
                sort_keys.append("total_pl")
                sort_asc.append(True)
            view_o = (
                view_o.sort_values(sort_keys, ascending=sort_asc, na_position="last")
                .drop(columns="__rank__")
                .reset_index(drop=True)
            )

            try:
                sty_o = view_o.style
                pl_cols_o = [c for c in ("total_pl", "unrealized_pl") if c in view_o.columns]
                if pl_cols_o:
                    sty_o = sty_o.applymap(_pi_color_pl, subset=pl_cols_o)
                    sty_o = sty_o.format({c: "{:,.2f}" for c in pl_cols_o})
                if "risk_flag" in view_o.columns:
                    sty_o = sty_o.applymap(_pi_risk_flag_cell_style, subset=["risk_flag"])
                st.dataframe(sty_o, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Risk overlay styling unavailable: {_short_err(e)}")
                st.dataframe(view_o, use_container_width=True, hide_index=True)

            st.caption(
                "Sort order: FORCE_EXIT → TRIM_PRIORITY → BLOCK_NEW_BUY → OK "
                "(then by `total_pl` ascending within each tier)."
            )
        except Exception as e:
            st.warning(
                f"Risk overlay rendering unavailable: {_short_err(e)}",
                icon="⚠️",
            )


# ──────────────────────────────
# ROUTER (single source of truth)
# ──────────────────────────────
PAGE_REGISTRY: Dict[Tuple[str, str], Callable[[], None]] = {
    ("Portfolio", "Portfolio History"): page_portfolio_history,
    ("Portfolio", "Positions & Exposure"): page_positions_exposure,
    ("Portfolio", "Trade Log"): page_trade_log,
    ("Signals", "🧭 Signal Lifecycle"): page_signal_lifecycle,
    ("Signals", "📜 Ticker Decision Timeline"): page_ticker_decision_timeline,
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
    ("System", "⚙️ Execution Intelligence"): page_execution_intelligence,
    ("System", "🧠 Feedback Intelligence"): page_feedback_intelligence,
    ("System", "🛠️ Adaptation Intelligence"): page_adaptation_intelligence,
    ("System", "✅ Applied Adjustments"): page_applied_adjustments,
    ("System", "🧪 Adaptation Simulation"): page_adaptation_simulation,
    ("System", "📊 Performance Intelligence"): page_performance_intelligence,
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
        "📜 Ticker Decision Timeline",
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
        "⚙️ Execution Intelligence",
        "🧠 Feedback Intelligence",
        "🛠️ Adaptation Intelligence",
        "✅ Applied Adjustments",
        "🧪 Adaptation Simulation",
        "📊 Performance Intelligence",
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
