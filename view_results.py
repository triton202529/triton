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

# Risk Office (Capital Preservation Doctrine stack — read-only watchdog artifacts)
CPI_JSON_PATH = RESULTS_DIR / "capital_preservation_intelligence.json"
CPE_JSON_PATH = RESULTS_DIR / "capital_preservation_escalation.json"
CPA_JSON_PATH = RESULTS_DIR / "capital_preservation_advisory.json"
CPD_JSON_PATH = RESULTS_DIR / "capital_preservation_decision_support.json"
GOV_RISK_SUMMARY_PATH = RESULTS_DIR / "governance_risk_summary.json"
WATCHDOG_ALERTS_PATH = RESULTS_DIR / "watchdog_alerts.json"
WATCHDOG_STATUS_PATH = RESULTS_DIR / "watchdog_status.json"

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

# Governance Command Center (Phase 2 — read-only observability)
GCC_READINESS_SUMMARY_PATH = RESULTS_DIR / "arm_runtime_governance_readiness_gate_summary.json"
GCC_ADMISSION_SUMMARY_PATH = RESULTS_DIR / "arm_runtime_governance_admission_board_summary.json"
GCC_ELIGIBILITY_SUMMARY_PATH = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board_summary.json"
)
GCC_RECOMMENDATION_SUMMARY_PATH = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine_summary.json"
)
GCC_REVIEW_SUMMARY_PATH = (
    RESULTS_DIR / "arm_runtime_governance_enablement_review_board_summary.json"
)
GCC_VERDICT_SUMMARY_PATH = (
    RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine_summary.json"
)
GCC_DOSSIER_SUMMARY_PATH = (
    RESULTS_DIR / "arm_runtime_governance_human_escalation_dossier_summary.json"
)
GCC_DOSSIER_JSON_PATH = RESULTS_DIR / "arm_runtime_governance_human_escalation_dossier.json"
GCC_READINESS_MEM_PATH = RESULTS_DIR / "arm_runtime_governance_readiness_gate_memory.csv"
GCC_ADMISSION_MEM_PATH = RESULTS_DIR / "arm_runtime_governance_admission_board_memory.csv"
GCC_ELIGIBILITY_MEM_PATH = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board_memory.csv"
)
GCC_RECOMMENDATION_MEM_PATH = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine_memory.csv"
)
GCC_REVIEW_MEM_PATH = RESULTS_DIR / "arm_runtime_governance_enablement_review_board_memory.csv"
GCC_VERDICT_MEM_PATH = (
    RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine_memory.csv"
)
GCC_DOSSIER_MEM_PATH = RESULTS_DIR / "arm_runtime_governance_human_escalation_dossier_memory.csv"

# Governance Library Center (Step 138 — read-only documentation UI)
GOVERNANCE_DOCS_DIR = PROJECT_ROOT / "docs" / "governance"

# Governance Operations Platform (Steps 139–146 — read-only CSV registries)
from dashboard import governance_operations as _gov_ops

load_governance_csv = _gov_ops.load_governance_csv
normalize_governance_status = _gov_ops.normalize_governance_status
safe_parse_datetime = _gov_ops.safe_parse_datetime
compute_record_age_days = _gov_ops.compute_record_age_days
filter_dataframe_by_search = _gov_ops.filter_dataframe_by_search
render_governance_kpi_card = _gov_ops.render_governance_kpi_card
render_selected_record = _gov_ops.render_selected_record
build_governance_timeline = _gov_ops.build_governance_timeline
page_governance_evidence_registry = _gov_ops.page_governance_evidence_registry
page_governance_audit_center = _gov_ops.page_governance_audit_center
page_governance_decision_registry = _gov_ops.page_governance_decision_registry
page_governance_escalation_registry = _gov_ops.page_governance_escalation_registry
page_governance_traceability_explorer = _gov_ops.page_governance_traceability_explorer
page_governance_investigation_center = _gov_ops.page_governance_investigation_center
page_governance_intelligence_lab = _gov_ops.page_governance_intelligence_lab
page_governance_timeline_center = _gov_ops.page_governance_timeline_center
render_gcc_operations_overview = _gov_ops.render_gcc_operations_overview

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


def page_risk_office() -> None:
    """🛡️ Risk Office — capital preservation posture (read-only)."""
    try:
        from dashboard.risk_office import render_risk_office_dashboard
        from dashboard.advanced_risk_intelligence import render_predictive_risk_section

        render_risk_office_dashboard(RESULTS_DIR)
        st.markdown("---")
        render_predictive_risk_section(RESULTS_DIR)
    except Exception as e:
        st.error(f"Risk Office dashboard failed to load: {_short_err(e)}")


def page_defensive_simulation_lab() -> None:
    """🧪 Defensive Simulation Lab — counterfactual risk controls (read-only)."""
    try:
        from dashboard.advanced_risk_intelligence import render_defensive_simulation_lab

        render_defensive_simulation_lab(RESULTS_DIR)
    except Exception as e:
        st.error(f"Defensive Simulation Lab failed to load: {_short_err(e)}")


def page_executive_risk_command_center() -> None:
    """🏛 Executive Risk Command Center — leadership briefing (read-only)."""
    try:
        from dashboard.advanced_risk_intelligence import render_executive_risk_command_center

        render_executive_risk_command_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Executive Risk Command Center failed to load: {_short_err(e)}")


def page_defensive_automation_sandbox() -> None:
    """🧱 Defensive Automation Sandbox — hypothetical actions only (no execution)."""
    try:
        from dashboard.activation_safety import render_defensive_automation_sandbox

        render_defensive_automation_sandbox(RESULTS_DIR)
    except Exception as e:
        st.error(f"Defensive Automation Sandbox failed to load: {_short_err(e)}")


def page_human_approval_center() -> None:
    """👤 Human Approval Center — queue status only (no execution)."""
    try:
        from dashboard.activation_safety import render_human_approval_center

        render_human_approval_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Human Approval Center failed to load: {_short_err(e)}")


def page_protective_action_policy_center() -> None:
    """🛡️ Protective Action Policy Center — policy definitions (all disabled)."""
    try:
        from dashboard.activation_safety import render_protective_action_policy_center

        render_protective_action_policy_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Protective Action Policy Center failed to load: {_short_err(e)}")


def page_governance_authorization_center() -> None:
    """🏛 Governance Authorization Center — four-layer authorization gate (read-only)."""
    try:
        from dashboard.governance_execution_readiness import render_governance_authorization_center

        render_governance_authorization_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Governance Authorization Center failed to load: {_short_err(e)}")


def page_execution_readiness_center() -> None:
    """⚙️ Execution Readiness Center — eligibility assessment (paper mode only)."""
    try:
        from dashboard.governance_execution_readiness import render_execution_readiness_center

        render_execution_readiness_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Execution Readiness Center failed to load: {_short_err(e)}")


def page_protective_action_trials() -> None:
    """🧪 Protective Action Trials — paper-mode simulations only."""
    try:
        from dashboard.governance_execution_readiness import render_protective_action_trials

        render_protective_action_trials(RESULTS_DIR)
    except Exception as e:
        st.error(f"Protective Action Trials failed to load: {_short_err(e)}")


def page_protective_action_evaluation() -> None:
    """📊 Protective Action Evaluation — trial effectiveness scoring (read-only)."""
    try:
        from dashboard.capital_preservation_evaluation import render_protective_action_evaluation

        render_protective_action_evaluation(RESULTS_DIR)
    except Exception as e:
        st.error(f"Protective Action Evaluation failed to load: {_short_err(e)}")


def page_adaptive_capital_preservation() -> None:
    """🧠 Adaptive Capital Preservation — learn from simulations (no actions)."""
    try:
        from dashboard.capital_preservation_evaluation import render_adaptive_capital_preservation

        render_adaptive_capital_preservation(RESULTS_DIR)
    except Exception as e:
        st.error(f"Adaptive Capital Preservation failed to load: {_short_err(e)}")


def page_capital_preservation_governor() -> None:
    """👑 Capital Preservation Governor — unified preservation posture (advisory)."""
    try:
        from dashboard.capital_preservation_evaluation import render_capital_preservation_governor

        render_capital_preservation_governor(RESULTS_DIR)
    except Exception as e:
        st.error(f"Capital Preservation Governor failed to load: {_short_err(e)}")


def page_capital_preservation_audit_center() -> None:
    """📋 Capital Preservation Audit Center — unified preservation audit trail."""
    try:
        from dashboard.institutional_autonomy import render_capital_preservation_audit_center

        render_capital_preservation_audit_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Capital Preservation Audit Center failed to load: {_short_err(e)}")


def page_preservation_stress_lab() -> None:
    """🧪 Preservation Stress Lab — counterfactual stress scenarios (simulation only)."""
    try:
        from dashboard.institutional_autonomy import render_preservation_stress_lab

        render_preservation_stress_lab(RESULTS_DIR)
    except Exception as e:
        st.error(f"Preservation Stress Lab failed to load: {_short_err(e)}")


def page_preservation_certification_center() -> None:
    """🏅 Preservation Certification Center — paper-mode certification scoring."""
    try:
        from dashboard.institutional_autonomy import render_preservation_certification_center

        render_preservation_certification_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Preservation Certification Center failed to load: {_short_err(e)}")


def page_risk_committee_oversight() -> None:
    """🏛 Risk Committee Oversight — committee assessments across preservation domains."""
    try:
        from dashboard.institutional_protection import render_risk_committee_oversight

        render_risk_committee_oversight(RESULTS_DIR)
    except Exception as e:
        st.error(f"Risk Committee Oversight failed to load: {_short_err(e)}")


def page_accountability_registry() -> None:
    """📑 Accountability Registry — protective decision path traceability."""
    try:
        from dashboard.institutional_protection import render_accountability_registry

        render_accountability_registry(RESULTS_DIR)
    except Exception as e:
        st.error(f"Accountability Registry failed to load: {_short_err(e)}")


def page_preservation_governance_board() -> None:
    """👑 Preservation Governance Board — unified advisory authority layer."""
    try:
        from dashboard.institutional_protection import render_preservation_governance_board

        render_preservation_governance_board(RESULTS_DIR)
    except Exception as e:
        st.error(f"Preservation Governance Board failed to load: {_short_err(e)}")


def page_investment_committee_review() -> None:
    """🏛 Investment Committee Review — portfolio, risk, governance, certification, readiness."""
    try:
        from dashboard.institutional_operations import render_investment_committee_review

        render_investment_committee_review(RESULTS_DIR)
    except Exception as e:
        st.error(f"Investment Committee Review failed to load: {_short_err(e)}")


def page_triton_maturity_assessment() -> None:
    """📈 Triton Maturity Assessment — institutional maturity scoring."""
    try:
        from dashboard.institutional_operations import render_triton_maturity_assessment

        render_triton_maturity_assessment(RESULTS_DIR)
    except Exception as e:
        st.error(f"Triton Maturity Assessment failed to load: {_short_err(e)}")


def page_strategic_oversight_center() -> None:
    """🎯 Strategic Oversight Center — unified strategic advisory view."""
    try:
        from dashboard.institutional_operations import render_strategic_oversight_center

        render_strategic_oversight_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Strategic Oversight Center failed to load: {_short_err(e)}")


def page_decision_quality_center() -> None:
    """🧩 Decision Quality Center — advisory and escalation consistency scoring."""
    try:
        from dashboard.institutional_intelligence import render_decision_quality_center

        render_decision_quality_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Decision Quality Center failed to load: {_short_err(e)}")


def page_institutional_intelligence() -> None:
    """🏛 Institutional Intelligence — cross-layer institutional scoring."""
    try:
        from dashboard.institutional_intelligence import render_institutional_intelligence

        render_institutional_intelligence(RESULTS_DIR)
    except Exception as e:
        st.error(f"Institutional Intelligence failed to load: {_short_err(e)}")


def page_strategic_self_improvement() -> None:
    """🚀 Strategic Self-Improvement — prioritized enhancement advisory."""
    try:
        from dashboard.institutional_intelligence import render_strategic_self_improvement

        render_strategic_self_improvement(RESULTS_DIR)
    except Exception as e:
        st.error(f"Strategic Self-Improvement failed to load: {_short_err(e)}")


def page_institutional_memory() -> None:
    """🧠 Institutional Memory — persistent organizational event memory."""
    try:
        from dashboard.institutional_memory import render_institutional_memory

        render_institutional_memory(RESULTS_DIR)
    except Exception as e:
        st.error(f"Institutional Memory failed to load: {_short_err(e)}")


def page_institutional_knowledge_graph() -> None:
    """🕸 Institutional Knowledge Graph — component relationship map."""
    try:
        from dashboard.institutional_memory import render_institutional_knowledge_graph

        render_institutional_knowledge_graph(RESULTS_DIR)
    except Exception as e:
        st.error(f"Institutional Knowledge Graph failed to load: {_short_err(e)}")


def page_organizational_learning_center() -> None:
    """📚 Organizational Learning Center — pattern analysis and lessons."""
    try:
        from dashboard.institutional_memory import render_organizational_learning_center

        render_organizational_learning_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Organizational Learning Center failed to load: {_short_err(e)}")


def page_strategic_reasoning_center() -> None:
    """♟ Strategic Reasoning Center — strategic importance ranking."""
    try:
        from dashboard.strategic_intelligence import render_strategic_reasoning_center

        render_strategic_reasoning_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Strategic Reasoning Center failed to load: {_short_err(e)}")


def page_consequence_forecast_center() -> None:
    """🔮 Consequence Forecast Center — 90-day institutional projections."""
    try:
        from dashboard.strategic_intelligence import render_consequence_forecast_center

        render_consequence_forecast_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Consequence Forecast Center failed to load: {_short_err(e)}")


def page_institutional_wisdom_center() -> None:
    """📜 Institutional Wisdom Center — long-term advisory guidance."""
    try:
        from dashboard.strategic_intelligence import render_institutional_wisdom_center

        render_institutional_wisdom_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Institutional Wisdom Center failed to load: {_short_err(e)}")


def page_scenario_planning_center() -> None:
    """🗺 Scenario Planning Center — institutional scenario lenses."""
    try:
        from dashboard.institutional_planning import render_scenario_planning_center

        render_scenario_planning_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Scenario Planning Center failed to load: {_short_err(e)}")


def page_future_path_analysis_center() -> None:
    """🛣 Future Path Analysis — trajectory evaluation."""
    try:
        from dashboard.institutional_planning import render_future_path_analysis_center

        render_future_path_analysis_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Future Path Analysis failed to load: {_short_err(e)}")


def page_strategic_priorities_center() -> None:
    """🎯 Strategic Priorities Center — ranked institutional objectives."""
    try:
        from dashboard.institutional_planning import render_strategic_priorities_center

        render_strategic_priorities_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Strategic Priorities Center failed to load: {_short_err(e)}")


def page_attention_allocation_center() -> None:
    """🎯 Attention Allocation Center — institutional focus scoring."""
    try:
        from dashboard.institutional_optimization import render_attention_allocation_center

        render_attention_allocation_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Attention Allocation Center failed to load: {_short_err(e)}")


def page_system_coordination_center() -> None:
    """🔗 System Coordination Center — cross-system alignment."""
    try:
        from dashboard.institutional_optimization import render_system_coordination_center

        render_system_coordination_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"System Coordination Center failed to load: {_short_err(e)}")


def page_institutional_optimization_center() -> None:
    """⚡ Institutional Optimization Center — optimization opportunities."""
    try:
        from dashboard.institutional_optimization import render_institutional_optimization_center

        render_institutional_optimization_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Institutional Optimization Center failed to load: {_short_err(e)}")


def page_causal_reasoning_center() -> None:
    """🔍 Causal Reasoning Center — cause-effect analysis for institutional issues."""
    try:
        from dashboard.institutional_reasoning import render_causal_reasoning_center

        render_causal_reasoning_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Causal Reasoning Center failed to load: {_short_err(e)}")


def page_explainability_center() -> None:
    """📖 Explainability Center — plain-language institutional explanations."""
    try:
        from dashboard.institutional_reasoning import render_explainability_center

        render_explainability_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Explainability Center failed to load: {_short_err(e)}")


def page_institutional_insights_center() -> None:
    """💡 Institutional Insights — strategic observations and synthesized headlines."""
    try:
        from dashboard.institutional_reasoning import render_institutional_insights_center

        render_institutional_insights_center(RESULTS_DIR)
    except Exception as e:
        st.error(f"Institutional Insights failed to load: {_short_err(e)}")


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
# GOVERNANCE COMMAND CENTER (Phase 2 — read-only)
# ──────────────────────────────

_GCC_MATURITY_TOKENS: Tuple[str, ...] = (
    "INSTITUTIONAL",
    "READY",
    "LIMITED",
    "OBSERVE",
    "DORMANT",
)

_GCC_LADDER_STAGES: List[Tuple[str, str, str, str]] = [
    ("Readiness", "readiness_state", "runtime_readiness_classification", "readiness_confidence"),
    ("Admission", "admission_state", "runtime_admission_classification", "admission_confidence"),
    (
        "Constitutional Eligibility",
        "constitutional_eligibility_state",
        "runtime_constitutional_eligibility_classification",
        "constitutional_eligibility_confidence",
    ),
    (
        "Recommendation",
        "recommendation_state",
        "runtime_enablement_recommendation_classification",
        "recommendation_confidence",
    ),
    (
        "Review",
        "review_state",
        "runtime_enablement_review_classification",
        "review_confidence",
    ),
    (
        "Verdict",
        "verdict_state",
        "runtime_verdict_classification",
        "verdict_confidence",
    ),
    (
        "Human Escalation",
        "dossier_state",
        "human_escalation_classification",
        "escalation_confidence",
    ),
]

_GCC_TIMELINE_SERIES: List[Tuple[str, Path, str]] = [
    ("Readiness", GCC_READINESS_MEM_PATH, "readiness_confidence"),
    ("Admission", GCC_ADMISSION_MEM_PATH, "admission_confidence"),
    ("Eligibility", GCC_ELIGIBILITY_MEM_PATH, "constitutional_eligibility_confidence"),
    ("Recommendation", GCC_RECOMMENDATION_MEM_PATH, "recommendation_confidence"),
    ("Review", GCC_REVIEW_MEM_PATH, "review_confidence"),
    ("Verdict", GCC_VERDICT_MEM_PATH, "verdict_confidence"),
    ("Escalation", GCC_DOSSIER_MEM_PATH, "escalation_confidence"),
]

_GCC_HIST_SERIES: List[Tuple[str, Path, str, str]] = [
    ("Readiness", GCC_READINESS_MEM_PATH, "readiness_state", "readiness_confidence"),
    ("Admission", GCC_ADMISSION_MEM_PATH, "admission_state", "admission_confidence"),
    (
        "Eligibility",
        GCC_ELIGIBILITY_MEM_PATH,
        "constitutional_eligibility_state",
        "constitutional_eligibility_confidence",
    ),
    (
        "Recommendation",
        GCC_RECOMMENDATION_MEM_PATH,
        "recommendation_state",
        "recommendation_confidence",
    ),
    ("Review", GCC_REVIEW_MEM_PATH, "review_state", "review_confidence"),
    ("Verdict", GCC_VERDICT_MEM_PATH, "verdict_state", "verdict_confidence"),
    ("Human Escalation", GCC_DOSSIER_MEM_PATH, "dossier_state", "escalation_confidence"),
]


def _gcc_fmt_conf(value: Any, default: str = "0.00") -> str:
    try:
        if value is None:
            return default
        return f"{float(value):.2f}"
    except Exception:
        return default


def _gcc_maturity_badge(state: Any) -> str:
    s = str(state or "").upper()
    for token in _GCC_MATURITY_TOKENS:
        if token in s:
            return "Institutional" if token == "INSTITUTIONAL" else token.capitalize()
    return "Unknown"


def _gcc_card_tone(classification: Any, state: Any) -> str:
    cls = str(classification or "").upper()
    st_name = str(state or "").upper()
    favorable = any(
        k in cls
        for k in (
            "FAVORABLE",
            "FULL",
            "ADMITTED",
            "ELIGIBLE",
            "RECOMMENDED",
            "UNDER_REVIEW",
            "RUNTIME_READY",
        )
    )
    if favorable and "NOT_" not in cls and "DO_NOT" not in cls:
        return "success"
    if "OBSERVE" in cls or "OBSERVE" in st_name:
        return "info"
    if "LIMITED" in cls or "LIMITED" in st_name:
        return "warning"
    return "warning" if "DORMANT" in st_name or "NOT_" in cls or "DO_NOT" in cls else "info"


def _gcc_badge_tone(classification: Any, state: Any) -> str:
    return _gcc_card_tone(classification, state)


def _gcc_badge_style(tone: str) -> str:
    if tone == "success":
        return "border-color:rgba(34,197,94,.55);color:rgba(134,239,172,.95);"
    if tone == "warning":
        return "border-color:rgba(234,179,8,.55);color:rgba(253,224,71,.95);"
    return "border-color:rgba(96,165,250,.45);color:rgba(147,197,253,.95);"


_GCC_UX_CSS = """
<style>
.gcc-status-card{border:1px solid rgba(148,163,184,.28);border-radius:10px;padding:.65rem .75rem .55rem;margin-bottom:.2rem;background:rgba(15,23,42,.38);}
.gcc-status-card .gcc-label{font-size:.76rem;opacity:.82;font-weight:600;margin:0 0 .4rem;letter-spacing:.03em;text-transform:uppercase;}
.gcc-status-card .gcc-class{font-size:1.08rem;font-weight:700;line-height:1.22;word-break:break-word;margin:0 0 .28rem;}
.gcc-status-card .gcc-conf{font-size:.8rem;opacity:.78;margin:0 0 .32rem;}
.gcc-status-card .gcc-badge{display:inline-block;font-size:.68rem;font-weight:700;letter-spacing:.08em;padding:.14rem .42rem;border-radius:4px;border:1px solid rgba(148,163,184,.4);margin:0 0 .28rem;}
.gcc-status-card .gcc-state{font-size:.66rem;opacity:.48;font-family:ui-monospace,monospace;word-break:break-all;margin:0;line-height:1.2;}
.gcc-ladder-wrap{border:1px solid rgba(148,163,184,.22);border-radius:10px;padding:.45rem .65rem;background:rgba(15,23,42,.28);}
.gcc-ladder{font-size:.84rem;line-height:1.2;margin:0;}
.gcc-ladder-row{display:flex;align-items:center;justify-content:space-between;gap:.65rem;padding:.14rem 0;}
.gcc-ladder-left{flex:1;min-width:0;display:flex;align-items:baseline;gap:.55rem;}
.gcc-ladder-stage{font-weight:600;white-space:nowrap;min-width:8.5rem;}
.gcc-ladder-detail{opacity:.72;font-size:.76rem;font-family:ui-monospace,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.gcc-ladder-badge{font-size:.68rem;font-weight:700;letter-spacing:.06em;padding:.08rem .38rem;border-radius:4px;border:1px solid rgba(148,163,184,.35);white-space:nowrap;}
.gcc-ladder-arrow{text-align:center;opacity:.42;font-size:.72rem;line-height:1;margin:0;padding:0;}
.gcc-block-group{margin:.35rem 0 .15rem;font-size:.88rem;font-weight:600;opacity:.92;}
.gcc-block-item{margin:.08rem 0 .08rem .15rem;font-size:.86rem;opacity:.88;}
.gcc-hist-wrap{border:1px solid rgba(148,163,184,.22);border-radius:10px;padding:.55rem .75rem;background:rgba(15,23,42,.30);margin:.35rem 0;}
.gcc-hist-title{font-size:.78rem;font-weight:600;opacity:.82;text-transform:uppercase;letter-spacing:.04em;margin:0 0 .25rem;}
.gcc-hist-value{font-size:.95rem;font-weight:600;line-height:1.3;margin:0 0 .18rem;}
.gcc-hist-detail{font-size:.82rem;opacity:.72;line-height:1.35;margin:0;}
.gcc-hist-metric{font-size:.84rem;opacity:.88;margin:.08rem 0;}
</style>
"""


def _gcc_render_card(
    title: str,
    classification: Any,
    confidence: Any,
    state: Any,
) -> None:
    badge = _gcc_maturity_badge(state).upper()
    tone = _gcc_badge_tone(classification, state)
    badge_style = _gcc_badge_style(tone)
    cls_text = str(classification or "—")
    st.markdown(
        f"""
        <div class="gcc-status-card">
          <div class="gcc-label">{title}</div>
          <div class="gcc-class">{cls_text}</div>
          <div class="gcc-conf">confidence {_gcc_fmt_conf(confidence)}</div>
          <div class="gcc-badge" style="{badge_style}">{badge}</div>
          <div class="gcc-state">state: {state or "—"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _gcc_get(d: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _gcc_collect_blocked_reason_groups(
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
) -> List[Tuple[str, List[str]]]:
    groups: Dict[str, List[str]] = {
        "Governance maturity": [],
        "Institutional authorization": [],
        "Constitutional concerns": [],
        "Final institutional position": [],
    }
    seen: set = set()

    def _add(group: str, text: Any) -> None:
        t = str(text or "").strip()
        if not t or t in seen or group not in groups:
            return
        seen.add(t)
        groups[group].append(t)

    ladder_checks = [
        (
            _gcc_get(readiness, "runtime_readiness_classification"),
            "NOT_RUNTIME_READY",
            "Governance maturity",
            "Runtime governance is not institutionally ready.",
        ),
        (
            _gcc_get(admission, "runtime_admission_classification"),
            "NOT_RUNTIME_ADMITTED",
            "Institutional authorization",
            "Runtime governance has not been admitted.",
        ),
        (
            _gcc_get(eligibility, "runtime_constitutional_eligibility_classification"),
            "NOT_CONSTITUTIONALLY_ELIGIBLE",
            "Institutional authorization",
            "Runtime governance is not constitutionally eligible.",
        ),
        (
            _gcc_get(recommendation, "runtime_enablement_recommendation_classification"),
            "NOT_RECOMMENDED",
            "Institutional authorization",
            "Runtime enablement recommendation not earned.",
        ),
        (
            _gcc_get(review, "runtime_enablement_review_classification"),
            "NOT_UNDER_REVIEW",
            "Institutional authorization",
            "Formal runtime enablement review is not active.",
        ),
        (
            _gcc_get(verdict, "runtime_verdict_classification"),
            "DO_NOT_ENABLE_RUNTIME",
            "Final institutional position",
            "Triton's institutional verdict does not support runtime governance enablement.",
        ),
        (
            _gcc_get(dossier_summary, "human_escalation_classification"),
            "NO_ESCALATION",
            "Final institutional position",
            "Human escalation is unnecessary.",
        ),
    ]
    for cls, blocked_token, group, msg in ladder_checks:
        if str(cls or "").upper() == blocked_token:
            _add(group, msg)

    if _gcc_get(dossier_summary, "constitutional_review_required") is True:
        _add("Constitutional concerns", "Constitutional review remains required.")
    if _gcc_get(verdict, "constitutional_review_required") is True:
        _add("Constitutional concerns", "Constitutional pressure remains elevated.")

    dossier = _gcc_get(dossier_record, "human_escalation_dossier") or {}
    exec_sum = str(_gcc_get(dossier, "executive_summary") or "")
    if exec_sum:
        low = exec_sum.lower()
        if any(k in low for k in ("maturity", "insufficient", "immature", "shadow")):
            if "governance maturity remains insufficient" in low:
                _add("Governance maturity", "Governance maturity remains insufficient.")
            elif "shadow governance reliability remains immature" in low:
                _add("Governance maturity", "Shadow governance reliability remains immature.")
            else:
                _add("Governance maturity", exec_sum)

    for item in _gcc_get(dossier_record, "dossier_reasons") or []:
        text = str(item or "")
        low = text.lower()
        if any(k in low for k in ("constitutional", "court", "violation")):
            _add("Constitutional concerns", text)
        elif any(k in low for k in ("maturity", "ready", "shadow", "immature")):
            _add("Governance maturity", text)
        else:
            _add("Final institutional position", text)

    rationale = _gcc_get(dossier, "human_escalation_rationale")
    if rationale:
        _add("Final institutional position", rationale)

    for item in _gcc_get(dossier, "case_against_runtime") or []:
        text = str(item or "")
        low = text.lower()
        if any(k in low for k in ("constitutional", "court", "overrul")):
            _add("Constitutional concerns", text)
        elif any(k in low for k in ("shadow", "readiness", "maturity", "immature", "confidence")):
            _add("Governance maturity", text)
        elif any(k in low for k in ("verdict", "enablement", "mutate", "communication")):
            _add("Final institutional position", text)
        else:
            _add("Governance maturity", text)

    ordered: List[Tuple[str, List[str]]] = []
    for name in (
        "Governance maturity",
        "Institutional authorization",
        "Constitutional concerns",
        "Final institutional position",
    ):
        items = groups[name]
        if items:
            ordered.append((name, items))

    if not ordered:
        ordered.append(
            (
                "Final institutional position",
                [
                    "Runtime governance posture is under institutional review. "
                    "Detailed block rationale is not yet available — treat runtime as locked."
                ],
            )
        )
    return ordered


def _gcc_render_ladder(
    stages: List[Tuple[str, Any, Any, Any]],
) -> None:
    rows_html: List[str] = []
    for idx, (label, classification, confidence, state) in enumerate(stages):
        badge = _gcc_maturity_badge(state).upper()
        tone = _gcc_badge_tone(classification, state)
        badge_style = _gcc_badge_style(tone)
        cls_text = str(classification or "—")
        arrow = '<div class="gcc-ladder-arrow">↓</div>' if idx < len(stages) - 1 else ""
        rows_html.append(
            f'<div class="gcc-ladder-row">'
            f'<div class="gcc-ladder-left">'
            f'<span class="gcc-ladder-stage">{label}</span>'
            f'<span class="gcc-ladder-detail">{cls_text} · {_gcc_fmt_conf(confidence)}</span>'
            f"</div>"
            f'<span class="gcc-ladder-badge" style="{badge_style}">{badge}</span>'
            f"</div>"
            f"{arrow}"
        )
    ladder_html = (
        f'<div class="gcc-ladder-wrap"><div class="gcc-ladder">{"".join(rows_html)}</div></div>'
    )
    st.markdown(ladder_html, unsafe_allow_html=True)


def _gcc_build_confidence_timeline() -> Tuple[Optional[Any], Optional[float]]:
    rows: List[Dict[str, Any]] = []
    for label, path, conf_col in _GCC_TIMELINE_SERIES:
        df = _ad_load_csv(path, f"{label} memory")
        if df is None or df.empty or "timestamp" not in df.columns or conf_col not in df.columns:
            continue
        try:
            sub = df[["timestamp", conf_col]].copy()
            sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce", utc=True)
            sub[conf_col] = pd.to_numeric(sub[conf_col], errors="coerce")
            sub = sub.dropna(subset=["timestamp"])
            if sub.empty:
                continue
            sub["stage"] = label
            sub = sub.rename(columns={conf_col: "confidence"})
            rows.extend(sub[["timestamp", "stage", "confidence"]].to_dict("records"))
        except Exception:
            continue
    if not rows:
        return None, None
    plot_df = pd.DataFrame(rows)
    max_conf: Optional[float] = None
    try:
        max_conf = float(pd.to_numeric(plot_df["confidence"], errors="coerce").max())
    except Exception:
        max_conf = None
    try:
        import plotly.express as px  # type: ignore

        fig = px.line(
            plot_df,
            x="timestamp",
            y="confidence",
            color="stage",
            title="Governance Confidence Timeline",
            markers=True,
        )
        fig.update_layout(
            yaxis_title="Confidence",
            xaxis_title="Timestamp (UTC)",
            legend_title="Stage",
            height=420,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig.update_yaxes(range=[0, 1])
        return fig, max_conf
    except Exception:
        return None, max_conf


def _gcc_maturity_rank(state: Any) -> int:
    s = str(state or "").upper()
    order = {"DORMANT": 0, "OBSERVE": 1, "LIMITED": 2, "READY": 3, "INSTITUTIONAL": 4}
    best = -1
    for token, rank in order.items():
        if token in s:
            best = max(best, rank)
    return best


def _gcc_fmt_conf_delta(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "0.00"
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v):.2f}"


def _gcc_load_hist_memory(
    stage: str, path: Path, state_col: str, conf_col: str
) -> Optional[pd.DataFrame]:
    df = _ad_load_csv(path, f"{stage} memory")
    if df is None or df.empty:
        return None
    if "timestamp" not in df.columns or state_col not in df.columns or conf_col not in df.columns:
        return None
    try:
        out = df[["timestamp", state_col, conf_col]].copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out[state_col] = out[state_col].astype(str)
        out[conf_col] = pd.to_numeric(out[conf_col], errors="coerce").fillna(0.0)
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return out if not out.empty else None
    except Exception:
        return None


def _gcc_transition_interpretation(
    stage: str,
    prev_state: Any,
    curr_state: Any,
    conf_delta: float,
) -> str:
    prev_rank = _gcc_maturity_rank(prev_state)
    curr_rank = _gcc_maturity_rank(curr_state)
    if str(prev_state) != str(curr_state):
        if curr_rank > prev_rank:
            return f"{stage} strengthened"
        if curr_rank < prev_rank:
            return f"{stage} weakened"
        return f"{stage} posture shifted"
    if conf_delta > 0.001:
        return f"{stage} confidence increased"
    if conf_delta < -0.001:
        return f"{stage} confidence decreased"
    return f"{stage} remained stable"


def _gcc_analyze_governance_history(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
) -> Dict[str, Any]:
    transitions: List[Dict[str, Any]] = []
    changes: List[str] = []
    conf_deltas: List[float] = []
    state_changes = 0
    loaded_stages = 0

    for stage, path, state_col, conf_col in _GCC_HIST_SERIES:
        mem = _gcc_load_hist_memory(stage, path, state_col, conf_col)
        if mem is None:
            continue
        loaded_stages += 1
        if len(mem) == 1:
            row = mem.iloc[0]
            changes.append(
                f"{stage} posture remained stable ({_gcc_maturity_badge(row[state_col])})."
            )
            conf_deltas.append(0.0)
            continue

        first = mem.iloc[0]
        last = mem.iloc[-1]
        span_delta = float(last[conf_col]) - float(first[conf_col])
        conf_deltas.append(span_delta)

        if str(first[state_col]) != str(last[state_col]):
            changes.append(
                f"Governance advanced from {_gcc_maturity_badge(first[state_col])} → "
                f"{_gcc_maturity_badge(last[state_col])} at {stage.lower()}."
            )
        elif abs(span_delta) >= 0.001:
            if span_delta > 0:
                changes.append(
                    f"{stage} confidence increased by {_gcc_fmt_conf_delta(span_delta)}."
                )
            else:
                changes.append(f"{stage} confidence weakened by {_gcc_fmt_conf_delta(span_delta)}.")
        else:
            changes.append(f"{stage} posture remained stable.")

        for i in range(1, len(mem)):
            prev = mem.iloc[i - 1]
            curr = mem.iloc[i]
            delta = float(curr[conf_col]) - float(prev[conf_col])
            prev_state = prev[state_col]
            curr_state = curr[state_col]
            if str(prev_state) == str(curr_state) and abs(delta) < 0.001:
                continue
            if str(prev_state) != str(curr_state):
                state_changes += 1
            ts = curr["timestamp"]
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
            transitions.append(
                {
                    "Timestamp": ts_str,
                    "Stage": stage,
                    "Previous": _gcc_maturity_badge(prev_state),
                    "Current": _gcc_maturity_badge(curr_state),
                    "Confidence Δ": _gcc_fmt_conf_delta(delta),
                    "Interpretation": _gcc_transition_interpretation(
                        stage, prev_state, curr_state, delta
                    ),
                }
            )

    max_conf = 0.0
    for stage, path, state_col, conf_col in _GCC_HIST_SERIES:
        mem = _gcc_load_hist_memory(stage, path, state_col, conf_col)
        if mem is not None:
            try:
                max_conf = max(max_conf, float(mem[conf_col].max()))
            except Exception:
                pass

    if max_conf <= 0.01 and state_changes == 0:
        confidence_direction = "dormant"
    else:
        pos = sum(1 for d in conf_deltas if d > 0.001)
        neg = sum(1 for d in conf_deltas if d < -0.001)
        zero = len(conf_deltas) - pos - neg
        if pos > 0 and neg > 0:
            confidence_direction = "mixed"
        elif pos > 0 and neg == 0:
            confidence_direction = "improving"
        elif neg > 0 and pos == 0:
            confidence_direction = "deteriorating"
        elif zero == len(conf_deltas) or not conf_deltas:
            confidence_direction = "stable"
        else:
            confidence_direction = "stable"

    total_movement = sum(abs(d) for d in conf_deltas)
    if state_changes >= 2 or total_movement >= 0.15:
        momentum = "HIGH"
    elif state_changes >= 1 or total_movement >= 0.05:
        momentum = "MODERATE"
    elif total_movement >= 0.01 or len(transitions) >= 1:
        momentum = "LOW"
    else:
        momentum = "NONE"

    if state_changes == 0:
        posture_stability = "Stable institutional posture"
    elif state_changes <= 2:
        posture_stability = "Limited governance posture shifts detected"
    else:
        posture_stability = "Frequent governance posture shifts detected"

    if confidence_direction == "dormant":
        posture_trend = "Governance posture remains dormant."
        posture_detail = "No institutional progression has yet occurred."
    elif confidence_direction == "improving":
        posture_trend = "Governance posture is improving."
        improving = [c for c in changes if "increased" in c.lower() or "advanced" in c.lower()]
        posture_detail = (
            improving[0] if improving else "Institutional confidence increased over time."
        )
    elif confidence_direction == "deteriorating":
        posture_trend = "Governance posture degraded."
        weakening = [c for c in changes if "weakened" in c.lower()]
        posture_detail = (
            weakening[0]
            if weakening
            else "Institutional confidence weakened across governance stages."
        )
    elif confidence_direction == "mixed":
        posture_trend = "Governance posture is mixed."
        posture_detail = "Some stages improved while others weakened or remained stable."
    else:
        posture_trend = "Governance posture is stable."
        posture_detail = "No material confidence movement detected across recorded history."

    dossier = _gcc_get(dossier_record, "human_escalation_dossier") or {}
    explanation: List[str] = []
    seen_expl: set = set()

    def _expl_add(text: str) -> None:
        t = text.strip()
        if t and t not in seen_expl:
            seen_expl.add(t)
            explanation.append(t)

    if confidence_direction == "dormant":
        _expl_add(
            "Governance remains dormant because institutional readiness, constitutional eligibility, "
            "and recommendation thresholds have not materially advanced."
        )
    elif confidence_direction == "improving":
        _expl_add("Governance confidence is trending upward across recorded institutional history.")
    elif confidence_direction == "deteriorating":
        _expl_add("Governance confidence has weakened across recorded institutional history.")

    esc_cls = str(_gcc_get(dossier_summary, "human_escalation_classification") or "").upper()
    if esc_cls == "NO_ESCALATION":
        _expl_add(
            "Human escalation remains inactive because governance confidence does not justify "
            "operator intervention."
        )
    elif esc_cls in ("OBSERVE_ONLY_ESCALATION", "LIMITED_ESCALATION", "FULL_OPERATOR_ESCALATION"):
        _expl_add(f"Human escalation posture is active ({esc_cls.replace('_', ' ').title()}).")

    exec_sum = str(_gcc_get(dossier, "executive_summary") or "").strip()
    if exec_sum:
        _expl_add(exec_sum)

    for item in (_gcc_get(dossier, "case_against_runtime") or [])[:2]:
        _expl_add(str(item))

    if not explanation:
        _expl_add(
            "Governance posture reflects current institutional summaries; historical depth remains limited."
        )
    explanation = explanation[:5]

    if not changes:
        changes = ["No material governance changes detected."]

    transitions.sort(key=lambda r: r.get("Timestamp", ""))

    return {
        "posture_trend": posture_trend,
        "posture_detail": posture_detail,
        "confidence_direction": confidence_direction,
        "institutional_momentum": momentum,
        "posture_stability": posture_stability,
        "transitions": transitions,
        "changes": changes,
        "operator_explanation": explanation,
        "has_history": loaded_stages > 0,
        "has_transitions": len(transitions) > 0,
    }


def _gcc_render_hist_card(title: str, value: str, detail: str = "") -> None:
    detail_html = f'<div class="gcc-hist-detail">{detail}</div>' if detail else ""
    st.markdown(
        f'<div class="gcc-hist-wrap"><div class="gcc-hist-title">{title}</div>'
        f'<div class="gcc-hist-value">{value}</div>{detail_html}</div>',
        unsafe_allow_html=True,
    )


def _gcc_render_historical_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Optional[Dict[str, Any]] = None,
) -> None:
    if hist is None:
        hist = _gcc_analyze_governance_history(
            readiness=readiness,
            admission=admission,
            eligibility=eligibility,
            recommendation=recommendation,
            review=review,
            verdict=verdict,
            dossier_summary=dossier_summary,
            dossier_record=dossier_record,
        )

    st.markdown("### Governance Historical Intelligence")
    st.caption("Institutional audit trail — how governance posture evolved over recorded history.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _gcc_render_hist_card(
            "Governance Posture Trend", hist["posture_trend"], hist["posture_detail"]
        )
    with c2:
        _gcc_render_hist_card(
            "Confidence Direction",
            hist["confidence_direction"].capitalize(),
            f"Confidence direction: {hist['confidence_direction'].capitalize()}",
        )
    with c3:
        _gcc_render_hist_card(
            "Institutional Momentum",
            hist["institutional_momentum"],
            f"Institutional momentum: {hist['institutional_momentum']}",
        )
    with c4:
        _gcc_render_hist_card("Posture Stability", hist["posture_stability"])

    st.markdown("#### Governance Transition History")
    if hist["has_transitions"]:
        trans_df = pd.DataFrame(hist["transitions"])
        _ei_render_table(trans_df, height=min(280, 40 + 35 * len(trans_df)))
    else:
        st.info(
            "No governance posture transitions detected yet. "
            "History will populate as governance maturity evolves."
        )

    st.markdown("#### Governance Change Detection")
    for change in hist["changes"][:10]:
        st.markdown(f'<div class="gcc-hist-metric">• {change}</div>', unsafe_allow_html=True)

    st.markdown("#### Why Governance Looks This Way")
    for item in hist["operator_explanation"]:
        st.markdown(f'<div class="gcc-block-item">• {item}</div>', unsafe_allow_html=True)


_GCC_REGIME_PRIORITY: Tuple[str, ...] = (
    "CONSTITUTIONAL_STRESS",
    "GOVERNANCE_REGRESSION",
    "INSTITUTIONAL_INSTABILITY",
    "RUNTIME_CANDIDATE",
    "PRE_RUNTIME_READINESS",
    "GOVERNANCE_ACCELERATION",
    "EARLY_INSTITUTIONAL_FORMATION",
    "DORMANT_GOVERNANCE",
)

_GCC_REGIME_DISPLAY: Dict[str, str] = {
    "DORMANT_GOVERNANCE": "Dormant Governance",
    "EARLY_INSTITUTIONAL_FORMATION": "Early Institutional Formation",
    "GOVERNANCE_ACCELERATION": "Governance Acceleration",
    "CONSTITUTIONAL_STRESS": "Constitutional Stress",
    "INSTITUTIONAL_INSTABILITY": "Institutional Instability",
    "PRE_RUNTIME_READINESS": "Pre-Runtime Readiness",
    "RUNTIME_CANDIDATE": "Runtime Candidate",
    "GOVERNANCE_REGRESSION": "Governance Regression",
}

_GCC_REGIME_OPERATOR_MEANING: Dict[str, str] = {
    "DORMANT_GOVERNANCE": (
        "Triton's runtime governance stack is present but inactive. The system is observing itself, "
        "but no institutional pathway toward runtime governance has been earned."
    ),
    "EARLY_INSTITUTIONAL_FORMATION": (
        "Governance signals are beginning to accumulate, but institutional thresholds for readiness, "
        "admission, and review have not yet been met."
    ),
    "GOVERNANCE_ACCELERATION": (
        "Governance confidence is improving across recorded history. Institutional momentum is building, "
        "but runtime mutation remains locked pending further maturity."
    ),
    "CONSTITUTIONAL_STRESS": (
        "Constitutional constraints dominate Triton's governance posture. Runtime governance should "
        "remain blocked until constitutional pressure falls."
    ),
    "INSTITUTIONAL_INSTABILITY": (
        "Governance signals are mixed or shifting frequently. Operators should treat posture changes "
        "as provisional until stability returns."
    ),
    "PRE_RUNTIME_READINESS": (
        "Several upstream governance stages show limited or advancing maturity. Runtime discussion may "
        "be approaching, but live runtime policy remains locked."
    ),
    "RUNTIME_CANDIDATE": (
        "Triton has accumulated enough institutional support to be discussed as a future runtime "
        "candidate, but runtime mutation remains locked."
    ),
    "GOVERNANCE_REGRESSION": (
        "Governance confidence or stage maturity is moving backward. Operators should investigate "
        "whether constitutional or institutional deterioration is underway."
    ),
}

_GCC_REGIME_OPERATOR_POSTURE: Dict[str, str] = {
    "DORMANT_GOVERNANCE": "OBSERVE_ONLY",
    "EARLY_INSTITUTIONAL_FORMATION": "CONTINUE_EVIDENCE_COLLECTION",
    "GOVERNANCE_ACCELERATION": "REVIEW_WITH_CAUTION",
    "CONSTITUTIONAL_STRESS": "MAINTAIN_CONSTITUTIONAL_LOCK",
    "INSTITUTIONAL_INSTABILITY": "REVIEW_WITH_CAUTION",
    "PRE_RUNTIME_READINESS": "PREPARE_OPERATOR_REVIEW",
    "RUNTIME_CANDIDATE": "PREPARE_OPERATOR_REVIEW",
    "GOVERNANCE_REGRESSION": "INVESTIGATE_REGRESSION",
}


def _gcc_cls_maturity_rank(classification: Any) -> int:
    c = str(classification or "").upper()
    if not c or c == "—":
        return -1
    if "NOT_" in c or "DO_NOT" in c:
        return 0
    if "FAVORABLE" in c or "FULL_OPERATOR" in c:
        return 4
    if "LIMITED" in c:
        return 2
    if "OBSERVE" in c:
        return 1
    if any(k in c for k in ("READY", "ADMITTED", "ELIGIBLE", "RECOMMENDED", "UNDER_REVIEW")):
        return 3
    return 0


def _gcc_collect_governance_snapshot(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
) -> Dict[str, Any]:
    dossier = _gcc_get(dossier_record, "human_escalation_dossier") or {}
    confidences = [
        _gcc_get(readiness, "readiness_confidence", 0.0),
        _gcc_get(admission, "admission_confidence", 0.0),
        _gcc_get(eligibility, "constitutional_eligibility_confidence", 0.0),
        _gcc_get(recommendation, "recommendation_confidence", 0.0),
        _gcc_get(review, "review_confidence", 0.0),
        _gcc_get(verdict, "verdict_confidence", 0.0),
        _gcc_get(dossier_summary, "escalation_confidence", 0.0),
    ]
    try:
        max_conf = max(float(c or 0.0) for c in confidences)
    except Exception:
        max_conf = 0.0

    states = [
        _gcc_get(readiness, "readiness_state"),
        _gcc_get(admission, "admission_state"),
        _gcc_get(eligibility, "constitutional_eligibility_state"),
        _gcc_get(recommendation, "recommendation_state"),
        _gcc_get(review, "review_state"),
        _gcc_get(verdict, "verdict_state"),
        _gcc_get(dossier_summary, "dossier_state"),
    ]
    state_text = [str(s or "").upper() for s in states if s]
    all_dormant = bool(state_text) and all("DORMANT" in s for s in state_text)

    const_review = any(
        _gcc_get(obj, "constitutional_review_required") is True
        for obj in (
            readiness,
            admission,
            eligibility,
            recommendation,
            review,
            verdict,
            dossier_summary,
        )
    )
    constitutional_safe = _gcc_get(dossier, "constitutional_safe")
    future_candidate = _gcc_get(
        dossier, "future_runtime_candidate", _gcc_get(dossier_summary, "future_runtime_candidate")
    )

    classifications = {
        "readiness": _gcc_get(readiness, "runtime_readiness_classification"),
        "admission": _gcc_get(admission, "runtime_admission_classification"),
        "eligibility": _gcc_get(eligibility, "runtime_constitutional_eligibility_classification"),
        "recommendation": _gcc_get(
            recommendation, "runtime_enablement_recommendation_classification"
        ),
        "review": _gcc_get(review, "runtime_enablement_review_classification"),
        "verdict": _gcc_get(verdict, "runtime_verdict_classification"),
        "escalation": _gcc_get(dossier_summary, "human_escalation_classification"),
    }

    upstream_ranks = [
        _gcc_cls_maturity_rank(classifications["readiness"]),
        _gcc_cls_maturity_rank(classifications["admission"]),
        _gcc_cls_maturity_rank(classifications["eligibility"]),
    ]
    max_upstream_rank = max(upstream_ranks) if upstream_ranks else 0

    return {
        "confidences": confidences,
        "max_conf": max_conf,
        "all_dormant": all_dormant,
        "constitutional_safe": constitutional_safe,
        "constitutional_review_required": const_review,
        "future_runtime_candidate": future_candidate is True,
        "classifications": classifications,
        "max_upstream_rank": max_upstream_rank,
        "dossier": dossier,
    }


def _gcc_regime_drivers(regime: str, snap: Dict[str, Any], hist: Dict[str, Any]) -> List[str]:
    cls = snap["classifications"]
    drivers: List[str] = []

    if regime == "DORMANT_GOVERNANCE":
        drivers.extend(
            [
                "All runtime governance stages remain dormant",
                "Runtime admission has not been earned",
                "Constitutional eligibility is not active",
                "Human escalation is inactive",
            ]
        )
    elif regime == "EARLY_INSTITUTIONAL_FORMATION":
        drivers.append(f"Governance confidence above zero (max {_gcc_fmt_conf(snap['max_conf'])})")
        drivers.append(f"Readiness posture: {cls.get('readiness') or '—'}")
        drivers.append("Runtime review has not yet been earned")
        drivers.append("Institutional signals are accumulating")
    elif regime == "GOVERNANCE_ACCELERATION":
        drivers.append(f"Confidence trend is {hist.get('confidence_direction', '—')}")
        drivers.append(f"Institutional momentum is {hist.get('institutional_momentum', '—')}")
        if hist.get("has_transitions"):
            drivers.append("Stage progression detected in governance history")
        else:
            drivers.append("Confidence movement detected across governance memory")
    elif regime == "CONSTITUTIONAL_STRESS":
        if snap["constitutional_safe"] is False:
            drivers.append("Constitutional safety is false")
        if snap["constitutional_review_required"]:
            drivers.append("Constitutional review remains required")
        drivers.append(f"Eligibility posture: {cls.get('eligibility') or '—'}")
        drivers.append("Court/council constraints dominate governance posture")
    elif regime == "INSTITUTIONAL_INSTABILITY":
        drivers.append(f"Confidence direction is {hist.get('confidence_direction', '—')}")
        drivers.append(hist.get("posture_stability", "Governance posture shifts detected"))
        drivers.append("Conflicting or unstable stage outcomes present")
    elif regime == "PRE_RUNTIME_READINESS":
        drivers.append(f"Readiness posture: {cls.get('readiness') or '—'}")
        drivers.append(f"Admission posture: {cls.get('admission') or '—'}")
        drivers.append(f"Eligibility posture: {cls.get('eligibility') or '—'}")
        drivers.append("Runtime mutation remains locked")
    elif regime == "RUNTIME_CANDIDATE":
        if snap["future_runtime_candidate"]:
            drivers.append("Future runtime candidate flag is true")
        drivers.append(f"Verdict posture: {cls.get('verdict') or '—'}")
        drivers.append(f"Human escalation posture: {cls.get('escalation') or '—'}")
        drivers.append("Runtime mutation remains locked")
    elif regime == "GOVERNANCE_REGRESSION":
        drivers.append(f"Confidence direction is {hist.get('confidence_direction', '—')}")
        weakened = [c for c in hist.get("changes", []) if "weakened" in c.lower()]
        if weakened:
            drivers.append(weakened[0])
        else:
            drivers.append("Governance confidence is deteriorating across recorded history")
        if snap["constitutional_safe"] is False:
            drivers.append("Constitutional safety remains weak")

    return drivers[:6]


def _gcc_regime_confidence(regime: str, snap: Dict[str, Any], hist: Dict[str, Any]) -> float:
    if regime == "DORMANT_GOVERNANCE" and snap["max_conf"] <= 0.01 and snap["all_dormant"]:
        return 0.95
    if regime == "CONSTITUTIONAL_STRESS":
        score = 0.70
        if snap["constitutional_safe"] is False:
            score += 0.12
        if snap["constitutional_review_required"]:
            score += 0.10
        return min(score, 0.98)
    if regime == "GOVERNANCE_REGRESSION":
        return 0.82 if hist.get("confidence_direction") == "deteriorating" else 0.68
    if regime == "INSTITUTIONAL_INSTABILITY":
        return 0.72 if hist.get("confidence_direction") == "mixed" else 0.60
    if regime == "RUNTIME_CANDIDATE":
        score = 0.65
        if snap["future_runtime_candidate"]:
            score += 0.15
        if str(snap["classifications"].get("verdict") or "").upper() == "FAVORABLE_RUNTIME_VERDICT":
            score += 0.12
        return min(score, 0.92)
    if regime == "PRE_RUNTIME_READINESS":
        return min(0.55 + 0.08 * snap["max_upstream_rank"], 0.88)
    if regime == "GOVERNANCE_ACCELERATION":
        mom = hist.get("institutional_momentum", "NONE")
        base = {"LOW": 0.62, "MODERATE": 0.74, "HIGH": 0.84}.get(mom, 0.58)
        return base
    if regime == "EARLY_INSTITUTIONAL_FORMATION":
        return min(0.50 + snap["max_conf"], 0.80)
    return 0.55


def _gcc_detect_governance_regime(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    cls = snap["classifications"]
    verdict_cls = str(cls.get("verdict") or "").upper()
    esc_cls = str(cls.get("escalation") or "").upper()

    def _match_constitutional_stress() -> bool:
        if snap["constitutional_safe"] is False:
            return True
        if snap["constitutional_review_required"]:
            return True
        if (
            str(cls.get("eligibility") or "").upper() == "NOT_CONSTITUTIONALLY_ELIGIBLE"
            and snap["constitutional_review_required"]
        ):
            return True
        return False

    def _match_regression() -> bool:
        if hist.get("confidence_direction") == "deteriorating":
            return True
        for tr in hist.get("transitions", []):
            if "weakened" in str(tr.get("Interpretation", "")).lower():
                return True
        for ch in hist.get("changes", []):
            if "weakened" in str(ch).lower():
                return True
        return False

    def _match_instability() -> bool:
        if hist.get("confidence_direction") == "mixed":
            return True
        if "Frequent" in str(hist.get("posture_stability", "")):
            return True
        return False

    def _match_runtime_candidate() -> bool:
        if snap["future_runtime_candidate"]:
            return True
        if verdict_cls == "FAVORABLE_RUNTIME_VERDICT":
            return True
        if esc_cls == "FULL_OPERATOR_ESCALATION":
            return True
        return False

    def _match_pre_runtime_readiness() -> bool:
        if snap["max_upstream_rank"] >= 2:
            return True
        if any(
            "LIMITED" in str(v or "").upper() and "NOT_" not in str(v or "").upper()
            for v in (cls.get("readiness"), cls.get("admission"), cls.get("eligibility"))
        ):
            return True
        return False

    def _match_acceleration() -> bool:
        return hist.get("confidence_direction") == "improving" and hist.get(
            "institutional_momentum"
        ) in ("LOW", "MODERATE", "HIGH")

    def _match_early_formation() -> bool:
        return (
            snap["max_conf"] > 0.01
            and str(cls.get("readiness") or "").upper() == "NOT_RUNTIME_READY"
        )

    def _match_dormant() -> bool:
        return snap["max_conf"] <= 0.01 and snap["all_dormant"]

    matchers = {
        "CONSTITUTIONAL_STRESS": _match_constitutional_stress,
        "GOVERNANCE_REGRESSION": _match_regression,
        "INSTITUTIONAL_INSTABILITY": _match_instability,
        "RUNTIME_CANDIDATE": _match_runtime_candidate,
        "PRE_RUNTIME_READINESS": _match_pre_runtime_readiness,
        "GOVERNANCE_ACCELERATION": _match_acceleration,
        "EARLY_INSTITUTIONAL_FORMATION": _match_early_formation,
        "DORMANT_GOVERNANCE": _match_dormant,
    }

    regime = "DORMANT_GOVERNANCE"
    for candidate in _GCC_REGIME_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            regime = candidate
            break

    confidence = _gcc_regime_confidence(regime, snap, hist)
    drivers = _gcc_regime_drivers(regime, snap, hist)
    if not drivers:
        drivers = ["Governance regime derived from current institutional summaries."]

    return {
        "regime": regime,
        "regime_display": _GCC_REGIME_DISPLAY.get(regime, regime.replace("_", " ").title()),
        "regime_confidence": confidence,
        "drivers": drivers,
        "operator_meaning": _GCC_REGIME_OPERATOR_MEANING.get(regime, ""),
        "operator_posture": _GCC_REGIME_OPERATOR_POSTURE.get(regime, "NO_ACTION_REQUIRED"),
    }


def _gcc_render_regime_detection(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if regime is None:
        regime = _gcc_detect_governance_regime(
            readiness=readiness,
            admission=admission,
            eligibility=eligibility,
            recommendation=recommendation,
            review=review,
            verdict=verdict,
            dossier_summary=dossier_summary,
            dossier_record=dossier_record,
            hist=hist,
        )

    st.markdown("### Governance Regime Detection")
    st.caption("Institutional classification of Triton's current governance environment.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Governance Regime", regime["regime_display"])
    c2.metric("Regime Confidence", _gcc_fmt_conf(regime["regime_confidence"]))
    c3.metric("Operator Posture", regime["operator_posture"])

    st.markdown("**Regime Drivers**")
    for driver in regime["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    meaning = regime["operator_meaning"]
    if regime["regime"] in (
        "CONSTITUTIONAL_STRESS",
        "GOVERNANCE_REGRESSION",
        "INSTITUTIONAL_INSTABILITY",
    ):
        st.warning(meaning)
    elif regime["regime"] in (
        "RUNTIME_CANDIDATE",
        "PRE_RUNTIME_READINESS",
        "GOVERNANCE_ACCELERATION",
    ):
        st.success(meaning)
    else:
        st.info(meaning)

    with st.expander("Regime classification detail", expanded=False):
        st.markdown(f"- **Internal label:** `{regime['regime']}`")
        st.markdown(f"- **Confidence direction:** `{hist.get('confidence_direction', '—')}`")
        st.markdown(f"- **Institutional momentum:** `{hist.get('institutional_momentum', '—')}`")
        st.markdown(f"- **Posture stability:** {hist.get('posture_stability', '—')}")

    return regime


_GCC_TRAJECTORY_PRIORITY: Tuple[str, ...] = (
    "CONSTITUTIONALLY_CONSTRAINED",
    "GOVERNANCE_REGRESSION_RISK",
    "RUNTIME_DISCUSSION_CANDIDATE",
    "PRE_RUNTIME_TRAJECTORY",
    "GOVERNANCE_ACCELERATING",
    "GOVERNANCE_IMPROVING",
    "GOVERNANCE_STABLE",
    "GOVERNANCE_DORMANT",
)

_GCC_TRAJECTORY_DISPLAY: Dict[str, str] = {
    "GOVERNANCE_DORMANT": "Governance Dormant",
    "GOVERNANCE_STABLE": "Governance Stable",
    "GOVERNANCE_IMPROVING": "Governance Improving",
    "GOVERNANCE_ACCELERATING": "Governance Accelerating",
    "GOVERNANCE_REGRESSION_RISK": "Governance Regression Risk",
    "CONSTITUTIONALLY_CONSTRAINED": "Constitutionally Constrained",
    "PRE_RUNTIME_TRAJECTORY": "Pre-Runtime Trajectory",
    "RUNTIME_DISCUSSION_CANDIDATE": "Runtime Discussion Candidate",
}

_GCC_FORECAST_OPERATOR_ACTION: Dict[str, str] = {
    "GOVERNANCE_DORMANT": "CONTINUE_OBSERVATION",
    "GOVERNANCE_STABLE": "CONTINUE_OBSERVATION",
    "GOVERNANCE_IMPROVING": "CONTINUE_EVIDENCE_COLLECTION",
    "GOVERNANCE_ACCELERATING": "MONITOR_ACCELERATION",
    "GOVERNANCE_REGRESSION_RISK": "REVIEW_GOVERNANCE_SHIFTS",
    "CONSTITUTIONALLY_CONSTRAINED": "MAINTAIN_CONSTITUTIONAL_LOCK",
    "PRE_RUNTIME_TRAJECTORY": "PREPARE_RUNTIME_DISCUSSION",
    "RUNTIME_DISCUSSION_CANDIDATE": "PREPARE_RUNTIME_DISCUSSION",
}


def _gcc_forecast_regression_risk(hist: Dict[str, Any], regime: Dict[str, Any]) -> str:
    direction = hist.get("confidence_direction", "stable")
    regime_key = regime.get("regime", "")
    stability = str(hist.get("posture_stability", ""))
    if (
        regime_key in ("GOVERNANCE_REGRESSION", "INSTITUTIONAL_INSTABILITY")
        and direction == "deteriorating"
    ):
        return "HIGH"
    if direction == "deteriorating" or "Frequent" in stability:
        return "MODERATE"
    if direction == "mixed" or regime_key == "INSTITUTIONAL_INSTABILITY":
        return "LOW"
    return "NONE"


def _gcc_forecast_runtime_probability(
    trajectory: str, regime: Dict[str, Any], snap: Dict[str, Any]
) -> str:
    if trajectory == "RUNTIME_DISCUSSION_CANDIDATE":
        return "HIGH"
    if trajectory == "PRE_RUNTIME_TRAJECTORY":
        return "ELEVATED"
    if trajectory in ("GOVERNANCE_ACCELERATING", "GOVERNANCE_IMPROVING"):
        return "MODERATE"
    if regime.get("regime") == "EARLY_INSTITUTIONAL_FORMATION":
        return "LOW"
    return "VERY_LOW"


def _gcc_forecast_constitutional_outlook(
    snap: Dict[str, Any],
    hist: Dict[str, Any],
    trajectory: str,
) -> str:
    if snap["constitutional_safe"] is True and not snap["constitutional_review_required"]:
        return "Constitutional outlook favorable"
    if trajectory == "CONSTITUTIONALLY_CONSTRAINED":
        if hist.get("confidence_direction") == "improving":
            return "Constitutional constraints improving"
        return "Constitutional pressure likely persistent"
    if snap["constitutional_safe"] is False and hist.get("confidence_direction") == "deteriorating":
        return "Constitutional risk elevated"
    if snap["constitutional_review_required"] and hist.get("confidence_direction") == "stable":
        return "Constitutional pressure stable"
    if snap["constitutional_safe"] is False:
        return "Constitutional pressure likely persistent"
    return "Constitutional pressure stable"


def _gcc_forecast_confidence(
    trajectory: str, regime: Dict[str, Any], hist: Dict[str, Any]
) -> float:
    base = float(regime.get("regime_confidence", 0.55))
    direction = hist.get("confidence_direction", "stable")
    if trajectory in ("GOVERNANCE_DORMANT", "GOVERNANCE_STABLE") and direction in (
        "dormant",
        "stable",
    ):
        return min(max(base, 0.85), 0.95)
    if direction == "mixed" or "Frequent" in str(hist.get("posture_stability", "")):
        return min(base, 0.65)
    if not hist.get("has_history"):
        return min(base, 0.55)
    if not hist.get("has_transitions") and direction == "dormant":
        return min(max(base, 0.88), 0.95)
    return min(max(base, 0.50), 0.90)


def _gcc_forecast_narrative(
    trajectory: str,
    snap: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    constitutional_outlook: str,
    runtime_probability: str,
) -> List[str]:
    bullets: List[str] = []
    cls = snap["classifications"]

    if trajectory == "GOVERNANCE_DORMANT":
        bullets.append(
            "Triton governance is expected to remain dormant in the near term. Institutional readiness, "
            "constitutional eligibility, and runtime recommendation thresholds remain materially inactive."
        )
    elif trajectory == "GOVERNANCE_STABLE":
        bullets.append(
            "Governance posture appears stable. Confidence and stage maturity show no material directional shift."
        )
    elif trajectory == "GOVERNANCE_IMPROVING":
        bullets.append(
            "Governance posture is gradually improving. Confidence trends suggest slow institutional progression."
        )
    elif trajectory == "GOVERNANCE_ACCELERATING":
        bullets.append(
            "Governance momentum is accelerating across recorded history with upward confidence movement."
        )
    elif trajectory == "GOVERNANCE_REGRESSION_RISK":
        bullets.append(
            "Governance regression risk is elevated due to weakening confidence or unstable posture shifts."
        )
    elif trajectory == "CONSTITUTIONALLY_CONSTRAINED":
        bullets.append(
            "Constitutional constraints are expected to dominate near-term governance trajectory."
        )
    elif trajectory == "PRE_RUNTIME_TRAJECTORY":
        bullets.append(
            "Upstream governance stages show limited or advancing maturity; runtime discussion may approach."
        )
    elif trajectory == "RUNTIME_DISCUSSION_CANDIDATE":
        bullets.append(
            "Institutional support is sufficient to sustain runtime governance discussion, though enablement remains locked."
        )

    if snap["constitutional_safe"] is False or snap["constitutional_review_required"]:
        bullets.append(constitutional_outlook + ".")

    if runtime_probability in ("VERY_LOW", "LOW"):
        bullets.append(
            "Runtime discussion probability remains low; institutional readiness unlikely near term."
        )
    elif runtime_probability == "MODERATE":
        bullets.append(
            "Runtime discussion probability is rising modestly as governance signals accumulate."
        )
    elif runtime_probability in ("ELEVATED", "HIGH"):
        bullets.append(
            f"Runtime discussion probability is {runtime_probability.lower().replace('_', ' ')} — not runtime enablement."
        )

    if str(cls.get("escalation") or "").upper() == "NO_ESCALATION":
        bullets.append("Human escalation remains inactive under current institutional confidence.")

    return bullets[:5]


def _gcc_detect_governance_forecast(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    cls = snap["classifications"]
    regime_key = regime.get("regime", "")
    direction = hist.get("confidence_direction", "stable")
    momentum = hist.get("institutional_momentum", "NONE")
    verdict_cls = str(cls.get("verdict") or "").upper()
    esc_cls = str(cls.get("escalation") or "").upper()

    improving_count = sum(
        1
        for tr in hist.get("transitions", [])
        if "increased" in str(tr.get("Interpretation", "")).lower()
        or "strengthened" in str(tr.get("Interpretation", "")).lower()
    )

    def _match_constitutionally_constrained() -> bool:
        return (
            snap["constitutional_safe"] is False
            or snap["constitutional_review_required"]
            or regime_key == "CONSTITUTIONAL_STRESS"
        )

    def _match_regression_risk() -> bool:
        return (
            direction == "deteriorating"
            or regime_key in ("GOVERNANCE_REGRESSION", "INSTITUTIONAL_INSTABILITY")
            or any("weakened" in str(c).lower() for c in hist.get("changes", []))
        )

    def _match_runtime_discussion() -> bool:
        return (
            regime_key == "RUNTIME_CANDIDATE"
            or snap["future_runtime_candidate"]
            or verdict_cls == "FAVORABLE_RUNTIME_VERDICT"
            or esc_cls == "FULL_OPERATOR_ESCALATION"
        )

    def _match_pre_runtime() -> bool:
        return regime_key == "PRE_RUNTIME_READINESS" or snap["max_upstream_rank"] >= 2

    def _match_accelerating() -> bool:
        return direction == "improving" and (momentum == "HIGH" or improving_count >= 2)

    def _match_improving() -> bool:
        return direction == "improving" and momentum in ("LOW", "MODERATE")

    def _match_stable() -> bool:
        return direction in ("stable", "dormant") and "Frequent" not in str(
            hist.get("posture_stability", "")
        )

    def _match_dormant() -> bool:
        return snap["max_conf"] <= 0.01 and snap["all_dormant"] and esc_cls == "NO_ESCALATION"

    matchers = {
        "CONSTITUTIONALLY_CONSTRAINED": _match_constitutionally_constrained,
        "GOVERNANCE_REGRESSION_RISK": _match_regression_risk,
        "RUNTIME_DISCUSSION_CANDIDATE": _match_runtime_discussion,
        "PRE_RUNTIME_TRAJECTORY": _match_pre_runtime,
        "GOVERNANCE_ACCELERATING": _match_accelerating,
        "GOVERNANCE_IMPROVING": _match_improving,
        "GOVERNANCE_STABLE": _match_stable,
        "GOVERNANCE_DORMANT": _match_dormant,
    }

    trajectory = "GOVERNANCE_DORMANT"
    for candidate in _GCC_TRAJECTORY_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            trajectory = candidate
            break

    regression_risk = _gcc_forecast_regression_risk(hist, regime)
    runtime_probability = _gcc_forecast_runtime_probability(trajectory, regime, snap)
    constitutional_outlook = _gcc_forecast_constitutional_outlook(snap, hist, trajectory)
    forecast_confidence = _gcc_forecast_confidence(trajectory, regime, hist)
    narrative = _gcc_forecast_narrative(
        trajectory, snap, hist, regime, constitutional_outlook, runtime_probability
    )
    operator_action = _GCC_FORECAST_OPERATOR_ACTION.get(trajectory, "CONTINUE_OBSERVATION")

    return {
        "trajectory": trajectory,
        "trajectory_display": _GCC_TRAJECTORY_DISPLAY.get(
            trajectory, trajectory.replace("_", " ").title()
        ),
        "forecast_confidence": forecast_confidence,
        "runtime_discussion_probability": runtime_probability,
        "constitutional_outlook": constitutional_outlook,
        "regression_risk": regression_risk,
        "narrative": narrative,
        "operator_action": operator_action,
    }


def _gcc_render_forecast_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
) -> Dict[str, Any]:
    forecast = _gcc_detect_governance_forecast(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
    )

    st.markdown("### Governance Forecasting & Trajectory Intelligence")
    st.caption(
        "Read-only institutional forecast derived from governance history, regime, and posture."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Trajectory", forecast["trajectory_display"])
    c2.metric("Forecast Confidence", _gcc_fmt_conf(forecast["forecast_confidence"]))
    c3.metric("Runtime Discussion Probability", forecast["runtime_discussion_probability"])
    c4.metric("Regression Risk", forecast["regression_risk"])

    st.warning(
        "**This is NOT runtime enablement.** Runtime mutation remains locked. "
        "Forecast probability reflects institutional discussion posture only."
    )

    outlook = forecast["constitutional_outlook"]
    if "persistent" in outlook.lower() or "elevated" in outlook.lower():
        st.warning(f"**Constitutional Outlook:** {outlook}")
    elif "favorable" in outlook.lower() or "improving" in outlook.lower():
        st.success(f"**Constitutional Outlook:** {outlook}")
    else:
        st.info(f"**Constitutional Outlook:** {outlook}")

    st.markdown("**Institutional Forecast Narrative**")
    for item in forecast["narrative"]:
        st.markdown(f'<div class="gcc-block-item">• {item}</div>', unsafe_allow_html=True)

    st.metric("Recommended Operator Action", forecast["operator_action"])

    with st.expander("Forecast derivation detail", expanded=False):
        st.markdown(f"- **Internal trajectory:** `{forecast['trajectory']}`")
        st.markdown(f"- **Regime:** `{regime.get('regime', '—')}`")
        st.markdown(f"- **Confidence direction:** `{hist.get('confidence_direction', '—')}`")
        st.markdown(f"- **Institutional momentum:** `{hist.get('institutional_momentum', '—')}`")

    return forecast


_GCC_TENSION_OPERATOR_RESPONSE: Dict[str, str] = {
    "NO_TENSION": "CONTINUE_OBSERVATION",
    "LOW_TENSION": "CONTINUE_OBSERVATION",
    "MODERATE_TENSION": "REVIEW_GOVERNANCE_SIGNALS",
    "HIGH_TENSION": "MAINTAIN_CONSTITUTIONAL_LOCK",
    "CRITICAL_TENSION": "BLOCK_RUNTIME_DISCUSSION",
}

_GCC_TENSION_INTERPRETATION: Dict[str, str] = {
    "NO_TENSION": (
        "Governance signals are internally consistent. Dormant posture, low confidence, blocked runtime, "
        "and inactive escalation all agree."
    ),
    "LOW_TENSION": (
        "Minor governance signal disagreement detected. Early momentum or confidence movement has not "
        "yet changed institutional lock posture."
    ),
    "MODERATE_TENSION": (
        "Governance shows early improvement, but constitutional constraints still dominate. "
        "Operators should avoid treating momentum as readiness."
    ),
    "HIGH_TENSION": (
        "Strong governance contradictions are present. Favorable or improving signals conflict with "
        "restrictive constitutional or verdict state; review carefully."
    ),
    "CRITICAL_TENSION": (
        "Governance contains a safety-critical contradiction. Runtime mutation lock or constitutional "
        "safety must be reviewed before any further autonomy discussion."
    ),
}


def _gcc_is_optimistic_classification(classification: Any) -> bool:
    c = str(classification or "").upper()
    if not c or "NOT_" in c or "DO_NOT" in c:
        return False
    return any(k in c for k in ("LIMITED", "RECOMMENDED", "UNDER_REVIEW", "FAVORABLE", "FULL"))


def _gcc_detect_governance_tensions(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    dossier = snap["dossier"]
    cls = snap["classifications"]
    contradictions: List[Dict[str, str]] = []

    def _add(severity: str, message: str, rule: str) -> None:
        contradictions.append({"severity": severity, "message": message, "rule": rule})

    trajectory = forecast.get("trajectory", "")
    regime_key = regime.get("regime", "")
    improving_trajectories = {
        "GOVERNANCE_IMPROVING",
        "GOVERNANCE_ACCELERATING",
        "PRE_RUNTIME_TRAJECTORY",
        "RUNTIME_DISCUSSION_CANDIDATE",
    }

    mutation_summaries = (
        ("readiness", readiness),
        ("admission", admission),
        ("eligibility", eligibility),
        ("recommendation", recommendation),
        ("review", review),
        ("verdict", verdict),
        ("dossier_summary", dossier_summary),
    )
    for _label, obj in mutation_summaries:
        if not isinstance(obj, dict) or not obj:
            continue
        val = _gcc_get(obj, "runtime_mutation_allowed")
        if val is True:
            _add(
                "critical",
                "Runtime mutation lock is not clearly preserved.",
                "runtime_mutation_true",
            )
            break
        if "engine" in obj and "runtime_mutation_allowed" not in obj:
            _add(
                "critical",
                "Runtime mutation lock is not clearly preserved.",
                "runtime_mutation_ambiguous",
            )
            break

    dossier_mutation = _gcc_get(dossier, "runtime_mutation_allowed")
    if dossier_mutation is True and not any(
        c["rule"] == "runtime_mutation_true" for c in contradictions
    ):
        _add(
            "critical",
            "Runtime mutation lock is not clearly preserved.",
            "runtime_mutation_true",
        )

    if snap["future_runtime_candidate"] and snap["constitutional_safe"] is False:
        _add(
            "critical",
            "Future runtime candidacy conflicts with constitutional safety.",
            "runtime_candidate_vs_constitutional_unsafe",
        )

    if trajectory in improving_trajectories and (
        regime_key == "CONSTITUTIONAL_STRESS" or snap["constitutional_safe"] is False
    ):
        _add(
            "warning",
            "Governance trajectory is improving, but constitutional constraints remain dominant.",
            "improving_trajectory_vs_constitutional_stress",
        )

    rec_opt = _gcc_is_optimistic_classification(cls.get("recommendation"))
    review_opt = _gcc_is_optimistic_classification(cls.get("review"))
    verdict_cls = str(cls.get("verdict") or "").upper()
    if (rec_opt or review_opt) and verdict_cls == "DO_NOT_ENABLE_RUNTIME":
        _add(
            "warning",
            "Recommendation/review posture is more optimistic than institutional verdict.",
            "recommendation_optimism_vs_verdict",
        )

    readiness_cls = str(cls.get("readiness") or "").upper()
    readiness_conf = float(_gcc_get(readiness, "readiness_confidence", 0.0) or 0.0)
    regime_conf = float(regime.get("regime_confidence", 0.0) or 0.0)
    forecast_conf = float(forecast.get("forecast_confidence", 0.0) or 0.0)
    if (
        max(regime_conf, forecast_conf) >= 0.75
        and readiness_cls == "NOT_RUNTIME_READY"
        and readiness_conf <= 0.05
    ):
        _add(
            "warning",
            "Governance classification confidence is high while runtime readiness remains dormant.",
            "high_confidence_vs_dormant_readiness",
        )

    momentum = hist.get("institutional_momentum", "NONE")
    esc_cls = str(cls.get("escalation") or "").upper()
    if momentum in ("LOW", "MODERATE", "HIGH") and esc_cls == "NO_ESCALATION":
        _add(
            "informational",
            "Governance momentum exists, but human escalation remains inactive.",
            "momentum_vs_no_escalation",
        )

    regression_risk = forecast.get("regression_risk", "NONE")
    if regression_risk in ("MODERATE", "HIGH") and regime_key in (
        "GOVERNANCE_ACCELERATION",
        "PRE_RUNTIME_READINESS",
        "RUNTIME_CANDIDATE",
    ):
        _add(
            "warning",
            "Positive governance regime conflicts with elevated regression risk.",
            "regression_risk_vs_positive_regime",
        )

    if (
        momentum in ("LOW", "MODERATE", "HIGH")
        and forecast.get("runtime_discussion_probability") == "VERY_LOW"
    ):
        _add(
            "informational",
            "Governance momentum is rising while runtime discussion probability remains very low.",
            "momentum_vs_low_runtime_probability",
        )

    key_risks = _gcc_get(dossier, "key_risks") or []
    if esc_cls == "NO_ESCALATION" and key_risks:
        _add(
            "informational",
            "Human escalation is inactive while key governance risk signals remain elevated.",
            "inactive_escalation_vs_key_risks",
        )

    if hist.get("confidence_direction") == "improving" and regime_key == "CONSTITUTIONAL_STRESS":
        if not any(
            c["rule"] == "improving_trajectory_vs_constitutional_stress" for c in contradictions
        ):
            _add(
                "warning",
                "Confidence is improving while constitutional stress remains elevated.",
                "improving_confidence_vs_constitutional_stress",
            )

    critical_n = sum(1 for c in contradictions if c["severity"] == "critical")
    warning_n = sum(1 for c in contradictions if c["severity"] == "warning")
    info_n = sum(1 for c in contradictions if c["severity"] == "informational")
    total = len(contradictions)

    if critical_n > 0:
        tension_level = "CRITICAL_TENSION"
    elif warning_n >= 2 or (
        warning_n >= 1 and snap["future_runtime_candidate"] and snap["constitutional_safe"] is False
    ):
        tension_level = "HIGH_TENSION"
    elif total >= 2 or warning_n >= 1:
        tension_level = "MODERATE_TENSION"
    elif total == 1:
        tension_level = "LOW_TENSION"
    else:
        tension_level = "NO_TENSION"

    if tension_level == "NO_TENSION" and snap["max_conf"] <= 0.01 and snap["all_dormant"]:
        tension_confidence = 0.90
    elif total > 0:
        tension_confidence = min(0.55 + 0.10 * total + 0.12 * critical_n + 0.06 * warning_n, 0.98)
    else:
        tension_confidence = 0.75

    if not any(
        isinstance(d, dict) and d
        for d in (
            readiness,
            admission,
            eligibility,
            recommendation,
            review,
            verdict,
            dossier_summary,
        )
    ):
        tension_confidence = min(tension_confidence, 0.50)

    operator_response = _GCC_TENSION_OPERATOR_RESPONSE.get(tension_level, "CONTINUE_OBSERVATION")
    if critical_n > 0 or any(c["rule"].startswith("runtime_mutation") for c in contradictions):
        operator_response = "BLOCK_RUNTIME_DISCUSSION"

    return {
        "tension_level": tension_level,
        "tension_display": tension_level.replace("_", " ").title(),
        "tension_confidence": tension_confidence,
        "contradiction_count": total,
        "critical_count": critical_n,
        "warning_count": warning_n,
        "informational_count": info_n,
        "contradictions": contradictions,
        "operator_interpretation": _GCC_TENSION_INTERPRETATION.get(tension_level, ""),
        "operator_response": operator_response,
    }


def _gcc_render_tension_detection(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    tension = _gcc_detect_governance_tensions(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
    )

    st.markdown("### Governance Contradiction & Tension Detection")
    st.caption("Institutional friction analysis — where governance signals disagree.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tension Level", tension["tension_display"])
    c2.metric("Tension Confidence", _gcc_fmt_conf(tension["tension_confidence"]))
    c3.metric("Contradiction Count", str(tension["contradiction_count"]))
    c4.metric("Recommended Response", tension["operator_response"])

    count_bits = (
        f"Critical: {tension['critical_count']} · "
        f"Warning: {tension['warning_count']} · "
        f"Informational: {tension['informational_count']}"
    )
    st.caption(count_bits)

    level = tension["tension_level"]
    if level in ("HIGH_TENSION", "CRITICAL_TENSION"):
        st.error(tension["operator_interpretation"])
    elif level == "MODERATE_TENSION":
        st.warning(tension["operator_interpretation"])
    elif level == "LOW_TENSION":
        st.info(tension["operator_interpretation"])
    else:
        st.success(tension["operator_interpretation"])

    if tension["critical_count"] > 0:
        st.error(
            "**Safety-critical contradiction detected.** Review runtime mutation lock and "
            "constitutional safety before any runtime discussion."
        )

    st.markdown("**Key Contradictions**")
    if tension["contradictions"]:
        for item in tension["contradictions"]:
            sev = item["severity"]
            msg = item["message"]
            if sev == "critical":
                st.error(f"• {msg}")
            elif sev == "warning":
                st.warning(f"• {msg}")
            else:
                st.info(f"• {msg}")
    else:
        st.success("No governance contradictions detected. Institutional signals are aligned.")

    with st.expander("Tension analysis detail", expanded=False):
        st.markdown(f"- **Internal level:** `{tension['tension_level']}`")
        st.markdown(f"- **Regime:** `{regime.get('regime', '—')}`")
        st.markdown(f"- **Trajectory:** `{forecast.get('trajectory', '—')}`")
        st.markdown(f"- **Regression risk:** `{forecast.get('regression_risk', '—')}`")

    return tension


_GCC_CONSENSUS_PRIORITY: Tuple[str, ...] = (
    "CONFLICTED_GOVERNANCE",
    "FRAGMENTED_GOVERNANCE",
    "PRE_RUNTIME_CONVERGENCE",
    "CONSTITUTIONAL_CONSENSUS",
    "STRONG_CONSENSUS",
    "MODERATE_CONSENSUS",
)

_GCC_CONSENSUS_DISPLAY: Dict[str, str] = {
    "STRONG_CONSENSUS": "Strong Consensus",
    "MODERATE_CONSENSUS": "Moderate Consensus",
    "CONSTITUTIONAL_CONSENSUS": "Constitutional Consensus",
    "PRE_RUNTIME_CONVERGENCE": "Pre-Runtime Convergence",
    "FRAGMENTED_GOVERNANCE": "Fragmented Governance",
    "CONFLICTED_GOVERNANCE": "Conflicted Governance",
}

_GCC_CONSENSUS_OPERATOR_STANCE: Dict[str, str] = {
    "STRONG_CONSENSUS": "CONTINUE_OBSERVATION",
    "MODERATE_CONSENSUS": "CONTINUE_OBSERVATION",
    "CONSTITUTIONAL_CONSENSUS": "TRUST_CONSTITUTIONAL_POSTURE",
    "PRE_RUNTIME_CONVERGENCE": "MONITOR_CONVERGENCE",
    "FRAGMENTED_GOVERNANCE": "REVIEW_FRAGMENTATION",
    "CONFLICTED_GOVERNANCE": "BLOCK_RUNTIME_ASSUMPTIONS",
}

_GCC_CONSENSUS_INTERPRETATION: Dict[str, str] = {
    "STRONG_CONSENSUS": (
        "Governance signals broadly agree across status, regime, trajectory, and escalation posture. "
        "Institutional consensus is strong and posture appears internally consistent."
    ),
    "MODERATE_CONSENSUS": (
        "Governance is mostly aligned with minor tensions present. Institutional signals support "
        "the same broad conclusion with limited fragmentation."
    ),
    "CONSTITUTIONAL_CONSENSUS": (
        "Governance signals strongly agree that runtime governance should remain institutionally constrained. "
        "Constitutional safeguards, verdict posture, escalation inactivity, and trajectory intelligence are aligned."
    ),
    "PRE_RUNTIME_CONVERGENCE": (
        "Governance signals are gradually converging toward a more mature institutional posture, "
        "although runtime mutation remains locked."
    ),
    "FRAGMENTED_GOVERNANCE": (
        "Governance signals are partially fragmented. Multiple governance layers disagree on institutional "
        "direction, reducing decision cohesion."
    ),
    "CONFLICTED_GOVERNANCE": (
        "Governance contains major contradictions across recommendation, verdict, constitutional safety, "
        "or runtime lock posture. Institutional consensus is weak."
    ),
}


def _gcc_consensus_drivers(
    state: str,
    snap: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
) -> List[str]:
    cls = snap["classifications"]
    drivers: List[str] = []

    if state == "CONSTITUTIONAL_CONSENSUS":
        drivers.extend(
            [
                f"Runtime readiness posture: {cls.get('readiness') or '—'}",
                f"Constitutional eligibility: {cls.get('eligibility') or '—'}",
                f"Institutional verdict: {cls.get('verdict') or '—'}",
                f"Runtime discussion probability: {forecast.get('runtime_discussion_probability', '—')}",
                f"Human escalation: {cls.get('escalation') or '—'}",
            ]
        )
    elif state == "PRE_RUNTIME_CONVERGENCE":
        drivers.append(f"Trajectory: {forecast.get('trajectory_display', '—')}")
        drivers.append(f"Institutional momentum: {hist.get('institutional_momentum', '—')}")
        drivers.append(f"Confidence direction: {hist.get('confidence_direction', '—')}")
        drivers.append(f"Readiness posture: {cls.get('readiness') or '—'}")
        if str(cls.get("escalation") or "").upper() != "NO_ESCALATION":
            drivers.append(f"Escalation posture: {cls.get('escalation')}")
    elif state == "FRAGMENTED_GOVERNANCE":
        drivers.append(f"Tension level: {tension.get('tension_display', '—')}")
        drivers.append(f"Contradictions detected: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(f"Trajectory: {forecast.get('trajectory_display', '—')}")
        if tension.get("contradictions"):
            drivers.append(tension["contradictions"][0]["message"])
    elif state == "CONFLICTED_GOVERNANCE":
        drivers.append(f"Critical contradictions: {tension.get('critical_count', 0)}")
        drivers.append(f"Warning contradictions: {tension.get('warning_count', 0)}")
        for item in (tension.get("contradictions") or [])[:3]:
            drivers.append(item["message"])
    elif state == "STRONG_CONSENSUS":
        drivers.append(f"Posture stability: {hist.get('posture_stability', '—')}")
        drivers.append(f"Tension level: {tension.get('tension_display', '—')}")
        drivers.append("Governance stages show aligned restrictive posture")
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
    else:
        drivers.append(f"Tension level: {tension.get('tension_display', '—')}")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(f"Trajectory: {forecast.get('trajectory_display', '—')}")

    return drivers[:6]


def _gcc_alignment_strength(consensus_state: str, tension: Dict[str, Any]) -> str:
    level = tension.get("tension_level", "NO_TENSION")
    count = int(tension.get("contradiction_count", 0) or 0)
    if consensus_state == "CONFLICTED_GOVERNANCE" or level == "CRITICAL_TENSION":
        return "VERY_LOW"
    if consensus_state == "FRAGMENTED_GOVERNANCE" or level == "HIGH_TENSION":
        return "LOW"
    if consensus_state == "MODERATE_CONSENSUS" or level == "MODERATE_TENSION":
        return "MODERATE"
    if consensus_state in ("STRONG_CONSENSUS", "CONSTITUTIONAL_CONSENSUS") and count == 0:
        return "VERY_HIGH"
    if consensus_state in (
        "STRONG_CONSENSUS",
        "CONSTITUTIONAL_CONSENSUS",
        "PRE_RUNTIME_CONVERGENCE",
    ):
        return "HIGH"
    return "MODERATE"


def _gcc_consensus_confidence(
    consensus_state: str,
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    hist: Dict[str, Any],
) -> float:
    regime_conf = float(regime.get("regime_confidence", 0.55) or 0.55)
    forecast_conf = float(forecast.get("forecast_confidence", 0.55) or 0.55)
    tension_conf = float(tension.get("tension_confidence", 0.55) or 0.55)
    count = int(tension.get("contradiction_count", 0) or 0)
    base = 0.25 * regime_conf + 0.20 * forecast_conf + 0.20 * tension_conf

    if consensus_state in ("STRONG_CONSENSUS", "CONSTITUTIONAL_CONSENSUS"):
        base += 0.30
        if count == 0:
            base += 0.12
    elif consensus_state == "MODERATE_CONSENSUS":
        base += 0.15
    elif consensus_state == "PRE_RUNTIME_CONVERGENCE":
        base += 0.10
    elif consensus_state == "FRAGMENTED_GOVERNANCE":
        base += 0.05
    else:
        base = min(base, 0.50)

    if hist.get("confidence_direction") == "mixed":
        base -= 0.08
    if tension.get("critical_count", 0) > 0:
        base = min(base, 0.40)

    return min(max(base, 0.35), 0.98)


def _gcc_detect_governance_consensus(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    cls = snap["classifications"]
    tension_level = tension.get("tension_level", "NO_TENSION")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    critical_count = int(tension.get("critical_count", 0) or 0)
    trajectory = forecast.get("trajectory", "")
    regime_key = regime.get("regime", "")
    verdict_cls = str(cls.get("verdict") or "").upper()
    esc_cls = str(cls.get("escalation") or "").upper()

    def _match_conflicted() -> bool:
        return (
            critical_count > 0
            or tension_level in ("CRITICAL_TENSION", "HIGH_TENSION")
            or (tension_level == "MODERATE_TENSION" and contradiction_count >= 3)
        )

    def _match_fragmented() -> bool:
        return (
            tension_level in ("MODERATE_TENSION", "HIGH_TENSION") and contradiction_count >= 2
        ) or hist.get("confidence_direction") == "mixed"

    def _match_pre_runtime_convergence() -> bool:
        return (
            trajectory
            in ("GOVERNANCE_IMPROVING", "GOVERNANCE_ACCELERATING", "PRE_RUNTIME_TRAJECTORY")
            and hist.get("institutional_momentum") in ("LOW", "MODERATE", "HIGH")
            and hist.get("confidence_direction") in ("improving", "stable")
        )

    def _match_constitutional_consensus() -> bool:
        return (
            (regime_key == "CONSTITUTIONAL_STRESS" or snap["constitutional_safe"] is False)
            and verdict_cls == "DO_NOT_ENABLE_RUNTIME"
            and esc_cls == "NO_ESCALATION"
            and forecast.get("runtime_discussion_probability") in ("VERY_LOW", "LOW")
            and tension_level in ("NO_TENSION", "LOW_TENSION", "MODERATE_TENSION")
            and critical_count == 0
        )

    def _match_strong_consensus() -> bool:
        return (
            tension_level in ("NO_TENSION", "LOW_TENSION")
            and contradiction_count == 0
            and "Stable" in str(hist.get("posture_stability", ""))
        )

    matchers = {
        "CONFLICTED_GOVERNANCE": _match_conflicted,
        "FRAGMENTED_GOVERNANCE": _match_fragmented,
        "PRE_RUNTIME_CONVERGENCE": _match_pre_runtime_convergence,
        "CONSTITUTIONAL_CONSENSUS": _match_constitutional_consensus,
        "STRONG_CONSENSUS": _match_strong_consensus,
        "MODERATE_CONSENSUS": lambda: True,
    }

    consensus_state = "MODERATE_CONSENSUS"
    for candidate in _GCC_CONSENSUS_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            consensus_state = candidate
            break

    alignment = _gcc_alignment_strength(consensus_state, tension)
    confidence = _gcc_consensus_confidence(consensus_state, regime, forecast, tension, hist)
    drivers = _gcc_consensus_drivers(consensus_state, snap, hist, regime, forecast, tension)

    return {
        "consensus_state": consensus_state,
        "consensus_display": _GCC_CONSENSUS_DISPLAY.get(
            consensus_state, consensus_state.replace("_", " ").title()
        ),
        "consensus_confidence": confidence,
        "alignment_strength": alignment,
        "drivers": drivers,
        "interpretation": _GCC_CONSENSUS_INTERPRETATION.get(consensus_state, ""),
        "operator_stance": _GCC_CONSENSUS_OPERATOR_STANCE.get(
            consensus_state, "CONTINUE_OBSERVATION"
        ),
    }


def _gcc_render_consensus_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
) -> Dict[str, Any]:
    consensus = _gcc_detect_governance_consensus(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
    )

    st.markdown("### Governance Convergence / Consensus Intelligence")
    st.caption("Institutional agreement analysis — how aligned and cohesive governance posture is.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Consensus", consensus["consensus_display"])
    c2.metric("Consensus Confidence", _gcc_fmt_conf(consensus["consensus_confidence"]))
    c3.metric("Alignment Strength", consensus["alignment_strength"])
    c4.metric("Operator Stance", consensus["operator_stance"])

    state = consensus["consensus_state"]
    if state == "CONFLICTED_GOVERNANCE":
        st.error(consensus["interpretation"])
    elif state == "FRAGMENTED_GOVERNANCE":
        st.warning(consensus["interpretation"])
    elif state in ("STRONG_CONSENSUS", "CONSTITUTIONAL_CONSENSUS"):
        st.success(consensus["interpretation"])
    elif state == "PRE_RUNTIME_CONVERGENCE":
        st.info(consensus["interpretation"])
    else:
        st.info(consensus["interpretation"])

    st.markdown("**Consensus Drivers**")
    for driver in consensus["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    with st.expander("Consensus analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{consensus['consensus_state']}`")
        st.markdown(f"- **Tension level:** `{tension.get('tension_level', '—')}`")
        st.markdown(f"- **Contradiction count:** `{tension.get('contradiction_count', 0)}`")
        st.markdown(f"- **Posture stability:** {hist.get('posture_stability', '—')}")

    return consensus


_GCC_INTEGRITY_PRIORITY: Tuple[str, ...] = (
    "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
    "GOVERNANCE_OVERCONFIDENT",
    "GOVERNANCE_CONFIDENCE_WEAK",
    "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT",
    "GOVERNANCE_CONFIDENCE_TRUSTWORTHY",
    "GOVERNANCE_CONFIDENCE_PARTIALLY_RELIABLE",
)

_GCC_INTEGRITY_DISPLAY: Dict[str, str] = {
    "GOVERNANCE_CONFIDENCE_TRUSTWORTHY": "Governance Confidence Trustworthy",
    "GOVERNANCE_CONFIDENCE_PARTIALLY_RELIABLE": "Governance Confidence Partially Reliable",
    "GOVERNANCE_CONFIDENCE_WEAK": "Governance Confidence Weak",
    "GOVERNANCE_OVERCONFIDENT": "Governance Overconfident",
    "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS": "Confidence Undercut by Contradictions",
    "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT": "Dormant but Consistent",
}

_GCC_INTEGRITY_INTERPRETATION: Dict[str, str] = {
    "GOVERNANCE_CONFIDENCE_TRUSTWORTHY": (
        "Governance confidence is supported by institutional consensus, low tension, and stable posture. "
        "Operators may use confidence signals with standard caution."
    ),
    "GOVERNANCE_CONFIDENCE_PARTIALLY_RELIABLE": (
        "Governance confidence is mostly usable but some tension or fragmentation exists. "
        "Operators should treat confidence cautiously and corroborate with stage posture."
    ),
    "GOVERNANCE_CONFIDENCE_WEAK": (
        "Governance confidence has a weak evidential basis due to sparse history or limited maturity signals. "
        "Operators should require more evidence before relying on confidence."
    ),
    "GOVERNANCE_OVERCONFIDENT": (
        "Governance confidence appears stronger than institutional maturity supports. Runtime readiness, "
        "eligibility, and verdict posture remain restrictive."
    ),
    "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS": (
        "Governance confidence is weakened by contradictions across the institutional stack. "
        "Operators should discount confidence until fragmentation decreases."
    ),
    "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT": (
        "Governance confidence is limited, but the restrictive institutional posture is internally coherent. "
        "Operators should not treat dormant confidence as readiness."
    ),
}

_GCC_INTEGRITY_TRUST_POSTURE: Dict[str, str] = {
    "GOVERNANCE_CONFIDENCE_TRUSTWORTHY": "TRUST_WITH_CAUTION",
    "GOVERNANCE_CONFIDENCE_PARTIALLY_RELIABLE": "TRUST_WITH_CAUTION",
    "GOVERNANCE_CONFIDENCE_WEAK": "REQUIRE_MORE_EVIDENCE",
    "GOVERNANCE_OVERCONFIDENT": "DISCOUNT_CONFIDENCE",
    "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS": "REVIEW_CONFIDENCE_INTEGRITY",
    "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT": "TRUST_RESTRICTIVE_POSTURE",
}


def _gcc_integrity_drivers(
    state: str,
    snap: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    hist: Dict[str, Any],
) -> List[str]:
    cls = snap["classifications"]
    drivers: List[str] = []

    if state == "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT":
        drivers.extend(
            [
                f"Governance confidence remains dormant (max {_gcc_fmt_conf(snap['max_conf'])})",
                f"Runtime readiness: {cls.get('readiness') or '—'}",
                f"Institutional verdict: {cls.get('verdict') or '—'}",
                "Runtime mutation lock remains preserved",
            ]
        )
    elif state == "GOVERNANCE_OVERCONFIDENT":
        drivers.append(f"Regime confidence: {_gcc_fmt_conf(regime.get('regime_confidence'))}")
        drivers.append(f"Forecast confidence: {_gcc_fmt_conf(forecast.get('forecast_confidence'))}")
        drivers.append(f"Runtime readiness: {cls.get('readiness') or '—'}")
        drivers.append("Institutional maturity has not materially advanced")
    elif state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
        drivers.append(f"Tension level: {tension.get('tension_display', '—')}")
        drivers.append(f"Contradictions detected: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Consensus state: {consensus.get('consensus_display', '—')}")
        drivers.append("Confidence should be discounted")
    elif state == "GOVERNANCE_CONFIDENCE_WEAK":
        drivers.append(
            f"Historical depth: {'limited' if not hist.get('has_transitions') else 'partial'}"
        )
        drivers.append(f"Posture stability: {hist.get('posture_stability', '—')}")
        drivers.append(
            f"Consensus confidence: {_gcc_fmt_conf(consensus.get('consensus_confidence'))}"
        )
    elif state == "GOVERNANCE_CONFIDENCE_TRUSTWORTHY":
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
        drivers.append(f"Alignment strength: {consensus.get('alignment_strength', '—')}")
        drivers.append(f"Tension level: {tension.get('tension_display', '—')}")
        drivers.append("Confidence supported by institutional agreement")
    else:
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
        drivers.append(f"Tension level: {tension.get('tension_display', '—')}")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Alignment strength: {consensus.get('alignment_strength', '—')}")

    return drivers[:6]


def _gcc_overconfidence_risk(
    snap: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    readiness: Dict[str, Any],
) -> str:
    regime_conf = float(regime.get("regime_confidence", 0.0) or 0.0)
    forecast_conf = float(forecast.get("forecast_confidence", 0.0) or 0.0)
    high_conf = max(regime_conf, forecast_conf) >= 0.75
    readiness_cls = str(_gcc_get(readiness, "runtime_readiness_classification") or "").upper()
    readiness_conf = float(_gcc_get(readiness, "readiness_confidence", 0.0) or 0.0)
    count = int(tension.get("contradiction_count", 0) or 0)
    critical = int(tension.get("critical_count", 0) or 0)

    mutation_ok = all(
        _gcc_get(obj, "runtime_mutation_allowed") is False
        for obj in (readiness,)
        if isinstance(obj, dict) and obj
    )

    if high_conf and snap["future_runtime_candidate"] and snap["constitutional_safe"] is False:
        return "CRITICAL"
    if high_conf and (critical > 0 or not mutation_ok):
        return "CRITICAL"
    if high_conf and count >= 1:
        return "HIGH"
    if high_conf and readiness_cls == "NOT_RUNTIME_READY" and readiness_conf <= 0.05:
        return "MODERATE"
    if max(regime_conf, forecast_conf) >= 0.60 and tension.get("tension_level") in (
        "NO_TENSION",
        "LOW_TENSION",
    ):
        return "LOW"
    return "NONE"


def _gcc_confidence_discount(
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    overconfidence_risk: str,
    integrity_state: str,
) -> float:
    discount = 0.0
    count = int(tension.get("contradiction_count", 0) or 0)
    level = tension.get("tension_level", "NO_TENSION")

    if count > 0:
        discount += 0.08 * min(count, 4)
    if level == "MODERATE_TENSION":
        discount += 0.15
    elif level == "HIGH_TENSION":
        discount += 0.25
    elif level == "CRITICAL_TENSION":
        discount += 0.35
    if consensus.get("consensus_state") in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
        discount += 0.20
    if integrity_state == "GOVERNANCE_OVERCONFIDENT":
        discount += 0.25
    if overconfidence_risk == "CRITICAL":
        discount += 0.20
    elif overconfidence_risk == "HIGH":
        discount += 0.12

    return min(discount, 0.65)


def _gcc_trustworthiness_score(
    consensus: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    hist: Dict[str, Any],
    discount: float,
) -> float:
    consensus_conf = float(consensus.get("consensus_confidence", 0.55) or 0.55)
    regime_conf = float(regime.get("regime_confidence", 0.55) or 0.55)
    forecast_conf = float(forecast.get("forecast_confidence", 0.55) or 0.55)
    alignment = consensus.get("alignment_strength", "MODERATE")
    align_bonus = {
        "VERY_HIGH": 0.22,
        "HIGH": 0.16,
        "MODERATE": 0.08,
        "LOW": 0.0,
        "VERY_LOW": -0.08,
    }.get(alignment, 0.0)

    base = 0.30 * consensus_conf + 0.20 * regime_conf + 0.20 * forecast_conf + align_bonus
    if tension.get("tension_level") == "NO_TENSION":
        base += 0.12
    elif tension.get("tension_level") in ("HIGH_TENSION", "CRITICAL_TENSION"):
        base -= 0.15

    count = int(tension.get("contradiction_count", 0) or 0)
    base -= 0.06 * min(count, 5)

    if not hist.get("has_history"):
        base -= 0.10
    if hist.get("confidence_direction") == "mixed":
        base -= 0.08

    trust = min(max(base, 0.20), 0.98)
    return min(max(trust * (1.0 - discount), 0.15), 0.98)


def _gcc_detect_confidence_integrity(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    cls = snap["classifications"]
    regime_conf = float(regime.get("regime_confidence", 0.0) or 0.0)
    forecast_conf = float(forecast.get("forecast_confidence", 0.0) or 0.0)
    high_conf = (
        max(regime_conf, forecast_conf, float(consensus.get("consensus_confidence", 0.0) or 0.0))
        >= 0.75
    )
    readiness_cls = str(cls.get("readiness") or "").upper()
    readiness_conf = float(_gcc_get(readiness, "readiness_confidence", 0.0) or 0.0)
    consensus_state = consensus.get("consensus_state", "")
    tension_level = tension.get("tension_level", "NO_TENSION")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)

    def _match_undercut() -> bool:
        return consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE") or (
            contradiction_count >= 2 and tension_level != "NO_TENSION"
        )

    def _match_overconfident() -> bool:
        return high_conf and readiness_cls == "NOT_RUNTIME_READY" and readiness_conf <= 0.05

    def _match_weak() -> bool:
        return (
            not hist.get("has_history")
            or (snap["max_conf"] <= 0.01 and not hist.get("has_transitions"))
            or float(consensus.get("consensus_confidence", 0.0) or 0.0) < 0.55
        )

    def _match_dormant_consistent() -> bool:
        return (
            snap["max_conf"] <= 0.01
            and snap["all_dormant"]
            and tension_level in ("NO_TENSION", "LOW_TENSION", "MODERATE_TENSION")
            and int(tension.get("critical_count", 0) or 0) == 0
            and str(cls.get("verdict") or "").upper() == "DO_NOT_ENABLE_RUNTIME"
        )

    def _match_trustworthy() -> bool:
        return (
            consensus_state in ("STRONG_CONSENSUS", "CONSTITUTIONAL_CONSENSUS")
            and contradiction_count == 0
            and tension_level in ("NO_TENSION", "LOW_TENSION")
            and consensus.get("alignment_strength") in ("HIGH", "VERY_HIGH")
        )

    matchers = {
        "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS": _match_undercut,
        "GOVERNANCE_OVERCONFIDENT": _match_overconfident,
        "GOVERNANCE_CONFIDENCE_WEAK": _match_weak,
        "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT": _match_dormant_consistent,
        "GOVERNANCE_CONFIDENCE_TRUSTWORTHY": _match_trustworthy,
        "GOVERNANCE_CONFIDENCE_PARTIALLY_RELIABLE": lambda: True,
    }

    integrity_state = "GOVERNANCE_CONFIDENCE_PARTIALLY_RELIABLE"
    for candidate in _GCC_INTEGRITY_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            integrity_state = candidate
            break

    overconfidence_risk = _gcc_overconfidence_risk(snap, regime, forecast, tension, readiness)
    discount = _gcc_confidence_discount(tension, consensus, overconfidence_risk, integrity_state)
    trustworthiness = _gcc_trustworthiness_score(
        consensus, regime, forecast, tension, hist, discount
    )
    raw_confidence_context = (regime_conf + forecast_conf) / 2.0
    drivers = _gcc_integrity_drivers(
        integrity_state, snap, regime, forecast, tension, consensus, hist
    )

    trust_posture = _GCC_INTEGRITY_TRUST_POSTURE.get(integrity_state, "TRUST_WITH_CAUTION")
    if overconfidence_risk == "CRITICAL":
        trust_posture = "BLOCK_CONFIDENCE_BASED_ESCALATION"

    return {
        "integrity_state": integrity_state,
        "integrity_display": _GCC_INTEGRITY_DISPLAY.get(
            integrity_state, integrity_state.replace("_", " ").title()
        ),
        "trustworthiness_score": trustworthiness,
        "raw_confidence_context": raw_confidence_context,
        "overconfidence_risk": overconfidence_risk,
        "confidence_discount": discount,
        "discounted_trust_score": trustworthiness,
        "drivers": drivers,
        "interpretation": _GCC_INTEGRITY_INTERPRETATION.get(integrity_state, ""),
        "trust_posture": trust_posture,
    }


def _gcc_render_confidence_integrity(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
) -> Dict[str, Any]:
    integrity = _gcc_detect_confidence_integrity(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
    )

    st.markdown("### Governance Confidence Integrity & Trustworthiness")
    st.caption("Confidence calibration — whether institutional confidence can be trusted.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence Integrity", integrity["integrity_display"])
    c2.metric("Trustworthiness Score", _gcc_fmt_conf(integrity["trustworthiness_score"]))
    c3.metric("Overconfidence Risk", integrity["overconfidence_risk"])
    c4.metric("Confidence Discount", _gcc_fmt_conf(integrity["confidence_discount"]))

    st.caption(
        f"Raw confidence context (regime/forecast avg): {_gcc_fmt_conf(integrity['raw_confidence_context'])} · "
        f"Discounted trust score: {_gcc_fmt_conf(integrity['discounted_trust_score'])}"
    )

    state = integrity["integrity_state"]
    if state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
        st.error(integrity["interpretation"])
    elif state in ("GOVERNANCE_OVERCONFIDENT", "GOVERNANCE_CONFIDENCE_WEAK"):
        st.warning(integrity["interpretation"])
    elif state == "GOVERNANCE_CONFIDENCE_TRUSTWORTHY":
        st.success(integrity["interpretation"])
    elif state == "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT":
        st.info(integrity["interpretation"])
    else:
        st.info(integrity["interpretation"])

    if integrity["overconfidence_risk"] == "CRITICAL":
        st.error(
            "**Critical overconfidence risk.** Block confidence-based escalation until constitutional "
            "safety and runtime mutation lock are verified."
        )

    st.markdown("**Integrity Drivers**")
    for driver in integrity["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    st.metric("Recommended Trust Posture", integrity["trust_posture"])

    with st.expander("Confidence integrity detail", expanded=False):
        st.markdown(f"- **Internal state:** `{integrity['integrity_state']}`")
        st.markdown(
            f"- **Consensus confidence:** `{_gcc_fmt_conf(consensus.get('consensus_confidence'))}`"
        )
        st.markdown(f"- **Regime confidence:** `{_gcc_fmt_conf(regime.get('regime_confidence'))}`")
        st.markdown(
            f"- **Forecast confidence:** `{_gcc_fmt_conf(forecast.get('forecast_confidence'))}`"
        )

    return integrity


_GCC_READINESS_PRIORITY: Tuple[str, ...] = (
    "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE",
    "GOVERNANCE_REVIEW_ELIGIBLE",
    "LIMITED_REVIEW_WORTHINESS",
    "OBSERVATION_ONLY",
    "NOT_INSTITUTIONALLY_DISCUSSABLE",
)

_GCC_READINESS_DISPLAY: Dict[str, str] = {
    "NOT_INSTITUTIONALLY_DISCUSSABLE": "Not Institutionally Discussable",
    "OBSERVATION_ONLY": "Observation Only",
    "LIMITED_REVIEW_WORTHINESS": "Limited Review Worthiness",
    "GOVERNANCE_REVIEW_ELIGIBLE": "Governance Review Eligible",
    "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE": "Operator Committee Review Eligible",
}

_GCC_READINESS_ESCALATION: Dict[str, str] = {
    "NOT_INSTITUTIONALLY_DISCUSSABLE": "NONE",
    "OBSERVATION_ONLY": "VERY_LOW",
    "LIMITED_REVIEW_WORTHINESS": "LOW",
    "GOVERNANCE_REVIEW_ELIGIBLE": "MODERATE",
    "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE": "HIGH",
}

_GCC_READINESS_DISCUSSABILITY: Dict[str, str] = {
    "NOT_INSTITUTIONALLY_DISCUSSABLE": "NOT_DISCUSSABLE",
    "OBSERVATION_ONLY": "INTERNAL_OBSERVATION_ONLY",
    "LIMITED_REVIEW_WORTHINESS": "LIMITED_INTERNAL_REVIEW",
    "GOVERNANCE_REVIEW_ELIGIBLE": "GOVERNANCE_DISCUSSION_APPROPRIATE",
    "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE": "COMMITTEE_DISCUSSION_APPROPRIATE",
}

_GCC_READINESS_ACTION: Dict[str, str] = {
    "NOT_INSTITUTIONALLY_DISCUSSABLE": "MAINTAIN_CONSTITUTIONAL_LOCK",
    "OBSERVATION_ONLY": "CONTINUE_OBSERVATION",
    "LIMITED_REVIEW_WORTHINESS": "COLLECT_MORE_EVIDENCE",
    "GOVERNANCE_REVIEW_ELIGIBLE": "INITIATE_GOVERNANCE_REVIEW",
    "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE": "PREPARE_OPERATOR_COMMITTEE_REVIEW",
}

_GCC_READINESS_INTERPRETATION: Dict[str, str] = {
    "NOT_INSTITUTIONALLY_DISCUSSABLE": (
        "Governance maturity remains insufficient for institutional discussion. Constitutional constraints, "
        "fragmented posture, and weakened confidence integrity limit escalation appropriateness."
    ),
    "OBSERVATION_ONLY": (
        "Governance posture is internally observable but not yet mature enough for formal review. "
        "Operator observation and evidence collection remain appropriate."
    ),
    "LIMITED_REVIEW_WORTHINESS": (
        "Governance shows early signs of institutional maturation, but escalation remains premature. "
        "Observation and evidence collection remain appropriate."
    ),
    "GOVERNANCE_REVIEW_ELIGIBLE": (
        "Governance maturity has improved enough to justify limited institutional review, although "
        "runtime mutation remains explicitly locked."
    ),
    "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE": (
        "Governance convergence and confidence integrity support operator committee discussion. "
        "This remains review eligibility only — not runtime enablement or autonomy approval."
    ),
}


def _gcc_governance_maturity_score(
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    tension: Dict[str, Any],
    hist: Dict[str, Any],
    snap: Dict[str, Any],
) -> float:
    regime_conf = float(regime.get("regime_confidence", 0.0) or 0.0)
    forecast_conf = float(forecast.get("forecast_confidence", 0.0) or 0.0)
    consensus_conf = float(consensus.get("consensus_confidence", 0.0) or 0.0)
    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)

    score = 0.15 * regime_conf + 0.15 * forecast_conf + 0.15 * consensus_conf + 0.30 * trust
    if "Stable" in str(hist.get("posture_stability", "")):
        score += 0.08
    if hist.get("confidence_direction") == "improving":
        score += 0.06

    count = int(tension.get("contradiction_count", 0) or 0)
    score -= 0.06 * min(count, 5)

    if snap["constitutional_safe"] is False or snap["constitutional_review_required"]:
        score -= 0.10
    if consensus.get("consensus_state") in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
        score -= 0.12
    if integrity.get("integrity_state") == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
        score -= 0.15

    if snap["max_conf"] <= 0.01 and hist.get("confidence_direction") == "dormant":
        score = min(score, 0.28)

    return min(max(score, 0.0), 1.0)


def _gcc_next_governance_gate(
    readiness_state: str,
    integrity: Dict[str, Any],
    consensus: Dict[str, Any],
    tension: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
) -> str:
    if readiness_state == "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE":
        return "Prepare Committee Review"
    if readiness_state == "GOVERNANCE_REVIEW_ELIGIBLE":
        return "Prepare Governance Review"
    if consensus.get("consensus_state") in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
        return "Reduce Contradictions"
    if integrity.get("integrity_state") in (
        "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
        "GOVERNANCE_OVERCONFIDENT",
    ):
        return "Improve Confidence Integrity"
    if (
        regime.get("regime") == "CONSTITUTIONAL_STRESS"
        or forecast.get("trajectory") == "CONSTITUTIONALLY_CONSTRAINED"
    ):
        return "Improve Governance Stability"
    if consensus.get("alignment_strength") in ("LOW", "VERY_LOW"):
        return "Strengthen Institutional Consensus"
    if readiness_state == "LIMITED_REVIEW_WORTHINESS":
        return "Collect More Evidence"
    if readiness_state == "OBSERVATION_ONLY":
        return "Continue Observation"
    return "Maintain Constitutional Lock"


def _gcc_readiness_drivers(
    state: str,
    snap: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    hist: Dict[str, Any],
) -> List[str]:
    cls = snap["classifications"]
    drivers: List[str] = []

    if state == "NOT_INSTITUTIONALLY_DISCUSSABLE":
        if regime.get("regime") == "CONSTITUTIONAL_STRESS":
            drivers.append("Constitutional stress dominates governance")
        drivers.append(
            f"Runtime discussion probability: {forecast.get('runtime_discussion_probability', '—')}"
        )
        if integrity.get("integrity_state") == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
            drivers.append("Governance confidence undercut by contradictions")
        drivers.append(f"Escalation posture: {cls.get('escalation') or '—'}")
        drivers.append(f"Consensus state: {consensus.get('consensus_display', '—')}")
    elif state == "LIMITED_REVIEW_WORTHINESS":
        drivers.append(f"Trajectory: {forecast.get('trajectory_display', '—')}")
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
        drivers.append(f"Integrity: {integrity.get('integrity_display', '—')}")
        drivers.append("Institutional maturity remains limited")
    elif state == "GOVERNANCE_REVIEW_ELIGIBLE":
        drivers.append(f"Trajectory: {forecast.get('trajectory_display', '—')}")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Contradictions: {tension.get('contradiction_count', 0)}")
        drivers.append("Runtime mutation remains locked")
    elif state == "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE":
        drivers.append("Governance convergence strengthening")
        drivers.append(f"Contradictions low: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Trajectory: {forecast.get('trajectory_display', '—')}")
    elif state == "OBSERVATION_ONLY":
        drivers.append(f"Governance maturity score remains limited")
        drivers.append(f"Posture stability: {hist.get('posture_stability', '—')}")
        drivers.append(f"Human escalation: {cls.get('escalation') or '—'}")
        drivers.append("Institutional discussion not yet warranted")
    else:
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(f"Tension: {tension.get('tension_display', '—')}")

    return drivers[:6]


def _gcc_detect_decision_readiness(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    tension_level = tension.get("tension_level", "NO_TENSION")
    trajectory = forecast.get("trajectory", "")
    consensus_state = consensus.get("consensus_state", "")
    integrity_state = integrity.get("integrity_state", "")
    maturity = _gcc_governance_maturity_score(
        regime, forecast, consensus, integrity, tension, hist, snap
    )

    def _match_committee() -> bool:
        return (
            trajectory in ("PRE_RUNTIME_TRAJECTORY", "RUNTIME_DISCUSSION_CANDIDATE")
            and consensus_state in ("PRE_RUNTIME_CONVERGENCE", "STRONG_CONSENSUS")
            and trust >= 0.70
            and contradiction_count <= 1
            and tension_level in ("NO_TENSION", "LOW_TENSION")
            and integrity_state
            not in (
                "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
                "GOVERNANCE_OVERCONFIDENT",
            )
            and maturity >= 0.65
        )

    def _match_review_eligible() -> bool:
        return (
            trajectory
            in (
                "GOVERNANCE_IMPROVING",
                "GOVERNANCE_ACCELERATING",
                "PRE_RUNTIME_TRAJECTORY",
            )
            and consensus_state not in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            and trust >= 0.45
            and contradiction_count <= 2
            and integrity_state not in ("GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",)
            and maturity >= 0.40
        )

    def _match_limited() -> bool:
        if consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
            return False
        if integrity_state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
            return False
        return (
            hist.get("confidence_direction") in ("improving", "stable")
            or trajectory in ("GOVERNANCE_IMPROVING",)
            or float(snap["max_conf"]) > 0.01
        ) and maturity >= 0.18

    def _match_observation() -> bool:
        if consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
            return False
        if integrity_state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
            return False
        return (
            integrity_state == "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT"
            or consensus_state == "CONSTITUTIONAL_CONSENSUS"
            or (
                snap["max_conf"] <= 0.01
                and tension_level in ("NO_TENSION", "LOW_TENSION", "MODERATE_TENSION")
                and int(tension.get("critical_count", 0) or 0) == 0
            )
        )

    def _match_not_discussable() -> bool:
        return (
            consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            or integrity_state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS"
            or integrity.get("overconfidence_risk") == "CRITICAL"
            or tension_level in ("HIGH_TENSION", "CRITICAL_TENSION")
            or maturity < 0.18
        )

    matchers = {
        "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE": _match_committee,
        "GOVERNANCE_REVIEW_ELIGIBLE": _match_review_eligible,
        "LIMITED_REVIEW_WORTHINESS": _match_limited,
        "OBSERVATION_ONLY": _match_observation,
        "NOT_INSTITUTIONALLY_DISCUSSABLE": _match_not_discussable,
    }

    readiness_state = "NOT_INSTITUTIONALLY_DISCUSSABLE"
    for candidate in _GCC_READINESS_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            readiness_state = candidate
            break

    drivers = _gcc_readiness_drivers(
        readiness_state, snap, regime, forecast, tension, consensus, integrity, hist
    )
    next_gate = _gcc_next_governance_gate(
        readiness_state, integrity, consensus, tension, regime, forecast
    )

    return {
        "readiness_state": readiness_state,
        "readiness_display": _GCC_READINESS_DISPLAY.get(
            readiness_state, readiness_state.replace("_", " ").title()
        ),
        "escalation_readiness": _GCC_READINESS_ESCALATION.get(readiness_state, "NONE"),
        "discussability": _GCC_READINESS_DISCUSSABILITY.get(readiness_state, "NOT_DISCUSSABLE"),
        "governance_maturity_score": maturity,
        "drivers": drivers,
        "interpretation": _GCC_READINESS_INTERPRETATION.get(readiness_state, ""),
        "institutional_action": _GCC_READINESS_ACTION.get(readiness_state, "CONTINUE_OBSERVATION"),
        "next_governance_gate": next_gate,
    }


def _gcc_render_decision_readiness(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
) -> Dict[str, Any]:
    decision = _gcc_detect_decision_readiness(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
    )

    st.markdown("### Governance Decision Readiness & Institutional Escalation Intelligence")
    st.caption(
        "Institutional gate assessment — discussability and review eligibility only. "
        "**Not runtime enablement. Not autonomy approval.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision Readiness", decision["readiness_display"])
    c2.metric("Escalation Readiness", decision["escalation_readiness"])
    c3.metric("Discussability", decision["discussability"])
    c4.metric("Governance Maturity Score", _gcc_fmt_conf(decision["governance_maturity_score"]))

    state = decision["readiness_state"]
    if state == "NOT_INSTITUTIONALLY_DISCUSSABLE":
        st.error(decision["interpretation"])
    elif state in ("OBSERVATION_ONLY", "LIMITED_REVIEW_WORTHINESS"):
        st.info(decision["interpretation"])
    elif state == "GOVERNANCE_REVIEW_ELIGIBLE":
        st.warning(decision["interpretation"])
    else:
        st.success(decision["interpretation"])

    st.warning(
        "**Runtime mutation remains locked.** Decision readiness reflects institutional discussion "
        "eligibility only — not runtime enablement or policy mutation."
    )

    st.markdown("**Escalation Drivers**")
    for driver in decision["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Institutional Action", decision["institutional_action"])
    a2.metric("Next Governance Gate", decision["next_governance_gate"])

    with st.expander("Decision readiness detail", expanded=False):
        st.markdown(f"- **Internal state:** `{decision['readiness_state']}`")
        st.markdown(f"- **Integrity state:** `{integrity.get('integrity_state', '—')}`")
        st.markdown(f"- **Consensus state:** `{consensus.get('consensus_state', '—')}`")
        st.markdown(
            f"- **Trustworthiness:** `{_gcc_fmt_conf(integrity.get('discounted_trust_score'))}`"
        )

    return decision


_GCC_FAILURE_PRIORITY: Tuple[str, ...] = (
    "FALSE_CONFIDENCE_ESCALATION_RISK",
    "ESCALATION_WITHOUT_MATURITY_RISK",
    "CONSTITUTIONAL_DRIFT_RISK",
    "GOVERNANCE_FRAGMENTATION_RISK",
    "PREMATURE_REVIEW_RISK",
    "CONSENSUS_ILLUSION_RISK",
    "DECISION_PARALYSIS_RISK",
    "INSTITUTIONAL_OVERFITTING_RISK",
    "GOVERNANCE_STABLE",
)

_GCC_FAILURE_DISPLAY: Dict[str, str] = {
    "GOVERNANCE_STABLE": "Governance Stable",
    "GOVERNANCE_FRAGMENTATION_RISK": "Governance Fragmentation Risk",
    "FALSE_CONFIDENCE_ESCALATION_RISK": "False Confidence Escalation Risk",
    "CONSTITUTIONAL_DRIFT_RISK": "Constitutional Drift Risk",
    "PREMATURE_REVIEW_RISK": "Premature Review Risk",
    "CONSENSUS_ILLUSION_RISK": "Consensus Illusion Risk",
    "ESCALATION_WITHOUT_MATURITY_RISK": "Escalation Without Maturity Risk",
    "INSTITUTIONAL_OVERFITTING_RISK": "Institutional Overfitting Risk",
    "DECISION_PARALYSIS_RISK": "Decision Paralysis Risk",
}

_GCC_FAILURE_INTERPRETATION: Dict[str, str] = {
    "GOVERNANCE_STABLE": (
        "Governance failure-mode exposure is limited. Institutional signals are coherent, confidence integrity "
        "is acceptable, and escalation pressure remains appropriately constrained."
    ),
    "GOVERNANCE_FRAGMENTATION_RISK": (
        "Governance remains vulnerable to fragmentation. Contradictions and weak cohesion reduce institutional "
        "reliability and increase decision-process fragility."
    ),
    "FALSE_CONFIDENCE_ESCALATION_RISK": (
        "Governance confidence appears stronger than maturity supports. Escalation or review would risk "
        "violating Capital Preservation Doctrine safeguards."
    ),
    "CONSTITUTIONAL_DRIFT_RISK": (
        "Constitutional stress and review pressure create drift risk. Runtime governance should remain "
        "blocked until constitutional posture stabilizes."
    ),
    "PREMATURE_REVIEW_RISK": (
        "Governance maturity remains insufficient for institutional review despite emerging discussion signals."
    ),
    "CONSENSUS_ILLUSION_RISK": (
        "Apparent governance alignment may mask underlying contradictions or weakened confidence integrity."
    ),
    "ESCALATION_WITHOUT_MATURITY_RISK": (
        "Escalation or review signals are emerging without sufficient governance maturity to support them safely."
    ),
    "INSTITUTIONAL_OVERFITTING_RISK": (
        "Governance may be reacting too strongly to weak or sparse signals. Confidence exceeds evidential support."
    ),
    "DECISION_PARALYSIS_RISK": (
        "Extreme contradictions and weak direction increase decision paralysis risk across the institutional stack."
    ),
}

_GCC_FAILURE_SAFEGUARD: Dict[str, str] = {
    "GOVERNANCE_STABLE": "CONTINUE_OBSERVATION",
    "GOVERNANCE_FRAGMENTATION_RISK": "REDUCE_FRAGMENTATION",
    "FALSE_CONFIDENCE_ESCALATION_RISK": "BLOCK_ESCALATION",
    "CONSTITUTIONAL_DRIFT_RISK": "MAINTAIN_CONSTITUTIONAL_LOCK",
    "PREMATURE_REVIEW_RISK": "REQUIRE_MORE_EVIDENCE",
    "CONSENSUS_ILLUSION_RISK": "IMPROVE_CONFIDENCE_INTEGRITY",
    "ESCALATION_WITHOUT_MATURITY_RISK": "BLOCK_ESCALATION",
    "INSTITUTIONAL_OVERFITTING_RISK": "REQUIRE_MORE_EVIDENCE",
    "DECISION_PARALYSIS_RISK": "IMPROVE_GOVERNANCE_STABILITY",
}


def _gcc_fragility_score(
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    regime: Dict[str, Any],
    snap: Dict[str, Any],
) -> float:
    score = 0.0
    count = int(tension.get("contradiction_count", 0) or 0)
    score += 0.08 * min(count, 6)

    level = tension.get("tension_level", "NO_TENSION")
    if level == "MODERATE_TENSION":
        score += 0.12
    elif level == "HIGH_TENSION":
        score += 0.22
    elif level == "CRITICAL_TENSION":
        score += 0.32

    if consensus.get("consensus_state") in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
        score += 0.18
    if integrity.get("integrity_state") in (
        "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
        "GOVERNANCE_OVERCONFIDENT",
    ):
        score += 0.16
    if integrity.get("overconfidence_risk") in ("HIGH", "CRITICAL"):
        score += 0.14
    if snap["constitutional_safe"] is False or snap["constitutional_review_required"]:
        score += 0.10
    if float(decision.get("governance_maturity_score", 0.0) or 0.0) < 0.20:
        score += 0.10
    if regime.get("regime") == "CONSTITUTIONAL_STRESS":
        score += 0.08

    return min(max(score, 0.0), 1.0)


def _gcc_risk_severity(
    risk_state: str,
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    fragility: float,
) -> str:
    if (
        risk_state in ("FALSE_CONFIDENCE_ESCALATION_RISK", "ESCALATION_WITHOUT_MATURITY_RISK")
        and integrity.get("overconfidence_risk") == "CRITICAL"
    ):
        return "CRITICAL"
    if (
        risk_state in ("FALSE_CONFIDENCE_ESCALATION_RISK", "DECISION_PARALYSIS_RISK")
        and fragility >= 0.55
    ):
        return "CRITICAL"
    if (
        risk_state
        in (
            "FALSE_CONFIDENCE_ESCALATION_RISK",
            "ESCALATION_WITHOUT_MATURITY_RISK",
            "GOVERNANCE_FRAGMENTATION_RISK",
            "DECISION_PARALYSIS_RISK",
        )
        and fragility >= 0.40
    ):
        return "HIGH"
    if risk_state in (
        "CONSTITUTIONAL_DRIFT_RISK",
        "GOVERNANCE_FRAGMENTATION_RISK",
        "PREMATURE_REVIEW_RISK",
    ):
        return "MODERATE"
    if risk_state == "GOVERNANCE_STABLE":
        return "NONE"
    if fragility >= 0.25:
        return "LOW"
    return "NONE"


def _gcc_failure_containment(
    risk_state: str,
    decision: Dict[str, Any],
    integrity: Dict[str, Any],
    consensus: Dict[str, Any],
) -> str:
    if risk_state == "FALSE_CONFIDENCE_ESCALATION_RISK":
        return "Prevent escalation; discount confidence before any review"
    if risk_state == "ESCALATION_WITHOUT_MATURITY_RISK":
        return "Delay institutional discussion until maturity improves"
    if risk_state == "CONSTITUTIONAL_DRIFT_RISK":
        return "Maintain constitutional lock; reduce review pressure"
    if risk_state == "GOVERNANCE_FRAGMENTATION_RISK":
        return "Reduce contradictions before review"
    if risk_state == "PREMATURE_REVIEW_RISK":
        return "Maintain observation only; collect more evidence"
    if risk_state == "CONSENSUS_ILLUSION_RISK":
        return "Strengthen confidence integrity before trusting alignment"
    if risk_state == "DECISION_PARALYSIS_RISK":
        return "Improve governance stability before advancing gates"
    if risk_state == "INSTITUTIONAL_OVERFITTING_RISK":
        return "Require more evidence; avoid overreacting to weak signals"
    if decision.get("readiness_state") == "NOT_INSTITUTIONALLY_DISCUSSABLE":
        return "Maintain observation only"
    return "Continue observation with standard institutional caution"


def _gcc_dominant_failure_modes(
    risk_state: str,
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    consensus: Dict[str, Any],
    decision: Dict[str, Any],
) -> List[str]:
    modes: List[str] = []
    primary = _GCC_FAILURE_DISPLAY.get(risk_state, risk_state)
    modes.append(primary)

    if integrity.get("integrity_state") == "GOVERNANCE_OVERCONFIDENT":
        modes.append("Confidence stronger than institutional maturity")
    if consensus.get("consensus_state") in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
        modes.append("Governance cohesion deteriorating")
    if int(tension.get("contradiction_count", 0) or 0) >= 2:
        modes.append("Contradictions elevated across governance layers")
    if (
        decision.get("escalation_readiness") not in ("NONE", "VERY_LOW")
        and float(decision.get("governance_maturity_score", 0.0) or 0.0) < 0.25
    ):
        modes.append("Escalation unsupported by readiness")
    if integrity.get("overconfidence_risk") in ("HIGH", "CRITICAL"):
        modes.append("False certainty risk elevated")

    seen: set = set()
    out: List[str] = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out[:6]


def _gcc_failure_drivers(
    snap: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    consensus: Dict[str, Any],
    decision: Dict[str, Any],
    regime: Dict[str, Any],
) -> List[str]:
    drivers: List[str] = []
    if float(integrity.get("confidence_discount", 0.0) or 0.0) > 0.10:
        drivers.append(
            f"Trustworthiness discounted ({_gcc_fmt_conf(integrity.get('confidence_discount'))})"
        )
    if float(decision.get("governance_maturity_score", 0.0) or 0.0) < 0.25:
        drivers.append("Governance maturity insufficient")
    if snap["constitutional_safe"] is False or snap["constitutional_review_required"]:
        drivers.append("Constitutional stress elevated")
    cls = snap["classifications"]
    if str(cls.get("readiness") or "").upper().startswith("NOT"):
        drivers.append(f"Runtime readiness: {cls.get('readiness') or '—'}")
    if consensus.get("consensus_state") in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
        drivers.append("Consensus fragmented")
    if decision.get("escalation_readiness") in ("NONE", "VERY_LOW"):
        drivers.append("Escalation inappropriate at current maturity")
    if regime.get("regime") == "CONSTITUTIONAL_STRESS":
        drivers.append("Constitutional stress regime active")
    return drivers[:6]


def _gcc_detect_governance_failure_modes(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)
    raw_conf = float(integrity.get("raw_confidence_context", 0.0) or 0.0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    consensus_state = consensus.get("consensus_state", "")
    integrity_state = integrity.get("integrity_state", "")
    overconf = integrity.get("overconfidence_risk", "NONE")
    readiness_cls = str(snap["classifications"].get("readiness") or "").upper()
    discussability = decision.get("discussability", "NOT_DISCUSSABLE")
    escalation = decision.get("escalation_readiness", "NONE")

    def _match_false_confidence() -> bool:
        return (
            integrity_state == "GOVERNANCE_OVERCONFIDENT"
            or overconf in ("HIGH", "CRITICAL")
            or (raw_conf >= 0.75 and trust < 0.35 and readiness_cls == "NOT_RUNTIME_READY")
        )

    def _match_escalation_without_maturity() -> bool:
        return (
            discussability not in ("NOT_DISCUSSABLE", "INTERNAL_OBSERVATION_ONLY")
            and maturity < 0.25
            and escalation in ("LOW", "MODERATE", "HIGH")
        )

    def _match_constitutional_drift() -> bool:
        return (
            snap["constitutional_safe"] is False
            or snap["constitutional_review_required"]
            or regime.get("regime") == "CONSTITUTIONAL_STRESS"
        ) and hist.get("confidence_direction") in ("deteriorating", "mixed", "dormant")

    def _match_fragmentation() -> bool:
        return (
            consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            or contradiction_count >= 2
        )

    def _match_premature_review() -> bool:
        return (
            decision.get("readiness_state")
            in ("LIMITED_REVIEW_WORTHINESS", "GOVERNANCE_REVIEW_ELIGIBLE")
            and maturity < 0.35
        )

    def _match_consensus_illusion() -> bool:
        return consensus_state in (
            "CONSTITUTIONAL_CONSENSUS",
            "MODERATE_CONSENSUS",
            "STRONG_CONSENSUS",
        ) and (
            integrity_state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS"
            or contradiction_count >= 1
        )

    def _match_decision_paralysis() -> bool:
        return consensus_state == "CONFLICTED_GOVERNANCE" or (
            contradiction_count >= 3
            and tension.get("tension_level") in ("HIGH_TENSION", "CRITICAL_TENSION")
        )

    def _match_overfitting() -> bool:
        return (
            raw_conf >= 0.70
            and not hist.get("has_transitions")
            and maturity < 0.30
            and hist.get("confidence_direction") in ("improving", "stable")
        )

    def _match_stable() -> bool:
        return (
            contradiction_count == 0
            and tension.get("tension_level") in ("NO_TENSION", "LOW_TENSION")
            and integrity_state
            in (
                "GOVERNANCE_CONFIDENCE_TRUSTWORTHY",
                "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT",
                "GOVERNANCE_CONFIDENCE_PARTIALLY_RELIABLE",
            )
            and overconf in ("NONE", "LOW")
        )

    matchers = {
        "FALSE_CONFIDENCE_ESCALATION_RISK": _match_false_confidence,
        "ESCALATION_WITHOUT_MATURITY_RISK": _match_escalation_without_maturity,
        "CONSTITUTIONAL_DRIFT_RISK": _match_constitutional_drift,
        "GOVERNANCE_FRAGMENTATION_RISK": _match_fragmentation,
        "PREMATURE_REVIEW_RISK": _match_premature_review,
        "CONSENSUS_ILLUSION_RISK": _match_consensus_illusion,
        "DECISION_PARALYSIS_RISK": _match_decision_paralysis,
        "INSTITUTIONAL_OVERFITTING_RISK": _match_overfitting,
        "GOVERNANCE_STABLE": _match_stable,
    }

    risk_state = "GOVERNANCE_STABLE"
    for candidate in _GCC_FAILURE_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            risk_state = candidate
            break

    fragility = _gcc_fragility_score(tension, consensus, integrity, decision, regime, snap)
    severity = _gcc_risk_severity(risk_state, tension, integrity, fragility)
    failure_modes = _gcc_dominant_failure_modes(risk_state, tension, integrity, consensus, decision)
    drivers = _gcc_failure_drivers(snap, tension, integrity, consensus, decision, regime)
    safeguard = _GCC_FAILURE_SAFEGUARD.get(risk_state, "CONTINUE_OBSERVATION")
    containment = _gcc_failure_containment(risk_state, decision, integrity, consensus)

    return {
        "risk_state": risk_state,
        "risk_display": _GCC_FAILURE_DISPLAY.get(risk_state, risk_state.replace("_", " ").title()),
        "risk_severity": severity,
        "fragility_score": fragility,
        "failure_modes": failure_modes,
        "drivers": drivers,
        "interpretation": _GCC_FAILURE_INTERPRETATION.get(risk_state, ""),
        "safeguard": safeguard,
        "containment_strategy": containment,
    }


def _gcc_render_failure_mode_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    failure = _gcc_detect_governance_failure_modes(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
    )

    st.markdown("### Governance Failure Modes & Institutional Risk Intelligence")
    st.caption(
        "Meta-governance risk analysis — how governance itself could fail. "
        "**Not runtime enablement. Not autonomy approval.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Risk State", failure["risk_display"])
    c2.metric("Institutional Risk Severity", failure["risk_severity"])
    c3.metric("Governance Fragility Score", _gcc_fmt_conf(failure["fragility_score"]))
    c4.metric("Governance Safeguard", failure["safeguard"])

    state = failure["risk_state"]
    sev = failure["risk_severity"]
    if sev == "CRITICAL" or state in (
        "FALSE_CONFIDENCE_ESCALATION_RISK",
        "DECISION_PARALYSIS_RISK",
    ):
        st.error(failure["interpretation"])
    elif sev in ("HIGH", "MODERATE") or state in (
        "GOVERNANCE_FRAGMENTATION_RISK",
        "CONSTITUTIONAL_DRIFT_RISK",
        "ESCALATION_WITHOUT_MATURITY_RISK",
    ):
        st.warning(failure["interpretation"])
    elif state == "GOVERNANCE_STABLE":
        st.success(failure["interpretation"])
    else:
        st.info(failure["interpretation"])

    st.markdown("**Dominant Failure Modes**")
    for mode in failure["failure_modes"]:
        if sev in ("CRITICAL", "HIGH"):
            st.warning(f"• {mode}")
        else:
            st.markdown(f'<div class="gcc-hist-metric">• {mode}</div>', unsafe_allow_html=True)

    st.markdown("**Failure Drivers**")
    for driver in failure["drivers"]:
        st.markdown(f'<div class="gcc-block-item">• {driver}</div>', unsafe_allow_html=True)

    st.metric("Failure Containment Strategy", failure["containment_strategy"])

    with st.expander("Failure mode analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{failure['risk_state']}`")
        st.markdown(f"- **Decision readiness:** `{decision.get('readiness_state', '—')}`")
        st.markdown(f"- **Integrity state:** `{integrity.get('integrity_state', '—')}`")
        st.markdown(f"- **Overconfidence risk:** `{integrity.get('overconfidence_risk', '—')}`")

    return failure


_GCC_AUDITABILITY_PRIORITY: Tuple[str, ...] = (
    "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",
    "EVIDENCE_FRAGMENTED",
    "LOW_AUDITABILITY",
    "SPARSE_EVIDENCE",
    "MODERATE_AUDITABILITY",
    "HIGH_AUDITABILITY",
)

_GCC_AUDITABILITY_DISPLAY: Dict[str, str] = {
    "HIGH_AUDITABILITY": "High Auditability",
    "MODERATE_AUDITABILITY": "Moderate Auditability",
    "LOW_AUDITABILITY": "Low Auditability",
    "SPARSE_EVIDENCE": "Sparse Evidence",
    "EVIDENCE_FRAGMENTED": "Evidence Fragmented",
    "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE": "Confidence Unsupported by Evidence",
}

_GCC_AUDITABILITY_INTERPRETATION: Dict[str, str] = {
    "HIGH_AUDITABILITY": (
        "Governance conclusions are institutionally traceable and sufficiently evidenced. "
        "Consensus, trajectory, and confidence integrity broadly align."
    ),
    "MODERATE_AUDITABILITY": (
        "Governance reasoning is mostly explainable, though some evidence weakness remains. "
        "Confidence is partially supported by institutional signals."
    ),
    "LOW_AUDITABILITY": (
        "Governance is difficult to justify with current evidence. Confidence integrity and "
        "institutional traceability remain weak."
    ),
    "SPARSE_EVIDENCE": (
        "Governance conclusions remain weakly evidenced. Institutional posture is traceable but "
        "supported by limited maturity and sparse historical progression."
    ),
    "EVIDENCE_FRAGMENTED": (
        "Governance reasoning remains partially fragmented. Contradictions reduce institutional "
        "traceability and weaken auditability."
    ),
    "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE": (
        "Governance confidence exceeds what current maturity and evidence depth support. "
        "Operators should treat confidence as insufficiently evidenced."
    ),
}

_GCC_AUDITABILITY_ACTION: Dict[str, str] = {
    "HIGH_AUDITABILITY": "MAINTAIN_OBSERVATION",
    "MODERATE_AUDITABILITY": "CONTINUE_INSTITUTIONAL_MONITORING",
    "LOW_AUDITABILITY": "AUDIT_GOVERNANCE_SIGNAL_QUALITY",
    "SPARSE_EVIDENCE": "COLLECT_MORE_EVIDENCE",
    "EVIDENCE_FRAGMENTED": "REDUCE_CONTRADICTIONS",
    "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE": "IMPROVE_CONFIDENCE_SUPPORT",
}


def _gcc_evidence_integrity_score(
    hist: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    tension: Dict[str, Any],
) -> float:
    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)
    consensus_conf = float(consensus.get("consensus_confidence", 0.0) or 0.0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    transition_n = len(hist.get("transitions") or [])

    score = 0.25 * trust + 0.20 * consensus_conf + 0.15 * maturity
    if hist.get("has_history"):
        score += 0.08
    if hist.get("has_transitions"):
        score += 0.10
    score += 0.02 * min(transition_n, 5)

    count = int(tension.get("contradiction_count", 0) or 0)
    score -= 0.06 * min(count, 5)
    score -= 0.12 * float(integrity.get("confidence_discount", 0.0) or 0.0)

    if integrity.get("integrity_state") == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
        score -= 0.12
    if integrity.get("overconfidence_risk") in ("HIGH", "CRITICAL"):
        score -= 0.15

    return min(max(score, 0.0), 1.0)


def _gcc_evidence_depth(
    hist: Dict[str, Any],
    snap: Dict[str, Any],
    evidence_score: float,
) -> str:
    transition_n = len(hist.get("transitions") or [])
    if snap["max_conf"] <= 0.01 and not hist.get("has_transitions"):
        return "VERY_LOW"
    if evidence_score < 0.20:
        return "LOW"
    if transition_n >= 2 or evidence_score >= 0.55:
        return "HIGH"
    if transition_n >= 1 or evidence_score >= 0.35:
        return "MODERATE"
    return "VERY_LOW" if evidence_score < 0.25 else "LOW"


def _gcc_explainability_quality(
    audit_state: str,
    hist: Dict[str, Any],
    evidence_score: float,
) -> str:
    if audit_state == "HIGH_AUDITABILITY" and hist.get("has_transitions"):
        return "HIGHLY_TRACEABLE"
    if audit_state == "HIGH_AUDITABILITY":
        return "STRONG"
    if audit_state == "MODERATE_AUDITABILITY":
        return "ADEQUATE"
    if audit_state in ("SPARSE_EVIDENCE", "LOW_AUDITABILITY"):
        return "LIMITED"
    return "POOR"


def _gcc_traceability_status(
    audit_state: str,
    hist: Dict[str, Any],
    evidence_score: float,
) -> str:
    if audit_state == "HIGH_AUDITABILITY" and hist.get("has_transitions"):
        return "HIGHLY_TRACEABLE"
    if audit_state in ("HIGH_AUDITABILITY", "MODERATE_AUDITABILITY") and evidence_score >= 0.40:
        return "TRACEABLE"
    if audit_state in ("SPARSE_EVIDENCE", "MODERATE_AUDITABILITY", "LOW_AUDITABILITY"):
        return "PARTIALLY_TRACEABLE"
    return "NOT_TRACEABLE"


def _gcc_audit_drivers(
    audit_state: str,
    hist: Dict[str, Any],
    snap: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
) -> List[str]:
    cls = snap["classifications"]
    drivers: List[str] = []

    if audit_state == "SPARSE_EVIDENCE":
        drivers.append("Governance history limited")
        drivers.append(f"Runtime readiness: {cls.get('readiness') or '—'}")
        drivers.append("Confidence supported by limited maturity evidence")
        drivers.append(f"Institutional transitions detected: {len(hist.get('transitions') or [])}")
    elif audit_state == "EVIDENCE_FRAGMENTED":
        drivers.append("Contradictions weaken institutional traceability")
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
        drivers.append("Governance cohesion reduced")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
    elif audit_state == "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE":
        drivers.append("Confidence stronger than maturity")
        drivers.append(
            f"Maturity score: {_gcc_fmt_conf(decision.get('governance_maturity_score'))}"
        )
        drivers.append(
            f"Trustworthiness discounted ({_gcc_fmt_conf(integrity.get('confidence_discount'))})"
        )
        drivers.append(f"Overconfidence risk: {integrity.get('overconfidence_risk', '—')}")
    elif audit_state == "HIGH_AUDITABILITY":
        drivers.append(f"Posture stability: {hist.get('posture_stability', '—')}")
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
        drivers.append("Governance reasoning broadly coherent")
    else:
        drivers.append(f"Integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(f"Tension: {tension.get('tension_display', '—')}")
        drivers.append(f"Evidence transitions: {len(hist.get('transitions') or [])}")

    return drivers[:6]


def _gcc_detect_governance_auditability(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)
    raw_conf = float(integrity.get("raw_confidence_context", 0.0) or 0.0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    consensus_state = consensus.get("consensus_state", "")
    integrity_state = integrity.get("integrity_state", "")
    overconf = integrity.get("overconfidence_risk", "NONE")

    evidence_score = _gcc_evidence_integrity_score(hist, consensus, integrity, decision, tension)

    def _match_unsupported() -> bool:
        return (
            integrity_state == "GOVERNANCE_OVERCONFIDENT"
            or overconf in ("HIGH", "CRITICAL")
            or (raw_conf >= 0.75 and trust < 0.35 and maturity < 0.25)
        )

    def _match_fragmented() -> bool:
        return (
            consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            or contradiction_count >= 2
        )

    def _match_low() -> bool:
        return (
            trust < 0.35
            or evidence_score < 0.25
            or integrity_state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS"
        )

    def _match_sparse() -> bool:
        return (
            not hist.get("has_transitions")
            and snap["max_conf"] <= 0.01
            and hist.get("confidence_direction") == "dormant"
        )

    def _match_moderate() -> bool:
        return evidence_score >= 0.25 and contradiction_count <= 2 and trust >= 0.20

    def _match_high() -> bool:
        return (
            contradiction_count == 0
            and tension.get("tension_level") in ("NO_TENSION", "LOW_TENSION")
            and trust >= 0.55
            and evidence_score >= 0.50
            and consensus_state not in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            and failure.get("risk_state") == "GOVERNANCE_STABLE"
        )

    matchers = {
        "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE": _match_unsupported,
        "EVIDENCE_FRAGMENTED": _match_fragmented,
        "LOW_AUDITABILITY": _match_low,
        "SPARSE_EVIDENCE": _match_sparse,
        "MODERATE_AUDITABILITY": _match_moderate,
        "HIGH_AUDITABILITY": _match_high,
    }

    audit_state = "MODERATE_AUDITABILITY"
    for candidate in _GCC_AUDITABILITY_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            audit_state = candidate
            break

    evidence_depth = _gcc_evidence_depth(hist, snap, evidence_score)
    explainability = _gcc_explainability_quality(audit_state, hist, evidence_score)
    traceability = _gcc_traceability_status(audit_state, hist, evidence_score)
    drivers = _gcc_audit_drivers(audit_state, hist, snap, tension, consensus, integrity, decision)

    return {
        "audit_state": audit_state,
        "audit_display": _GCC_AUDITABILITY_DISPLAY.get(
            audit_state, audit_state.replace("_", " ").title()
        ),
        "evidence_integrity_score": evidence_score,
        "evidence_depth": evidence_depth,
        "explainability_quality": explainability,
        "drivers": drivers,
        "interpretation": _GCC_AUDITABILITY_INTERPRETATION.get(audit_state, ""),
        "evidence_action": _GCC_AUDITABILITY_ACTION.get(
            audit_state, "CONTINUE_INSTITUTIONAL_MONITORING"
        ),
        "traceability_status": traceability,
    }


def _gcc_render_auditability_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
) -> Dict[str, Any]:
    audit = _gcc_detect_governance_auditability(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
    )

    st.markdown("### Governance Institutional Auditability & Evidence Integrity")
    st.caption(
        "Evidence and traceability analysis — whether governance conclusions are sufficiently supported. "
        "**Read-only audit view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Auditability State", audit["audit_display"])
    c2.metric("Evidence Integrity Score", _gcc_fmt_conf(audit["evidence_integrity_score"]))
    c3.metric("Evidence Depth", audit["evidence_depth"])
    c4.metric("Explainability Quality", audit["explainability_quality"])

    state = audit["audit_state"]
    if state in ("CONFIDENCE_UNSUPPORTED_BY_EVIDENCE", "EVIDENCE_FRAGMENTED", "LOW_AUDITABILITY"):
        st.warning(audit["interpretation"])
    elif state == "HIGH_AUDITABILITY":
        st.success(audit["interpretation"])
    elif state == "SPARSE_EVIDENCE":
        st.info(audit["interpretation"])
    else:
        st.info(audit["interpretation"])

    st.markdown("**Audit Drivers**")
    for driver in audit["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Evidence Action", audit["evidence_action"])
    a2.metric("Governance Traceability Status", audit["traceability_status"])

    with st.expander("Auditability analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{audit['audit_state']}`")
        st.markdown(f"- **Transition count:** `{len(hist.get('transitions') or [])}`")
        st.markdown(
            f"- **Trustworthiness:** `{_gcc_fmt_conf(integrity.get('discounted_trust_score'))}`"
        )
        st.markdown(f"- **Failure risk state:** `{failure.get('risk_state', '—')}`")

    return audit


_GCC_COHERENCE_PRIORITY: Tuple[str, ...] = (
    "ESCALATION_LOGIC_MISMATCH",
    "INTERNALLY_INCONSISTENT_GOVERNANCE",
    "FRAGMENTED_REASONING_CHAIN",
    "LOW_COHERENCE_GOVERNANCE",
    "LOGICALLY_CONSTRAINED_BUT_COHERENT",
    "MODERATELY_COHERENT_GOVERNANCE",
    "HIGHLY_COHERENT_GOVERNANCE",
)

_GCC_COHERENCE_DISPLAY: Dict[str, str] = {
    "HIGHLY_COHERENT_GOVERNANCE": "Highly Coherent Governance",
    "MODERATELY_COHERENT_GOVERNANCE": "Moderately Coherent Governance",
    "LOGICALLY_CONSTRAINED_BUT_COHERENT": "Logically Constrained but Coherent",
    "LOW_COHERENCE_GOVERNANCE": "Low Coherence Governance",
    "INTERNALLY_INCONSISTENT_GOVERNANCE": "Internally Inconsistent Governance",
    "FRAGMENTED_REASONING_CHAIN": "Fragmented Reasoning Chain",
    "ESCALATION_LOGIC_MISMATCH": "Escalation Logic Mismatch",
}

_GCC_COHERENCE_INTERPRETATION: Dict[str, str] = {
    "HIGHLY_COHERENT_GOVERNANCE": (
        "Governance sections broadly agree. Confidence, trajectory, maturity, risk, and "
        "discussability align into a coherent institutional narrative."
    ),
    "MODERATELY_COHERENT_GOVERNANCE": (
        "Governance mostly makes institutional sense. Minor inconsistencies exist, but the "
        "committee-level story remains usable."
    ),
    "LOGICALLY_CONSTRAINED_BUT_COHERENT": (
        "Governance remains institutionally constrained but internally coherent. Constitutional "
        "safeguards, maturity limits, escalation posture, and auditability broadly agree."
    ),
    "LOW_COHERENCE_GOVERNANCE": (
        "Institutional reasoning is weak. Governance sections partially disagree and the "
        "narrative remains unstable."
    ),
    "INTERNALLY_INCONSISTENT_GOVERNANCE": (
        "Governance sections materially conflict. The reasoning chain is difficult to justify "
        "at committee level."
    ),
    "FRAGMENTED_REASONING_CHAIN": (
        "Governance reasoning remains fragmented. Institutional sections disagree on maturity, "
        "trustworthiness, and escalation appropriateness."
    ),
    "ESCALATION_LOGIC_MISMATCH": (
        "Governance escalation logic is unsupported by institutional maturity and evidence "
        "integrity. Committee-level reasoning remains insufficient."
    ),
}

_GCC_COHERENCE_ACTION: Dict[str, str] = {
    "HIGHLY_COHERENT_GOVERNANCE": "CONTINUE_OBSERVATION",
    "MODERATELY_COHERENT_GOVERNANCE": "CONTINUE_OBSERVATION",
    "LOGICALLY_CONSTRAINED_BUT_COHERENT": "MAINTAIN_CONSTITUTIONAL_LOGIC",
    "LOW_COHERENCE_GOVERNANCE": "IMPROVE_GOVERNANCE_COHERENCE",
    "INTERNALLY_INCONSISTENT_GOVERNANCE": "IMPROVE_CROSS_SECTION_ALIGNMENT",
    "FRAGMENTED_REASONING_CHAIN": "REDUCE_REASONING_FRAGMENTATION",
    "ESCALATION_LOGIC_MISMATCH": "BLOCK_ESCALATION_ASSUMPTIONS",
}

_GCC_REGIME_TRAJECTORY_ALIGNED: Tuple[Tuple[str, str], ...] = (
    ("CONSTITUTIONAL_STRESS", "CONSTITUTIONALLY_CONSTRAINED"),
    ("GOVERNANCE_REGRESSION", "GOVERNANCE_STALLED"),
    ("INSTITUTIONAL_INSTABILITY", "GOVERNANCE_STALLED"),
    ("EARLY_INSTITUTIONAL_FORMATION", "GOVERNANCE_FORMING"),
    ("RUNTIME_CANDIDATE", "PRE_RUNTIME_TRAJECTORY"),
    ("PRE_RUNTIME_READINESS", "PRE_RUNTIME_TRAJECTORY"),
    ("GOVERNANCE_ACCELERATION", "GOVERNANCE_ACCELERATING"),
)


def _gcc_regime_trajectory_aligned(regime: Dict[str, Any], forecast: Dict[str, Any]) -> bool:
    regime_key = regime.get("regime", "")
    trajectory = forecast.get("trajectory", "")
    if (regime_key, trajectory) in _GCC_REGIME_TRAJECTORY_ALIGNED:
        return True
    if regime_key in ("CONSTITUTIONAL_STRESS", "GOVERNANCE_REGRESSION") and trajectory in (
        "CONSTITUTIONALLY_CONSTRAINED",
        "GOVERNANCE_STALLED",
        "GOVERNANCE_DETERIORATING",
    ):
        return True
    return False


def _gcc_institutional_logic_score(
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
) -> float:
    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)
    consensus_conf = float(consensus.get("consensus_confidence", 0.0) or 0.0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    evidence = float(audit.get("evidence_integrity_score", 0.0) or 0.0)
    count = int(tension.get("contradiction_count", 0) or 0)

    score = 0.18 * trust + 0.18 * consensus_conf + 0.14 * maturity + 0.12 * evidence
    if _gcc_regime_trajectory_aligned(regime, forecast):
        score += 0.12
    if failure.get("risk_state") == "GOVERNANCE_STABLE":
        score += 0.08
    discussable = decision.get("discussability", "NOT_DISCUSSABLE")
    escalation = decision.get("escalation_readiness", "NONE")
    if discussable in ("NOT_DISCUSSABLE", "INTERNAL_OBSERVATION_ONLY") and escalation in (
        "NONE",
        "VERY_LOW",
    ):
        score += 0.06

    score -= 0.08 * min(count, 5)
    if consensus.get("consensus_state") in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
        score -= 0.18
    if integrity.get("integrity_state") == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
        score -= 0.14
    if audit.get("audit_state") in (
        "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",
        "EVIDENCE_FRAGMENTED",
    ):
        score -= 0.16
    if failure.get("risk_state") in (
        "FALSE_CONFIDENCE_ESCALATION_RISK",
        "ESCALATION_WITHOUT_MATURITY_RISK",
    ):
        score -= 0.20

    return min(max(score, 0.0), 1.0)


def _gcc_cross_section_agreement(
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
) -> str:
    checks = 0
    aligned = 0

    checks += 1
    if _gcc_regime_trajectory_aligned(regime, forecast):
        aligned += 1

    checks += 1
    consensus_state = consensus.get("consensus_state", "")
    count = int(tension.get("contradiction_count", 0) or 0)
    if consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
        if count >= 2:
            aligned += 1
    elif count <= 1:
        aligned += 1

    checks += 1
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    discussable = decision.get("discussability", "NOT_DISCUSSABLE")
    if maturity < 0.25 and discussable in ("NOT_DISCUSSABLE", "INTERNAL_OBSERVATION_ONLY"):
        aligned += 1
    elif maturity >= 0.40 and discussable not in ("NOT_DISCUSSABLE",):
        aligned += 1

    checks += 1
    integrity_state = integrity.get("integrity_state", "")
    audit_state = audit.get("audit_state", "")
    if (
        integrity_state == "GOVERNANCE_OVERCONFIDENT"
        and audit_state == "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE"
    ):
        aligned += 1
    elif (
        integrity_state != "GOVERNANCE_OVERCONFIDENT"
        and audit_state != "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE"
    ):
        aligned += 1

    checks += 1
    failure_risk = failure.get("risk_state", "")
    if failure_risk in ("FALSE_CONFIDENCE_ESCALATION_RISK", "ESCALATION_WITHOUT_MATURITY_RISK"):
        if decision.get("escalation_readiness") not in ("NONE", "VERY_LOW"):
            aligned += 1
    elif failure_risk == "GOVERNANCE_FRAGMENTATION_RISK":
        if consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE"):
            aligned += 1
    elif failure_risk == "GOVERNANCE_STABLE":
        aligned += 1

    checks += 1
    if regime.get("regime") == "CONSTITUTIONAL_STRESS" and forecast.get("trajectory") in (
        "CONSTITUTIONALLY_CONSTRAINED",
        "GOVERNANCE_STALLED",
    ):
        if decision.get("readiness_state") in (
            "NOT_INSTITUTIONALLY_DISCUSSABLE",
            "OBSERVATION_ONLY",
        ):
            aligned += 1

    ratio = aligned / max(checks, 1)
    if ratio >= 0.85:
        return "VERY_HIGH"
    if ratio >= 0.70:
        return "HIGH"
    if ratio >= 0.50:
        return "MODERATE"
    if ratio >= 0.30:
        return "LOW"
    return "VERY_LOW"


def _gcc_narrative_integrity(coherence_state: str, logic_score: float) -> str:
    if coherence_state == "HIGHLY_COHERENT_GOVERNANCE":
        return "HIGHLY_COHERENT"
    if coherence_state == "LOGICALLY_CONSTRAINED_BUT_COHERENT":
        return "STRONG"
    if coherence_state == "MODERATELY_COHERENT_GOVERNANCE":
        return "PARTIAL"
    if coherence_state == "LOW_COHERENCE_GOVERNANCE":
        return "WEAK"
    if coherence_state in (
        "INTERNALLY_INCONSISTENT_GOVERNANCE",
        "FRAGMENTED_REASONING_CHAIN",
        "ESCALATION_LOGIC_MISMATCH",
    ):
        return "BROKEN" if logic_score < 0.20 else "WEAK"
    return "PARTIAL"


def _gcc_reasoning_chain_status(
    coherence_state: str,
    narrative: str,
    cross_section: str,
) -> str:
    if coherence_state == "HIGHLY_COHERENT_GOVERNANCE":
        return "HIGHLY_COHERENT"
    if coherence_state in ("INTERNALLY_INCONSISTENT_GOVERNANCE", "FRAGMENTED_REASONING_CHAIN"):
        return "BROKEN" if narrative == "BROKEN" else "FRAGMENTED"
    if coherence_state == "ESCALATION_LOGIC_MISMATCH":
        return "FRAGMENTED"
    if coherence_state == "LOGICALLY_CONSTRAINED_BUT_COHERENT":
        return "COHERENT"
    if coherence_state == "MODERATELY_COHERENT_GOVERNANCE":
        return "PARTIALLY_COHERENT"
    if cross_section in ("VERY_LOW", "LOW"):
        return "FRAGMENTED"
    return "PARTIALLY_COHERENT"


def _gcc_coherence_drivers(
    coherence_state: str,
    snap: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    audit: Dict[str, Any],
) -> List[str]:
    cls = snap["classifications"]
    drivers: List[str] = []

    if coherence_state == "LOGICALLY_CONSTRAINED_BUT_COHERENT":
        drivers.append("Constitutional lock consistently supported")
        drivers.append(f"Runtime readiness: {cls.get('readiness') or '—'}")
        drivers.append("Escalation blocked coherently")
        drivers.append(
            f"Governance maturity insufficient ({_gcc_fmt_conf(decision.get('governance_maturity_score'))})"
        )
    elif coherence_state == "FRAGMENTED_REASONING_CHAIN":
        drivers.append("Contradictions weaken institutional story")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
        drivers.append(f"Trajectory: {forecast.get('trajectory_display', '—')}")
    elif coherence_state == "ESCALATION_LOGIC_MISMATCH":
        drivers.append("Escalation posture exceeds maturity")
        drivers.append(f"Escalation readiness: {decision.get('escalation_readiness', '—')}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
        drivers.append(f"Discussability: {decision.get('discussability', '—')}")
    elif coherence_state == "INTERNALLY_INCONSISTENT_GOVERNANCE":
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(f"Trajectory: {forecast.get('trajectory_display', '—')}")
        drivers.append(f"Integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(f"Decision readiness: {decision.get('readiness_display', '—')}")
    elif coherence_state == "HIGHLY_COHERENT_GOVERNANCE":
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
        drivers.append(f"Tension: {tension.get('tension_display', '—')}")
        drivers.append("Cross-section governance signals align")
    else:
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(f"Failure alignment: {consensus.get('consensus_state', '—')}")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
        drivers.append(
            f"Evidence integrity: {_gcc_fmt_conf(audit.get('evidence_integrity_score'))}"
        )

    return drivers[:6]


def _gcc_detect_governance_coherence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    logic_score = _gcc_institutional_logic_score(
        regime, forecast, tension, consensus, integrity, decision, failure, audit
    )
    cross_section = _gcc_cross_section_agreement(
        regime, forecast, tension, consensus, integrity, decision, failure, audit
    )

    regime_key = regime.get("regime", "")
    trajectory = forecast.get("trajectory", "")
    consensus_state = consensus.get("consensus_state", "")
    integrity_state = integrity.get("integrity_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    escalation = decision.get("escalation_readiness", "NONE")
    discussability = decision.get("discussability", "NOT_DISCUSSABLE")
    readiness_state = decision.get("readiness_state", "")
    failure_risk = failure.get("risk_state", "")
    audit_state = audit.get("audit_state", "")

    def _match_escalation_mismatch() -> bool:
        return (
            failure_risk == "ESCALATION_WITHOUT_MATURITY_RISK"
            or (
                escalation not in ("NONE", "VERY_LOW")
                and (
                    discussability in ("NOT_DISCUSSABLE", "INTERNAL_OBSERVATION_ONLY")
                    or maturity < 0.25
                    or audit_state == "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE"
                )
            )
            or (
                readiness_state in ("GOVERNANCE_REVIEW_ELIGIBLE", "LIMITED_REVIEW_WORTHINESS")
                and maturity < 0.25
                and audit_state == "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE"
            )
        )

    def _match_internally_inconsistent() -> bool:
        return (
            (
                regime_key in ("CONSTITUTIONAL_STRESS", "GOVERNANCE_REGRESSION")
                and trajectory in ("GOVERNANCE_ACCELERATING", "PRE_RUNTIME_TRAJECTORY")
                and readiness_state != "NOT_INSTITUTIONALLY_DISCUSSABLE"
            )
            or (
                consensus_state in ("CONSTITUTIONAL_CONSENSUS", "STRONG_CONSENSUS")
                and integrity_state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS"
            )
            or (
                readiness_state
                in ("GOVERNANCE_REVIEW_ELIGIBLE", "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE")
                and failure_risk
                in ("FALSE_CONFIDENCE_ESCALATION_RISK", "GOVERNANCE_FRAGMENTATION_RISK")
            )
        )

    def _match_fragmented_chain() -> bool:
        return (
            consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            and contradiction_count >= 2
            and audit_state
            in (
                "EVIDENCE_FRAGMENTED",
                "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",
                "LOW_AUDITABILITY",
            )
        )

    def _match_low_coherence() -> bool:
        return logic_score < 0.35 or cross_section in ("VERY_LOW", "LOW")

    def _match_constrained_coherent() -> bool:
        return (
            (
                regime_key
                in ("CONSTITUTIONAL_STRESS", "GOVERNANCE_REGRESSION", "INSTITUTIONAL_INSTABILITY")
                or snap["constitutional_safe"] is False
                or snap["constitutional_review_required"]
            )
            and _gcc_regime_trajectory_aligned(regime, forecast)
            and readiness_state in ("NOT_INSTITUTIONALLY_DISCUSSABLE", "OBSERVATION_ONLY")
            and discussability in ("NOT_DISCUSSABLE", "INTERNAL_OBSERVATION_ONLY")
            and escalation in ("NONE", "VERY_LOW")
            and integrity_state
            in (
                "GOVERNANCE_CONFIDENCE_DORMANT_BUT_CONSISTENT",
                "GOVERNANCE_CONFIDENCE_STABLE",
            )
        )

    def _match_moderate() -> bool:
        return logic_score >= 0.30 and cross_section in ("MODERATE", "HIGH")

    def _match_highly_coherent() -> bool:
        return (
            logic_score >= 0.60
            and cross_section in ("HIGH", "VERY_HIGH")
            and contradiction_count == 0
            and consensus_state not in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            and failure_risk == "GOVERNANCE_STABLE"
            and audit_state == "HIGH_AUDITABILITY"
        )

    matchers = {
        "ESCALATION_LOGIC_MISMATCH": _match_escalation_mismatch,
        "INTERNALLY_INCONSISTENT_GOVERNANCE": _match_internally_inconsistent,
        "FRAGMENTED_REASONING_CHAIN": _match_fragmented_chain,
        "LOW_COHERENCE_GOVERNANCE": _match_low_coherence,
        "LOGICALLY_CONSTRAINED_BUT_COHERENT": _match_constrained_coherent,
        "MODERATELY_COHERENT_GOVERNANCE": _match_moderate,
        "HIGHLY_COHERENT_GOVERNANCE": _match_highly_coherent,
    }

    coherence_state = "MODERATELY_COHERENT_GOVERNANCE"
    for candidate in _GCC_COHERENCE_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            coherence_state = candidate
            break

    narrative = _gcc_narrative_integrity(coherence_state, logic_score)
    reasoning_chain = _gcc_reasoning_chain_status(coherence_state, narrative, cross_section)
    drivers = _gcc_coherence_drivers(
        coherence_state, snap, regime, forecast, tension, consensus, integrity, decision, audit
    )

    return {
        "coherence_state": coherence_state,
        "coherence_display": _GCC_COHERENCE_DISPLAY.get(
            coherence_state, coherence_state.replace("_", " ").title()
        ),
        "logic_score": logic_score,
        "narrative_integrity": narrative,
        "cross_section_agreement": cross_section,
        "drivers": drivers,
        "interpretation": _GCC_COHERENCE_INTERPRETATION.get(coherence_state, ""),
        "logic_action": _GCC_COHERENCE_ACTION.get(coherence_state, "CONTINUE_OBSERVATION"),
        "reasoning_chain_status": reasoning_chain,
    }


def _gcc_render_coherence_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
) -> Dict[str, Any]:
    coherence = _gcc_detect_governance_coherence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
    )

    st.markdown("### Governance Institutional Coherence & Internal Logic Integrity")
    st.caption(
        "Institutional sense-making — whether governance logically holds together across sections. "
        "**Read-only logic validation. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Coherence", coherence["coherence_display"])
    c2.metric("Institutional Logic Score", _gcc_fmt_conf(coherence["logic_score"]))
    c3.metric("Narrative Integrity", coherence["narrative_integrity"])
    c4.metric("Cross-Section Agreement", coherence["cross_section_agreement"])

    state = coherence["coherence_state"]
    if state in (
        "ESCALATION_LOGIC_MISMATCH",
        "INTERNALLY_INCONSISTENT_GOVERNANCE",
        "FRAGMENTED_REASONING_CHAIN",
    ):
        st.error(coherence["interpretation"])
    elif state == "LOW_COHERENCE_GOVERNANCE":
        st.warning(coherence["interpretation"])
    elif state in ("HIGHLY_COHERENT_GOVERNANCE", "LOGICALLY_CONSTRAINED_BUT_COHERENT"):
        st.success(coherence["interpretation"])
    else:
        st.info(coherence["interpretation"])

    st.markdown("**Coherence Drivers**")
    for driver in coherence["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Logic Action", coherence["logic_action"])
    a2.metric("Reasoning Chain Status", coherence["reasoning_chain_status"])

    with st.expander("Coherence analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{coherence['coherence_state']}`")
        st.markdown(f"- **Regime:** `{regime.get('regime', '—')}`")
        st.markdown(f"- **Trajectory:** `{forecast.get('trajectory', '—')}`")
        st.markdown(f"- **Auditability state:** `{audit.get('audit_state', '—')}`")

    return coherence


_GCC_STABILITY_PRIORITY: Tuple[str, ...] = (
    "GOVERNANCE_DETERIORATING",
    "REASONING_INSTABILITY_DETECTED",
    "GOVERNANCE_VOLATILITY_ELEVATED",
    "CONSTITUTIONAL_DRIFT_RISK",
    "CONFIDENCE_OSCILLATION_RISK",
    "INSTITUTIONAL_DRIFT_DETECTED",
    "GOVERNANCE_STABILITY_IMPROVING",
    "STABLE_GOVERNANCE_POSTURE",
)

_GCC_STABILITY_DISPLAY: Dict[str, str] = {
    "STABLE_GOVERNANCE_POSTURE": "Stable Governance Posture",
    "INSTITUTIONAL_DRIFT_DETECTED": "Institutional Drift Detected",
    "CONSTITUTIONAL_DRIFT_RISK": "Constitutional Drift Risk",
    "GOVERNANCE_VOLATILITY_ELEVATED": "Governance Volatility Elevated",
    "REASONING_INSTABILITY_DETECTED": "Reasoning Instability Detected",
    "CONFIDENCE_OSCILLATION_RISK": "Confidence Oscillation Risk",
    "GOVERNANCE_STABILITY_IMPROVING": "Governance Stability Improving",
    "GOVERNANCE_DETERIORATING": "Governance Deteriorating",
}

_GCC_STABILITY_INTERPRETATION: Dict[str, str] = {
    "STABLE_GOVERNANCE_POSTURE": (
        "Governance posture is stable but restrictive. Runtime governance remains blocked "
        "consistently, and no durable institutional shift has occurred."
    ),
    "INSTITUTIONAL_DRIFT_DETECTED": (
        "Governance regime, trajectory, or readiness posture is shifting over time. "
        "Institutional movement is present but not yet clearly stabilized."
    ),
    "CONSTITUTIONAL_DRIFT_RISK": (
        "Constitutional constraints remain the dominant source of governance instability. "
        "Operators should maintain the constitutional lock until pressure subsides."
    ),
    "GOVERNANCE_VOLATILITY_ELEVATED": (
        "Governance posture changes frequently. Confidence direction and transition history "
        "indicate elevated institutional volatility."
    ),
    "REASONING_INSTABILITY_DETECTED": (
        "Governance reasoning remains unstable. Fragmented coherence, weak auditability, "
        "and discounted confidence reduce institutional durability."
    ),
    "CONFIDENCE_OSCILLATION_RISK": (
        "Confidence direction is unstable across recorded history. Trustworthiness is difficult "
        "to interpret while oscillation persists."
    ),
    "GOVERNANCE_STABILITY_IMPROVING": (
        "Governance posture is becoming more stable. Contradictions and coherence signals "
        "suggest institutional durability is improving."
    ),
    "GOVERNANCE_DETERIORATING": (
        "Governance is deteriorating. Confidence is weakening, contradictions are elevated, "
        "and coherence or auditability is degrading."
    ),
}

_GCC_STABILITY_ACTION: Dict[str, str] = {
    "STABLE_GOVERNANCE_POSTURE": "CONTINUE_OBSERVATION",
    "INSTITUTIONAL_DRIFT_DETECTED": "INVESTIGATE_DRIFT",
    "CONSTITUTIONAL_DRIFT_RISK": "MAINTAIN_CONSTITUTIONAL_LOCK",
    "GOVERNANCE_VOLATILITY_ELEVATED": "REDUCE_GOVERNANCE_VOLATILITY",
    "REASONING_INSTABILITY_DETECTED": "REDUCE_REASONING_INSTABILITY",
    "CONFIDENCE_OSCILLATION_RISK": "STRENGTHEN_STABILITY_EVIDENCE",
    "GOVERNANCE_STABILITY_IMPROVING": "CONTINUE_OBSERVATION",
    "GOVERNANCE_DETERIORATING": "BLOCK_ESCALATION_UNTIL_STABLE",
}

_GCC_STABILITY_CONTAINMENT: Dict[str, str] = {
    "STABLE_GOVERNANCE_POSTURE": "Continue monitoring transition history",
    "INSTITUTIONAL_DRIFT_DETECTED": "Delay institutional discussion until posture stabilizes",
    "CONSTITUTIONAL_DRIFT_RISK": "Preserve constitutional lock",
    "GOVERNANCE_VOLATILITY_ELEVATED": "Stabilize confidence integrity",
    "REASONING_INSTABILITY_DETECTED": "Reduce reasoning fragmentation",
    "CONFIDENCE_OSCILLATION_RISK": "Strengthen stability evidence before escalation review",
    "GOVERNANCE_STABILITY_IMPROVING": "Maintain observation only",
    "GOVERNANCE_DETERIORATING": "Delay institutional discussion until posture stabilizes",
}


def _gcc_stability_score(
    hist: Dict[str, Any],
    tension: Dict[str, Any],
    coherence: Dict[str, Any],
    audit: Dict[str, Any],
    failure: Dict[str, Any],
) -> float:
    direction = hist.get("confidence_direction", "stable")
    stability_text = str(hist.get("posture_stability", ""))
    transition_n = len(hist.get("transitions") or [])
    count = int(tension.get("contradiction_count", 0) or 0)

    score = 0.30
    if "Stable" in stability_text:
        score += 0.22
    elif "Limited" in stability_text:
        score += 0.10
    if direction in ("stable", "dormant"):
        score += 0.18
    elif direction == "improving":
        score += 0.12
    elif direction == "mixed":
        score -= 0.10
    elif direction == "deteriorating":
        score -= 0.18

    score += 0.04 * max(0, 3 - min(transition_n, 3))
    score -= 0.06 * min(count, 4)
    score += 0.10 * float(coherence.get("logic_score", 0.0) or 0.0)
    score += 0.08 * float(audit.get("evidence_integrity_score", 0.0) or 0.0)

    if coherence.get("reasoning_chain_status") in ("BROKEN", "FRAGMENTED"):
        score -= 0.20
    if failure.get("risk_severity") == "CRITICAL":
        score -= 0.15
    elif failure.get("risk_severity") == "HIGH":
        score -= 0.08

    return min(max(score, 0.0), 1.0)


def _gcc_drift_severity(
    hist: Dict[str, Any],
    snap: Dict[str, Any],
    tension: Dict[str, Any],
    failure: Dict[str, Any],
    stability_score: float,
) -> str:
    transition_n = len(hist.get("transitions") or [])
    direction = hist.get("confidence_direction", "stable")
    count = int(tension.get("contradiction_count", 0) or 0)
    fragility = float(failure.get("fragility_score", 0.0) or 0.0)
    risk_sev = failure.get("risk_severity", "LOW")

    if (direction == "deteriorating" and count >= 2 and risk_sev in ("CRITICAL", "HIGH")) or (
        fragility >= 0.85 and transition_n >= 2
    ):
        return "CRITICAL"
    if (
        snap["constitutional_safe"] is False
        and snap["constitutional_review_required"]
        and direction in ("deteriorating", "mixed")
    ):
        return "HIGH"
    if transition_n >= 3 or direction == "mixed" or count >= 2:
        return "MODERATE" if stability_score >= 0.35 else "HIGH"
    if transition_n >= 1 or "Limited" in str(hist.get("posture_stability", "")):
        return "LOW"
    if stability_score >= 0.55 and transition_n == 0:
        return "NONE"
    return "LOW"


def _gcc_volatility_level(
    hist: Dict[str, Any],
    tension: Dict[str, Any],
    coherence: Dict[str, Any],
) -> str:
    transition_n = len(hist.get("transitions") or [])
    direction = hist.get("confidence_direction", "stable")
    stability_text = str(hist.get("posture_stability", ""))
    count = int(tension.get("contradiction_count", 0) or 0)

    if transition_n >= 4 or (direction == "mixed" and "Frequent" in stability_text):
        return "EXTREME"
    if transition_n >= 2 or direction == "mixed" or count >= 3:
        return "HIGH"
    if (
        transition_n >= 1
        or count >= 2
        or coherence.get("coherence_state") == "FRAGMENTED_REASONING_CHAIN"
    ):
        return "MODERATE"
    if "Limited" in stability_text or direction == "deteriorating":
        return "LOW"
    return "NONE"


def _gcc_stability_drivers(
    stability_state: str,
    snap: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    coherence: Dict[str, Any],
    audit: Dict[str, Any],
    failure: Dict[str, Any],
) -> List[str]:
    cls = snap["classifications"]
    drivers: List[str] = []

    if stability_state == "STABLE_GOVERNANCE_POSTURE":
        drivers.append("Runtime governance remains consistently constrained")
        drivers.append(f"Human escalation: {cls.get('escalation') or '—'}")
        drivers.append("Institutional posture has not materially shifted")
        drivers.append("Runtime mutation lock remains preserved")
    elif stability_state == "REASONING_INSTABILITY_DETECTED":
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
        drivers.append(f"Integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(f"Reasoning chain: {coherence.get('reasoning_chain_status', '—')}")
    elif stability_state == "CONSTITUTIONAL_DRIFT_RISK":
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(
            f"Constitutional safe: {'Yes' if snap['constitutional_safe'] is True else 'No'}"
        )
        drivers.append(
            f"Constitutional review required: {'Yes' if snap['constitutional_review_required'] else 'No'}"
        )
        drivers.append(f"Failure risk: {failure.get('risk_display', '—')}")
    elif stability_state == "GOVERNANCE_DETERIORATING":
        drivers.append(f"Confidence direction: {hist.get('confidence_direction', '—')}")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Fragility score: {_gcc_fmt_conf(failure.get('fragility_score'))}")
    elif stability_state == "GOVERNANCE_VOLATILITY_ELEVATED":
        drivers.append(hist.get("posture_stability", "Posture shifts detected"))
        drivers.append(f"Transition count: {len(hist.get('transitions') or [])}")
        drivers.append(f"Confidence direction: {hist.get('confidence_direction', '—')}")
        drivers.append(f"Tension: {tension.get('tension_display', '—')}")
    elif stability_state == "CONFIDENCE_OSCILLATION_RISK":
        drivers.append(f"Confidence direction: {hist.get('confidence_direction', '—')}")
        drivers.append(f"Transition count: {len(hist.get('transitions') or [])}")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(hist.get("posture_stability", "Posture shifts detected"))
    else:
        drivers.append(hist.get("posture_stability", "Posture stability assessed"))
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(f"Momentum: {hist.get('institutional_momentum', '—')}")
        drivers.append(f"Transition count: {len(hist.get('transitions') or [])}")

    return drivers[:6]


def _gcc_detect_governance_stability(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    stability_score = _gcc_stability_score(hist, tension, coherence, audit, failure)
    drift_severity = _gcc_drift_severity(hist, snap, tension, failure, stability_score)
    volatility = _gcc_volatility_level(hist, tension, coherence)

    direction = hist.get("confidence_direction", "stable")
    stability_text = str(hist.get("posture_stability", ""))
    transition_n = len(hist.get("transitions") or [])
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    failure_risk = failure.get("risk_state", "")
    coherence_state = coherence.get("coherence_state", "")
    reasoning = coherence.get("reasoning_chain_status", "")
    audit_state = audit.get("audit_state", "")

    def _match_deteriorating() -> bool:
        return (
            direction == "deteriorating"
            or (
                direction == "mixed"
                and contradiction_count >= 2
                and coherence_state in ("FRAGMENTED_REASONING_CHAIN", "LOW_COHERENCE_GOVERNANCE")
            )
            or (
                integrity.get("integrity_state")
                == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS"
                and direction != "improving"
                and float(coherence.get("logic_score", 0.0) or 0.0) < 0.25
            )
        )

    def _match_reasoning_instability() -> bool:
        return (
            reasoning in ("BROKEN", "FRAGMENTED")
            or coherence_state
            in (
                "FRAGMENTED_REASONING_CHAIN",
                "INTERNALLY_INCONSISTENT_GOVERNANCE",
            )
            or (
                audit_state
                in ("CONFIDENCE_UNSUPPORTED_BY_EVIDENCE", "EVIDENCE_FRAGMENTED", "LOW_AUDITABILITY")
                and coherence_state != "LOGICALLY_CONSTRAINED_BUT_COHERENT"
            )
        )

    def _match_volatility_elevated() -> bool:
        return (
            "Frequent" in stability_text
            or transition_n >= 3
            or (direction == "mixed" and transition_n >= 1)
        )

    def _match_constitutional_drift() -> bool:
        return failure_risk == "CONSTITUTIONAL_DRIFT_RISK" or (
            snap["constitutional_safe"] is False
            and (
                regime.get("regime") == "CONSTITUTIONAL_STRESS"
                or snap["constitutional_review_required"]
            )
        )

    def _match_confidence_oscillation() -> bool:
        return direction == "mixed" and (
            transition_n >= 1 or "Limited" in stability_text or "Frequent" in stability_text
        )

    def _match_institutional_drift() -> bool:
        return (
            transition_n >= 1
            and direction not in ("stable", "dormant")
            and not _match_volatility_elevated()
        ) or (
            hist.get("posture_trend")
            not in ("Stable Restrictive Posture", "Stable Governance Posture")
            and transition_n >= 1
            and direction in ("improving", "mixed")
        )

    def _match_improving() -> bool:
        return (
            direction == "improving"
            and contradiction_count <= 1
            and reasoning not in ("BROKEN", "FRAGMENTED")
            and float(coherence.get("logic_score", 0.0) or 0.0) >= 0.30
        )

    def _match_stable() -> bool:
        return (
            direction in ("stable", "dormant")
            and "Frequent" not in stability_text
            and transition_n <= 1
            and (
                coherence_state
                in (
                    "LOGICALLY_CONSTRAINED_BUT_COHERENT",
                    "MODERATELY_COHERENT_GOVERNANCE",
                    "HIGHLY_COHERENT_GOVERNANCE",
                )
                or (
                    snap["max_conf"] <= 0.01
                    and decision.get("discussability")
                    in ("NOT_DISCUSSABLE", "INTERNAL_OBSERVATION_ONLY")
                )
            )
        )

    matchers = {
        "GOVERNANCE_DETERIORATING": _match_deteriorating,
        "REASONING_INSTABILITY_DETECTED": _match_reasoning_instability,
        "GOVERNANCE_VOLATILITY_ELEVATED": _match_volatility_elevated,
        "CONSTITUTIONAL_DRIFT_RISK": _match_constitutional_drift,
        "CONFIDENCE_OSCILLATION_RISK": _match_confidence_oscillation,
        "INSTITUTIONAL_DRIFT_DETECTED": _match_institutional_drift,
        "GOVERNANCE_STABILITY_IMPROVING": _match_improving,
        "STABLE_GOVERNANCE_POSTURE": _match_stable,
    }

    stability_state = "INSTITUTIONAL_DRIFT_DETECTED"
    for candidate in _GCC_STABILITY_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            stability_state = candidate
            break

    drivers = _gcc_stability_drivers(
        stability_state, snap, hist, regime, tension, integrity, coherence, audit, failure
    )

    return {
        "stability_state": stability_state,
        "stability_display": _GCC_STABILITY_DISPLAY.get(
            stability_state, stability_state.replace("_", " ").title()
        ),
        "drift_severity": drift_severity,
        "stability_score": stability_score,
        "volatility_level": volatility,
        "drivers": drivers,
        "interpretation": _GCC_STABILITY_INTERPRETATION.get(stability_state, ""),
        "stability_action": _GCC_STABILITY_ACTION.get(stability_state, "CONTINUE_OBSERVATION"),
        "containment_strategy": _GCC_STABILITY_CONTAINMENT.get(
            stability_state, "Maintain observation only"
        ),
    }


def _gcc_render_stability_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
) -> Dict[str, Any]:
    stability = _gcc_detect_governance_stability(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
    )

    st.markdown("### Governance Institutional Stability & Drift Intelligence")
    st.caption(
        "Institutional durability analysis — whether governance posture is stable, drifting, or volatile. "
        "**Read-only stability view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stability State", stability["stability_display"])
    c2.metric("Drift Severity", stability["drift_severity"])
    c3.metric("Stability Score", _gcc_fmt_conf(stability["stability_score"]))
    c4.metric("Volatility Level", stability["volatility_level"])

    state = stability["stability_state"]
    drift = stability["drift_severity"]
    if state == "GOVERNANCE_DETERIORATING" or drift == "CRITICAL":
        st.error(stability["interpretation"])
    elif state in (
        "REASONING_INSTABILITY_DETECTED",
        "GOVERNANCE_VOLATILITY_ELEVATED",
        "CONSTITUTIONAL_DRIFT_RISK",
    ) or drift in ("HIGH", "MODERATE"):
        st.warning(stability["interpretation"])
    elif state in ("STABLE_GOVERNANCE_POSTURE", "GOVERNANCE_STABILITY_IMPROVING"):
        st.success(stability["interpretation"])
    else:
        st.info(stability["interpretation"])

    st.markdown("**Stability / Drift Drivers**")
    for driver in stability["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Stability Action", stability["stability_action"])
    a2.metric("Drift Containment Strategy", stability["containment_strategy"])

    with st.expander("Stability analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{stability['stability_state']}`")
        st.markdown(f"- **Confidence direction:** `{hist.get('confidence_direction', '—')}`")
        st.markdown(f"- **Posture stability:** {hist.get('posture_stability', '—')}")
        st.markdown(f"- **Transition count:** `{len(hist.get('transitions') or [])}`")

    return stability


_GCC_RESILIENCE_PRIORITY: Tuple[str, ...] = (
    "GOVERNANCE_RECOVERY_BLOCKED",
    "CONSTITUTIONAL_RECOVERY_UNLIKELY",
    "LOW_GOVERNANCE_RESILIENCE",
    "RECOVERABLE_FRAGMENTATION",
    "REVERSIBLE_GOVERNANCE_DETERIORATION",
    "GOVERNANCE_SELF_STABILIZATION_EMERGING",
    "MODERATE_GOVERNANCE_RESILIENCE",
    "HIGH_GOVERNANCE_RESILIENCE",
)

_GCC_RESILIENCE_DISPLAY: Dict[str, str] = {
    "HIGH_GOVERNANCE_RESILIENCE": "High Governance Resilience",
    "MODERATE_GOVERNANCE_RESILIENCE": "Moderate Governance Resilience",
    "LOW_GOVERNANCE_RESILIENCE": "Low Governance Resilience",
    "GOVERNANCE_RECOVERY_BLOCKED": "Governance Recovery Blocked",
    "RECOVERABLE_FRAGMENTATION": "Recoverable Fragmentation",
    "CONSTITUTIONAL_RECOVERY_UNLIKELY": "Constitutional Recovery Unlikely",
    "GOVERNANCE_SELF_STABILIZATION_EMERGING": "Governance Self-Stabilization Emerging",
    "REVERSIBLE_GOVERNANCE_DETERIORATION": "Reversible Governance Deterioration",
}

_GCC_RESILIENCE_INTERPRETATION: Dict[str, str] = {
    "HIGH_GOVERNANCE_RESILIENCE": (
        "Governance resilience is strong. Institutional posture is stable, contradictions are "
        "low, and recovery capability appears durable."
    ),
    "MODERATE_GOVERNANCE_RESILIENCE": (
        "Governance is partially recoverable. Instability is manageable and institutional "
        "improvement remains plausible with continued observation."
    ),
    "LOW_GOVERNANCE_RESILIENCE": (
        "Governance resilience remains weak. Fragmented coherence, poor trustworthiness, and "
        "institutional instability limit recovery confidence."
    ),
    "GOVERNANCE_RECOVERY_BLOCKED": (
        "Governance recovery remains blocked by constitutional pressure, weak evidence integrity, "
        "and fragmented institutional reasoning."
    ),
    "RECOVERABLE_FRAGMENTATION": (
        "Governance fragmentation is present but appears recoverable. Contradictions remain "
        "manageable and coherence repair is plausible."
    ),
    "CONSTITUTIONAL_RECOVERY_UNLIKELY": (
        "Constitutional constraints dominate recovery outlook. Governance remains constrained "
        "until constitutional pressure subsides."
    ),
    "GOVERNANCE_SELF_STABILIZATION_EMERGING": (
        "Governance self-stabilization signals are emerging. Coherence and stability indicators "
        "suggest institutional recovery may be underway."
    ),
    "REVERSIBLE_GOVERNANCE_DETERIORATION": (
        "Governance deterioration appears reversible. Institutional signals suggest stabilization "
        "may become possible if contradictions decline."
    ),
}

_GCC_RESILIENCE_ACTION: Dict[str, str] = {
    "HIGH_GOVERNANCE_RESILIENCE": "CONTINUE_OBSERVATION",
    "MODERATE_GOVERNANCE_RESILIENCE": "MONITOR_RECOVERY_SIGNALS",
    "LOW_GOVERNANCE_RESILIENCE": "IMPROVE_GOVERNANCE_COHERENCE",
    "GOVERNANCE_RECOVERY_BLOCKED": "PRESERVE_CONSTITUTIONAL_LOCK",
    "RECOVERABLE_FRAGMENTATION": "REDUCE_FRAGMENTATION",
    "CONSTITUTIONAL_RECOVERY_UNLIKELY": "PRESERVE_CONSTITUTIONAL_LOCK",
    "GOVERNANCE_SELF_STABILIZATION_EMERGING": "CONTINUE_OBSERVATION",
    "REVERSIBLE_GOVERNANCE_DETERIORATION": "MONITOR_RECOVERY_SIGNALS",
}

_GCC_RESILIENCE_PATHWAY: Dict[str, str] = {
    "HIGH_GOVERNANCE_RESILIENCE": "Stabilize institutional posture",
    "MODERATE_GOVERNANCE_RESILIENCE": "Monitor recovery signals",
    "LOW_GOVERNANCE_RESILIENCE": "Improve governance coherence",
    "GOVERNANCE_RECOVERY_BLOCKED": "Delay escalation until stability improves",
    "RECOVERABLE_FRAGMENTATION": "Reduce contradictions",
    "CONSTITUTIONAL_RECOVERY_UNLIKELY": "Delay escalation until stability improves",
    "GOVERNANCE_SELF_STABILIZATION_EMERGING": "Stabilize institutional posture",
    "REVERSIBLE_GOVERNANCE_DETERIORATION": "Strengthen confidence integrity",
}


def _gcc_recovery_probability(
    resilience_state: str,
    stability: Dict[str, Any],
    coherence: Dict[str, Any],
    audit: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    if resilience_state in ("GOVERNANCE_RECOVERY_BLOCKED", "CONSTITUTIONAL_RECOVERY_UNLIKELY"):
        return "VERY_LOW"
    if resilience_state == "LOW_GOVERNANCE_RESILIENCE":
        return "LOW"
    if resilience_state in ("RECOVERABLE_FRAGMENTATION", "REVERSIBLE_GOVERNANCE_DETERIORATION"):
        return "MODERATE"
    if resilience_state in (
        "GOVERNANCE_SELF_STABILIZATION_EMERGING",
        "MODERATE_GOVERNANCE_RESILIENCE",
    ):
        return "HIGH"
    if resilience_state == "HIGH_GOVERNANCE_RESILIENCE":
        return "VERY_HIGH"

    score = (
        float(stability.get("stability_score", 0.0) or 0.0) * 0.35
        + float(coherence.get("logic_score", 0.0) or 0.0) * 0.25
        + float(audit.get("evidence_integrity_score", 0.0) or 0.0) * 0.20
        + float(decision.get("governance_maturity_score", 0.0) or 0.0) * 0.20
    )
    if score < 0.20:
        return "VERY_LOW"
    if score < 0.35:
        return "LOW"
    if score < 0.55:
        return "MODERATE"
    if score < 0.75:
        return "HIGH"
    return "VERY_HIGH"


def _gcc_recovery_severity(
    resilience_state: str,
    failure: Dict[str, Any],
    coherence: Dict[str, Any],
) -> str:
    if resilience_state == "GOVERNANCE_RECOVERY_BLOCKED":
        return "CRITICAL"
    if resilience_state in ("CONSTITUTIONAL_RECOVERY_UNLIKELY", "LOW_GOVERNANCE_RESILIENCE"):
        return "SEVERE"
    if resilience_state == "REVERSIBLE_GOVERNANCE_DETERIORATION":
        return "MATERIAL"
    if resilience_state == "RECOVERABLE_FRAGMENTATION":
        return "MODERATE"
    if (
        failure.get("risk_severity") == "CRITICAL"
        and coherence.get("reasoning_chain_status") == "BROKEN"
    ):
        return "CRITICAL"
    if resilience_state in (
        "MODERATE_GOVERNANCE_RESILIENCE",
        "GOVERNANCE_SELF_STABILIZATION_EMERGING",
    ):
        return "MINOR"
    return "MODERATE"


def _gcc_self_stabilization_potential(
    hist: Dict[str, Any],
    stability: Dict[str, Any],
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    resilience_state: str,
) -> str:
    if resilience_state in ("GOVERNANCE_RECOVERY_BLOCKED", "CONSTITUTIONAL_RECOVERY_UNLIKELY"):
        return "NONE"
    direction = hist.get("confidence_direction", "stable")
    count = int(tension.get("contradiction_count", 0) or 0)

    if resilience_state == "GOVERNANCE_SELF_STABILIZATION_EMERGING":
        return "HIGH"
    if resilience_state == "HIGH_GOVERNANCE_RESILIENCE":
        return "HIGH"
    if direction == "improving" and count <= 1:
        return "MODERATE"
    if resilience_state in ("RECOVERABLE_FRAGMENTATION", "REVERSIBLE_GOVERNANCE_DETERIORATION"):
        return "MODERATE"
    if float(stability.get("stability_score", 0.0) or 0.0) >= 0.50 and count <= 1:
        return "MODERATE"
    if resilience_state == "LOW_GOVERNANCE_RESILIENCE":
        return "LOW"
    return "LOW"


def _gcc_recovery_drivers(
    resilience_state: str,
    snap: Dict[str, Any],
    regime: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    coherence: Dict[str, Any],
    audit: Dict[str, Any],
    stability: Dict[str, Any],
    decision: Dict[str, Any],
    hist: Dict[str, Any],
) -> List[str]:
    drivers: List[str] = []

    if resilience_state == "GOVERNANCE_RECOVERY_BLOCKED":
        drivers.append(f"Constitutional stress: {regime.get('regime_display', '—')}")
        drivers.append(f"Confidence integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(
            f"Institutional maturity insufficient ({_gcc_fmt_conf(decision.get('governance_maturity_score'))})"
        )
    elif resilience_state == "RECOVERABLE_FRAGMENTATION":
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)} (manageable)")
        drivers.append(f"Stability score: {_gcc_fmt_conf(stability.get('stability_score'))}")
        drivers.append("Coherence repair plausible")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
    elif resilience_state == "GOVERNANCE_SELF_STABILIZATION_EMERGING":
        drivers.append(f"Consensus: {coherence.get('cross_section_agreement', '—')}")
        drivers.append(f"Drift severity: {stability.get('drift_severity', '—')}")
        drivers.append(f"Confidence direction: {hist.get('confidence_direction', '—')}")
        drivers.append(f"Stability: {stability.get('stability_display', '—')}")
    elif resilience_state == "CONSTITUTIONAL_RECOVERY_UNLIKELY":
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(
            f"Constitutional safe: {'Yes' if snap['constitutional_safe'] is True else 'No'}"
        )
        drivers.append(f"Stability state: {stability.get('stability_display', '—')}")
        drivers.append("Constitutional posture dominates recovery outlook")
    elif resilience_state == "REVERSIBLE_GOVERNANCE_DETERIORATION":
        drivers.append(f"Stability state: {stability.get('stability_display', '—')}")
        drivers.append(f"Confidence direction: {hist.get('confidence_direction', '—')}")
        drivers.append("Recovery signals partially visible")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
    else:
        drivers.append(f"Stability: {stability.get('stability_display', '—')}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Fragility: {_gcc_fmt_conf(stability.get('stability_score'))}")

    return drivers[:6]


def _gcc_detect_governance_resilience(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    stability: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )

    stability_state = stability.get("stability_state", "")
    coherence_state = coherence.get("coherence_state", "")
    reasoning = coherence.get("reasoning_chain_status", "")
    audit_state = audit.get("audit_state", "")
    failure_risk = failure.get("risk_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    direction = hist.get("confidence_direction", "stable")
    stability_score = float(stability.get("stability_score", 0.0) or 0.0)
    logic_score = float(coherence.get("logic_score", 0.0) or 0.0)
    evidence_score = float(audit.get("evidence_integrity_score", 0.0) or 0.0)
    drift_sev = stability.get("drift_severity", "LOW")

    def _match_recovery_blocked() -> bool:
        return (
            stability_state in ("GOVERNANCE_DETERIORATING", "REASONING_INSTABILITY_DETECTED")
            and (
                reasoning in ("BROKEN", "FRAGMENTED")
                or audit_state
                in (
                    "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",
                    "EVIDENCE_FRAGMENTED",
                    "LOW_AUDITABILITY",
                )
            )
            and (
                failure.get("risk_severity") in ("CRITICAL", "HIGH")
                or drift_sev in ("CRITICAL", "HIGH")
            )
        )

    def _match_constitutional_unlikely() -> bool:
        return (
            stability_state == "CONSTITUTIONAL_DRIFT_RISK"
            or failure_risk == "CONSTITUTIONAL_DRIFT_RISK"
            or (
                snap["constitutional_safe"] is False
                and regime.get("regime") == "CONSTITUTIONAL_STRESS"
                and logic_score < 0.35
            )
        )

    def _match_low_resilience() -> bool:
        return (
            logic_score < 0.25
            or stability_score < 0.30
            or evidence_score < 0.15
            or (
                coherence_state in ("FRAGMENTED_REASONING_CHAIN", "LOW_COHERENCE_GOVERNANCE")
                and float(failure.get("fragility_score", 0.0) or 0.0) >= 0.70
            )
        )

    def _match_recoverable_fragmentation() -> bool:
        return (
            consensus.get("consensus_state") in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            and contradiction_count <= 2
            and reasoning != "BROKEN"
            and not _match_recovery_blocked()
        )

    def _match_reversible_deterioration() -> bool:
        return (
            stability_state == "GOVERNANCE_DETERIORATING"
            and direction in ("improving", "stable", "mixed")
            and contradiction_count <= 2
            and logic_score >= 0.15
        )

    def _match_self_stabilization() -> bool:
        return stability_state == "GOVERNANCE_STABILITY_IMPROVING" or (
            direction == "improving"
            and contradiction_count <= 1
            and reasoning not in ("BROKEN", "FRAGMENTED")
        )

    def _match_moderate_resilience() -> bool:
        return (
            stability_score >= 0.30
            and logic_score >= 0.25
            and evidence_score >= 0.15
            and drift_sev in ("NONE", "LOW", "MODERATE")
        )

    def _match_high_resilience() -> bool:
        return (
            stability_state == "STABLE_GOVERNANCE_POSTURE"
            and stability_score >= 0.55
            and contradiction_count == 0
            and coherence_state
            in (
                "HIGHLY_COHERENT_GOVERNANCE",
                "MODERATELY_COHERENT_GOVERNANCE",
                "LOGICALLY_CONSTRAINED_BUT_COHERENT",
            )
            and failure_risk == "GOVERNANCE_STABLE"
        )

    matchers = {
        "GOVERNANCE_RECOVERY_BLOCKED": _match_recovery_blocked,
        "CONSTITUTIONAL_RECOVERY_UNLIKELY": _match_constitutional_unlikely,
        "LOW_GOVERNANCE_RESILIENCE": _match_low_resilience,
        "RECOVERABLE_FRAGMENTATION": _match_recoverable_fragmentation,
        "REVERSIBLE_GOVERNANCE_DETERIORATION": _match_reversible_deterioration,
        "GOVERNANCE_SELF_STABILIZATION_EMERGING": _match_self_stabilization,
        "MODERATE_GOVERNANCE_RESILIENCE": _match_moderate_resilience,
        "HIGH_GOVERNANCE_RESILIENCE": _match_high_resilience,
    }

    resilience_state = "MODERATE_GOVERNANCE_RESILIENCE"
    for candidate in _GCC_RESILIENCE_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            resilience_state = candidate
            break

    recovery_prob = _gcc_recovery_probability(
        resilience_state, stability, coherence, audit, decision
    )
    recovery_severity = _gcc_recovery_severity(resilience_state, failure, coherence)
    self_stab = _gcc_self_stabilization_potential(
        hist, stability, coherence, tension, resilience_state
    )
    drivers = _gcc_recovery_drivers(
        resilience_state,
        snap,
        regime,
        tension,
        integrity,
        coherence,
        audit,
        stability,
        decision,
        hist,
    )

    return {
        "resilience_state": resilience_state,
        "resilience_display": _GCC_RESILIENCE_DISPLAY.get(
            resilience_state, resilience_state.replace("_", " ").title()
        ),
        "recovery_probability": recovery_prob,
        "recovery_severity": recovery_severity,
        "self_stabilization_potential": self_stab,
        "drivers": drivers,
        "interpretation": _GCC_RESILIENCE_INTERPRETATION.get(resilience_state, ""),
        "recovery_action": _GCC_RESILIENCE_ACTION.get(resilience_state, "CONTINUE_OBSERVATION"),
        "recovery_pathway": _GCC_RESILIENCE_PATHWAY.get(
            resilience_state, "Monitor recovery signals"
        ),
    }


def _gcc_render_resilience_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    stability: Dict[str, Any],
) -> Dict[str, Any]:
    resilience = _gcc_detect_governance_resilience(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
    )

    st.markdown("### Governance Institutional Resilience & Recovery Intelligence")
    st.caption(
        "Recoverability analysis — whether governance can self-stabilize and what recovery path exists. "
        "**Read-only resilience view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Resilience", resilience["resilience_display"])
    c2.metric("Recovery Probability", resilience["recovery_probability"])
    c3.metric("Recovery Severity", resilience["recovery_severity"])
    c4.metric("Self-Stabilization Potential", resilience["self_stabilization_potential"])

    state = resilience["resilience_state"]
    severity = resilience["recovery_severity"]
    if state == "GOVERNANCE_RECOVERY_BLOCKED" or severity == "CRITICAL":
        st.error(resilience["interpretation"])
    elif state in (
        "CONSTITUTIONAL_RECOVERY_UNLIKELY",
        "LOW_GOVERNANCE_RESILIENCE",
    ) or severity in ("SEVERE", "MATERIAL"):
        st.warning(resilience["interpretation"])
    elif state in ("HIGH_GOVERNANCE_RESILIENCE", "GOVERNANCE_SELF_STABILIZATION_EMERGING"):
        st.success(resilience["interpretation"])
    else:
        st.info(resilience["interpretation"])

    st.markdown("**Recovery Drivers**")
    for driver in resilience["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Recovery Action", resilience["recovery_action"])
    a2.metric("Recovery Pathway", resilience["recovery_pathway"])

    with st.expander("Resilience analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{resilience['resilience_state']}`")
        st.markdown(f"- **Stability state:** `{stability.get('stability_state', '—')}`")
        st.markdown(f"- **Coherence state:** `{coherence.get('coherence_state', '—')}`")
        st.markdown(f"- **Drift severity:** `{stability.get('drift_severity', '—')}`")

    return resilience


_GCC_IMPROVEMENT_PRIORITY: Tuple[str, ...] = (
    "IMPROVEMENT_TRAJECTORY_BLOCKED",
    "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP",
    "GOVERNANCE_LEARNING_STALLED",
    "INSTITUTIONAL_IMPROVEMENT_POSSIBLE",
    "GOVERNANCE_ADAPTATION_EMERGING",
    "CONTRADICTIONS_DECLINING",
    "GOVERNANCE_MATURITY_IMPROVING",
    "GOVERNANCE_EVOLUTION_STRENGTHENING",
)

_GCC_IMPROVEMENT_DISPLAY: Dict[str, str] = {
    "GOVERNANCE_LEARNING_STALLED": "Governance Learning Stalled",
    "INSTITUTIONAL_IMPROVEMENT_POSSIBLE": "Institutional Improvement Possible",
    "GOVERNANCE_ADAPTATION_EMERGING": "Governance Adaptation Emerging",
    "GOVERNANCE_MATURITY_IMPROVING": "Governance Maturity Improving",
    "IMPROVEMENT_TRAJECTORY_BLOCKED": "Improvement Trajectory Blocked",
    "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP": "Governance Trapped in Low-Quality Loop",
    "CONTRADICTIONS_DECLINING": "Contradictions Declining",
    "GOVERNANCE_EVOLUTION_STRENGTHENING": "Governance Evolution Strengthening",
}

_GCC_IMPROVEMENT_INTERPRETATION: Dict[str, str] = {
    "GOVERNANCE_LEARNING_STALLED": (
        "Governance institutional learning remains limited. Persistent contradictions, weak "
        "coherence, and poor confidence integrity reduce evidence of meaningful improvement."
    ),
    "INSTITUTIONAL_IMPROVEMENT_POSSIBLE": (
        "Governance remains weak but improvement is plausible. Stabilization signals exist and "
        "contradictions appear manageable with continued institutional observation."
    ),
    "GOVERNANCE_ADAPTATION_EMERGING": (
        "Governance adaptation signals are emerging. Coherence and trustworthiness indicators "
        "suggest constructive institutional adjustment may be underway."
    ),
    "GOVERNANCE_MATURITY_IMPROVING": (
        "Governance maturity appears to be progressing. Stability and readiness trajectory "
        "signals suggest institutional strengthening over time."
    ),
    "IMPROVEMENT_TRAJECTORY_BLOCKED": (
        "Governance improvement remains constrained by constitutional pressure and institutional "
        "instability. Recovery remains insufficient to support material learning."
    ),
    "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP": (
        "Governance appears trapped in a low-quality loop. Repeated contradictions and stagnant "
        "maturity limit evidence of constructive institutional progression."
    ),
    "CONTRADICTIONS_DECLINING": (
        "Governance is becoming cleaner institutionally. Fragmentation is reducing and coherence "
        "indicators suggest improving institutional quality."
    ),
    "GOVERNANCE_EVOLUTION_STRENGTHENING": (
        "Governance shows signs of institutional evolution. Stability, coherence, trustworthiness, "
        "and maturity are improving together."
    ),
}

_GCC_IMPROVEMENT_ACTION: Dict[str, str] = {
    "GOVERNANCE_LEARNING_STALLED": "IMPROVE_GOVERNANCE_COHERENCE",
    "INSTITUTIONAL_IMPROVEMENT_POSSIBLE": "MONITOR_IMPROVEMENT_SIGNALS",
    "GOVERNANCE_ADAPTATION_EMERGING": "CONTINUE_OBSERVATION",
    "GOVERNANCE_MATURITY_IMPROVING": "STRENGTHEN_GOVERNANCE_MATURITY",
    "IMPROVEMENT_TRAJECTORY_BLOCKED": "PRESERVE_CONSTITUTIONAL_LOCK",
    "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP": "REDUCE_CONTRADICTIONS",
    "CONTRADICTIONS_DECLINING": "CONTINUE_OBSERVATION",
    "GOVERNANCE_EVOLUTION_STRENGTHENING": "CONTINUE_OBSERVATION",
}

_GCC_IMPROVEMENT_PATHWAY: Dict[str, str] = {
    "GOVERNANCE_LEARNING_STALLED": "Improve coherence",
    "INSTITUTIONAL_IMPROVEMENT_POSSIBLE": "Stabilize governance posture",
    "GOVERNANCE_ADAPTATION_EMERGING": "Monitor improvement signals",
    "GOVERNANCE_MATURITY_IMPROVING": "Improve institutional maturity",
    "IMPROVEMENT_TRAJECTORY_BLOCKED": "Delay escalation until governance improves",
    "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP": "Reduce contradictions",
    "CONTRADICTIONS_DECLINING": "Strengthen evidence integrity",
    "GOVERNANCE_EVOLUTION_STRENGTHENING": "Stabilize governance posture",
}


def _gcc_improvement_probability(
    improvement_state: str,
    resilience: Dict[str, Any],
    stability: Dict[str, Any],
    coherence: Dict[str, Any],
) -> str:
    recovery_prob = resilience.get("recovery_probability", "LOW")
    prob_map = {
        "VERY_LOW": "VERY_LOW",
        "LOW": "LOW",
        "MODERATE": "MODERATE",
        "HIGH": "HIGH",
        "VERY_HIGH": "VERY_HIGH",
    }
    if improvement_state in (
        "IMPROVEMENT_TRAJECTORY_BLOCKED",
        "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP",
    ):
        return "VERY_LOW"
    if improvement_state == "GOVERNANCE_LEARNING_STALLED":
        return "LOW"
    if improvement_state == "INSTITUTIONAL_IMPROVEMENT_POSSIBLE":
        return "MODERATE"
    if improvement_state in ("GOVERNANCE_ADAPTATION_EMERGING", "CONTRADICTIONS_DECLINING"):
        return "HIGH"
    if improvement_state in ("GOVERNANCE_MATURITY_IMPROVING", "GOVERNANCE_EVOLUTION_STRENGTHENING"):
        return "VERY_HIGH"
    base = prob_map.get(recovery_prob, "LOW")
    if (
        float(stability.get("stability_score", 0.0) or 0.0) >= 0.50
        and float(coherence.get("logic_score", 0.0) or 0.0) >= 0.40
    ):
        return "HIGH" if base in ("LOW", "MODERATE") else base
    return base


def _gcc_adaptation_quality(
    improvement_state: str,
    hist: Dict[str, Any],
    coherence: Dict[str, Any],
) -> str:
    direction = hist.get("confidence_direction", "stable")
    if improvement_state in (
        "IMPROVEMENT_TRAJECTORY_BLOCKED",
        "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP",
    ):
        return "NONE"
    if improvement_state == "GOVERNANCE_LEARNING_STALLED":
        return "POOR"
    if improvement_state == "INSTITUTIONAL_IMPROVEMENT_POSSIBLE":
        return "LIMITED"
    if improvement_state in ("GOVERNANCE_ADAPTATION_EMERGING", "CONTRADICTIONS_DECLINING"):
        return "MODERATE"
    if improvement_state in ("GOVERNANCE_MATURITY_IMPROVING", "GOVERNANCE_EVOLUTION_STRENGTHENING"):
        return "STRONG"
    if direction == "improving":
        return "MODERATE"
    return "LIMITED"


def _gcc_improvement_velocity(
    improvement_state: str,
    hist: Dict[str, Any],
    stability: Dict[str, Any],
) -> str:
    direction = hist.get("confidence_direction", "stable")
    if improvement_state in (
        "IMPROVEMENT_TRAJECTORY_BLOCKED",
        "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP",
        "GOVERNANCE_LEARNING_STALLED",
    ):
        return "NONE"
    if improvement_state == "GOVERNANCE_EVOLUTION_STRENGTHENING":
        return "FAST"
    if improvement_state in ("GOVERNANCE_MATURITY_IMPROVING", "CONTRADICTIONS_DECLINING"):
        return "MODERATE"
    if improvement_state == "GOVERNANCE_ADAPTATION_EMERGING" or direction == "improving":
        return "SLOW"
    if stability.get("stability_state") == "GOVERNANCE_STABILITY_IMPROVING":
        return "SLOW"
    return "NONE"


def _gcc_improvement_drivers(
    improvement_state: str,
    hist: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    coherence: Dict[str, Any],
    resilience: Dict[str, Any],
    stability: Dict[str, Any],
    decision: Dict[str, Any],
    regime: Dict[str, Any],
) -> List[str]:
    drivers: List[str] = []

    if improvement_state == "GOVERNANCE_LEARNING_STALLED":
        drivers.append(
            f"Contradictions remain persistent ({tension.get('contradiction_count', 0)})"
        )
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Confidence integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(
            f"Institutional maturity stagnant ({_gcc_fmt_conf(decision.get('governance_maturity_score'))})"
        )
    elif improvement_state == "IMPROVEMENT_TRAJECTORY_BLOCKED":
        drivers.append(f"Constitutional pressure: {regime.get('regime_display', '—')}")
        drivers.append(f"Resilience: {resilience.get('resilience_display', '—')}")
        drivers.append(f"Recovery probability: {resilience.get('recovery_probability', '—')}")
        drivers.append(f"Drift containment: {stability.get('containment_strategy', '—')}")
    elif improvement_state == "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP":
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Stability: {stability.get('stability_display', '—')}")
        drivers.append(f"Confidence direction: {hist.get('confidence_direction', '—')}")
        drivers.append("No meaningful progression detected")
    elif improvement_state == "GOVERNANCE_MATURITY_IMPROVING":
        drivers.append(f"Stability: {stability.get('stability_display', '—')}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Consensus alignment: {coherence.get('cross_section_agreement', '—')}")
        drivers.append(
            f"Maturity score: {_gcc_fmt_conf(decision.get('governance_maturity_score'))}"
        )
    elif improvement_state in ("GOVERNANCE_ADAPTATION_EMERGING", "CONTRADICTIONS_DECLINING"):
        drivers.append(f"Confidence direction: {hist.get('confidence_direction', '—')}")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Self-stabilization: {resilience.get('self_stabilization_potential', '—')}")
    else:
        drivers.append(f"Resilience: {resilience.get('resilience_display', '—')}")
        drivers.append(f"Recovery probability: {resilience.get('recovery_probability', '—')}")
        drivers.append(f"Stability score: {_gcc_fmt_conf(stability.get('stability_score'))}")
        drivers.append(f"Institutional momentum: {hist.get('institutional_momentum', '—')}")

    return drivers[:6]


def _gcc_detect_governance_improvement(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
) -> Dict[str, Any]:
    direction = hist.get("confidence_direction", "stable")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    logic_score = float(coherence.get("logic_score", 0.0) or 0.0)
    resilience_state = resilience.get("resilience_state", "")
    stability_state = stability.get("stability_state", "")
    coherence_state = coherence.get("coherence_state", "")

    def _match_blocked() -> bool:
        return (
            resilience_state in ("GOVERNANCE_RECOVERY_BLOCKED", "CONSTITUTIONAL_RECOVERY_UNLIKELY")
            or stability_state in ("CONSTITUTIONAL_DRIFT_RISK", "GOVERNANCE_DETERIORATING")
            or resilience.get("recovery_probability") == "VERY_LOW"
        )

    def _match_low_quality_loop() -> bool:
        return (
            contradiction_count >= 2
            and direction in ("dormant", "mixed", "deteriorating")
            and maturity < 0.20
            and coherence_state in ("FRAGMENTED_REASONING_CHAIN", "LOW_COHERENCE_GOVERNANCE")
            and not hist.get("has_transitions")
        )

    def _match_learning_stalled() -> bool:
        return (
            contradiction_count >= 2
            and logic_score < 0.30
            and maturity < 0.25
            and integrity.get("integrity_state")
            in (
                "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
                "GOVERNANCE_OVERCONFIDENT",
            )
        )

    def _match_improvement_possible() -> bool:
        return resilience_state in (
            "RECOVERABLE_FRAGMENTATION",
            "REVERSIBLE_GOVERNANCE_DETERIORATION",
            "MODERATE_GOVERNANCE_RESILIENCE",
        )

    def _match_adaptation_emerging() -> bool:
        return resilience_state == "GOVERNANCE_SELF_STABILIZATION_EMERGING" or (
            direction == "improving"
            and contradiction_count <= 1
            and coherence.get("reasoning_chain_status") not in ("BROKEN",)
        )

    def _match_contradictions_declining() -> bool:
        return (
            contradiction_count <= 1
            and direction in ("improving", "stable")
            and coherence_state
            not in ("FRAGMENTED_REASONING_CHAIN", "INTERNALLY_INCONSISTENT_GOVERNANCE")
            and stability_state in ("GOVERNANCE_STABILITY_IMPROVING", "STABLE_GOVERNANCE_POSTURE")
        )

    def _match_maturity_improving() -> bool:
        return (
            direction == "improving"
            and maturity >= 0.18
            and stability_state in ("GOVERNANCE_STABILITY_IMPROVING", "STABLE_GOVERNANCE_POSTURE")
            and logic_score >= 0.30
        )

    def _match_evolution_strengthening() -> bool:
        return (
            resilience_state == "HIGH_GOVERNANCE_RESILIENCE"
            and stability_state == "STABLE_GOVERNANCE_POSTURE"
            and direction in ("improving", "stable")
            and contradiction_count == 0
            and logic_score >= 0.55
            and float(stability.get("stability_score", 0.0) or 0.0) >= 0.55
        )

    matchers = {
        "IMPROVEMENT_TRAJECTORY_BLOCKED": _match_blocked,
        "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP": _match_low_quality_loop,
        "GOVERNANCE_LEARNING_STALLED": _match_learning_stalled,
        "INSTITUTIONAL_IMPROVEMENT_POSSIBLE": _match_improvement_possible,
        "GOVERNANCE_ADAPTATION_EMERGING": _match_adaptation_emerging,
        "CONTRADICTIONS_DECLINING": _match_contradictions_declining,
        "GOVERNANCE_MATURITY_IMPROVING": _match_maturity_improving,
        "GOVERNANCE_EVOLUTION_STRENGTHENING": _match_evolution_strengthening,
    }

    improvement_state = "GOVERNANCE_LEARNING_STALLED"
    for candidate in _GCC_IMPROVEMENT_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            improvement_state = candidate
            break

    improvement_prob = _gcc_improvement_probability(
        improvement_state, resilience, stability, coherence
    )
    adaptation = _gcc_adaptation_quality(improvement_state, hist, coherence)
    velocity = _gcc_improvement_velocity(improvement_state, hist, stability)
    drivers = _gcc_improvement_drivers(
        improvement_state,
        hist,
        tension,
        integrity,
        coherence,
        resilience,
        stability,
        decision,
        regime,
    )

    return {
        "improvement_state": improvement_state,
        "improvement_display": _GCC_IMPROVEMENT_DISPLAY.get(
            improvement_state, improvement_state.replace("_", " ").title()
        ),
        "improvement_probability": improvement_prob,
        "adaptation_quality": adaptation,
        "improvement_velocity": velocity,
        "drivers": drivers,
        "interpretation": _GCC_IMPROVEMENT_INTERPRETATION.get(improvement_state, ""),
        "improvement_action": _GCC_IMPROVEMENT_ACTION.get(
            improvement_state, "CONTINUE_OBSERVATION"
        ),
        "improvement_pathway": _GCC_IMPROVEMENT_PATHWAY.get(
            improvement_state, "Monitor improvement signals"
        ),
    }


def _gcc_render_improvement_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
) -> Dict[str, Any]:
    improvement = _gcc_detect_governance_improvement(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
    )

    st.markdown("### Governance Institutional Learning & Improvement Intelligence")
    st.caption(
        "Institutional learning analysis — whether governance is improving and adapting constructively. "
        "**Read-only improvement view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Learning / Improvement State", improvement["improvement_display"])
    c2.metric("Improvement Probability", improvement["improvement_probability"])
    c3.metric("Adaptation Quality", improvement["adaptation_quality"])
    c4.metric("Improvement Velocity", improvement["improvement_velocity"])

    state = improvement["improvement_state"]
    if state in ("IMPROVEMENT_TRAJECTORY_BLOCKED", "GOVERNANCE_TRAPPED_IN_LOW_QUALITY_LOOP"):
        st.error(improvement["interpretation"])
    elif state == "GOVERNANCE_LEARNING_STALLED":
        st.warning(improvement["interpretation"])
    elif state in ("GOVERNANCE_EVOLUTION_STRENGTHENING", "GOVERNANCE_MATURITY_IMPROVING"):
        st.success(improvement["interpretation"])
    else:
        st.info(improvement["interpretation"])

    st.markdown("**Improvement Drivers**")
    for driver in improvement["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Improvement Action", improvement["improvement_action"])
    a2.metric("Institutional Improvement Pathway", improvement["improvement_pathway"])

    with st.expander("Improvement analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{improvement['improvement_state']}`")
        st.markdown(f"- **Resilience state:** `{resilience.get('resilience_state', '—')}`")
        st.markdown(f"- **Confidence direction:** `{hist.get('confidence_direction', '—')}`")
        st.markdown(
            f"- **Maturity score:** `{_gcc_fmt_conf(decision.get('governance_maturity_score'))}`"
        )

    return improvement


_GCC_FAILURE_SCENARIO_PRIORITY: Tuple[str, ...] = (
    "SYSTEMIC_GOVERNANCE_FAILURE_RISK",
    "CONSTITUTIONAL_BREAKDOWN_RISK",
    "GOVERNANCE_DECISION_PARALYSIS",
    "TRUSTWORTHINESS_COLLAPSE_RISK",
    "AUDITABILITY_FAILURE_RISK",
    "FRAGMENTATION_RISK_ELEVATED",
    "RECOVERABLE_GOVERNANCE_STRESS",
    "GOVERNANCE_FAILURE_CONTAINED",
)

_GCC_FAILURE_SCENARIO_DISPLAY: Dict[str, str] = {
    "SYSTEMIC_GOVERNANCE_FAILURE_RISK": "Systemic Governance Failure Risk",
    "CONSTITUTIONAL_BREAKDOWN_RISK": "Constitutional Breakdown Risk",
    "GOVERNANCE_DECISION_PARALYSIS": "Governance Decision Paralysis",
    "TRUSTWORTHINESS_COLLAPSE_RISK": "Trustworthiness Collapse Risk",
    "AUDITABILITY_FAILURE_RISK": "Auditability Failure Risk",
    "FRAGMENTATION_RISK_ELEVATED": "Fragmentation Risk Elevated",
    "RECOVERABLE_GOVERNANCE_STRESS": "Recoverable Governance Stress",
    "GOVERNANCE_FAILURE_CONTAINED": "Governance Failure Contained",
}

_GCC_FAILURE_SCENARIO_INTERPRETATION: Dict[str, str] = {
    "SYSTEMIC_GOVERNANCE_FAILURE_RISK": (
        "Governance failure risk appears systemic. Constitutional pressure, fragmented coherence, "
        "and deteriorating institutional trustworthiness materially constrain governance reliability."
    ),
    "CONSTITUTIONAL_BREAKDOWN_RISK": (
        "Constitutional breakdown risk is elevated. Governance posture remains unstable under "
        "persistent constitutional pressure and constrained institutional recovery."
    ),
    "GOVERNANCE_DECISION_PARALYSIS": (
        "Governance decision paralysis risk is present. Persistent contradictions and weak consensus "
        "block institutional reasoning and committee-level decision progress."
    ),
    "TRUSTWORTHINESS_COLLAPSE_RISK": (
        "Trustworthiness collapse risk is elevated. Confidence integrity is poor and governance "
        "evidence no longer sufficiently supports institutional conclusions."
    ),
    "AUDITABILITY_FAILURE_RISK": (
        "Auditability failure risk is present. Governance explanations remain weak and institutional "
        "transparency is deteriorating."
    ),
    "FRAGMENTATION_RISK_ELEVATED": (
        "Fragmentation risk is elevated. Contradictions are manageable but growing, and coherence "
        "indicators continue to weaken."
    ),
    "RECOVERABLE_GOVERNANCE_STRESS": (
        "Governance remains under institutional stress, but failure appears containable. "
        "Stabilization pathways remain plausible if coherence improves and contradictions decline."
    ),
    "GOVERNANCE_FAILURE_CONTAINED": (
        "Governance failure risk currently appears contained. Institutional safeguards, auditability, "
        "and governance coherence remain sufficiently stable."
    ),
}

_GCC_FAILURE_SCENARIO_ACTION: Dict[str, str] = {
    "SYSTEMIC_GOVERNANCE_FAILURE_RISK": "PRESERVE_CONSTITUTIONAL_LOCK",
    "CONSTITUTIONAL_BREAKDOWN_RISK": "PRESERVE_CONSTITUTIONAL_LOCK",
    "GOVERNANCE_DECISION_PARALYSIS": "STABILIZE_REASONING_CHAIN",
    "TRUSTWORTHINESS_COLLAPSE_RISK": "IMPROVE_CONFIDENCE_INTEGRITY",
    "AUDITABILITY_FAILURE_RISK": "IMPROVE_AUDITABILITY",
    "FRAGMENTATION_RISK_ELEVATED": "REDUCE_FRAGMENTATION",
    "RECOVERABLE_GOVERNANCE_STRESS": "MONITOR_FAILURE_SIGNALS",
    "GOVERNANCE_FAILURE_CONTAINED": "CONTINUE_OBSERVATION",
}


def _gcc_failure_probability(
    scenario_state: str,
    resilience: Dict[str, Any],
    improvement: Dict[str, Any],
    stability: Dict[str, Any],
) -> str:
    if scenario_state in ("SYSTEMIC_GOVERNANCE_FAILURE_RISK", "CONSTITUTIONAL_BREAKDOWN_RISK"):
        return "VERY_HIGH"
    if scenario_state in ("GOVERNANCE_DECISION_PARALYSIS", "TRUSTWORTHINESS_COLLAPSE_RISK"):
        return "HIGH"
    if scenario_state in ("AUDITABILITY_FAILURE_RISK", "FRAGMENTATION_RISK_ELEVATED"):
        return "MODERATE"
    if scenario_state == "RECOVERABLE_GOVERNANCE_STRESS":
        return "LOW"
    if scenario_state == "GOVERNANCE_FAILURE_CONTAINED":
        return "VERY_LOW"
    if resilience.get("recovery_probability") == "VERY_LOW":
        return "HIGH"
    if improvement.get("improvement_probability") == "VERY_LOW":
        return "HIGH"
    if float(stability.get("stability_score", 0.0) or 0.0) < 0.30:
        return "MODERATE"
    return "LOW"


def _gcc_failure_severity(
    scenario_state: str,
    failure: Dict[str, Any],
) -> str:
    if scenario_state == "SYSTEMIC_GOVERNANCE_FAILURE_RISK":
        return "CRITICAL"
    if scenario_state in ("CONSTITUTIONAL_BREAKDOWN_RISK", "GOVERNANCE_DECISION_PARALYSIS"):
        return "SEVERE"
    if scenario_state in ("TRUSTWORTHINESS_COLLAPSE_RISK", "AUDITABILITY_FAILURE_RISK"):
        return "MATERIAL"
    if scenario_state == "FRAGMENTATION_RISK_ELEVATED":
        return "MODERATE"
    if scenario_state == "RECOVERABLE_GOVERNANCE_STRESS":
        return "MINOR"
    if failure.get("risk_severity") == "CRITICAL":
        return "CRITICAL"
    return "MINOR"


def _gcc_containment_strength(
    scenario_state: str,
    audit: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    if scenario_state in (
        "SYSTEMIC_GOVERNANCE_FAILURE_RISK",
        "CONSTITUTIONAL_BREAKDOWN_RISK",
        "TRUSTWORTHINESS_COLLAPSE_RISK",
    ):
        return "NONE" if scenario_state == "SYSTEMIC_GOVERNANCE_FAILURE_RISK" else "WEAK"
    if scenario_state in ("GOVERNANCE_DECISION_PARALYSIS", "AUDITABILITY_FAILURE_RISK"):
        return "WEAK"
    if scenario_state == "FRAGMENTATION_RISK_ELEVATED":
        return "LIMITED"
    if scenario_state == "RECOVERABLE_GOVERNANCE_STRESS":
        return "MODERATE"
    if scenario_state == "GOVERNANCE_FAILURE_CONTAINED":
        return "STRONG"

    score = (
        float(stability.get("stability_score", 0.0) or 0.0) * 0.30
        + float(audit.get("evidence_integrity_score", 0.0) or 0.0) * 0.25
        + float(decision.get("governance_maturity_score", 0.0) or 0.0) * 0.20
    )
    recovery = resilience.get("recovery_probability", "LOW")
    if recovery in ("HIGH", "VERY_HIGH"):
        score += 0.15
    elif recovery == "VERY_LOW":
        score -= 0.15

    if score >= 0.65:
        return "STRONG"
    if score >= 0.45:
        return "MODERATE"
    if score >= 0.25:
        return "LIMITED"
    return "WEAK"


def _gcc_failure_pathway_warnings(
    scenario_state: str,
    hist: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    regime: Dict[str, Any],
    decision: Dict[str, Any],
) -> Tuple[str, List[str]]:
    warnings: List[str] = []

    if int(tension.get("contradiction_count", 0) or 0) >= 1:
        warnings.append("Persistent contradictions")
    if coherence.get("reasoning_chain_status") in ("BROKEN", "FRAGMENTED"):
        warnings.append("Declining governance coherence")
    if float(integrity.get("discounted_trust_score", 0.0) or 0.0) < 0.35:
        warnings.append("Trustworthiness deterioration")
    if audit.get("audit_state") in (
        "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",
        "EVIDENCE_FRAGMENTED",
        "LOW_AUDITABILITY",
    ):
        warnings.append("Weakening auditability")
    if regime.get("regime") == "CONSTITUTIONAL_STRESS":
        warnings.append("Escalating constitutional pressure")
    if decision.get("discussability") in ("NOT_DISCUSSABLE",):
        warnings.append("Delayed institutional escalation")

    pathway_map = {
        "SYSTEMIC_GOVERNANCE_FAILURE_RISK": "Systemic institutional breakdown pathway",
        "CONSTITUTIONAL_BREAKDOWN_RISK": "Constitutional constraint escalation pathway",
        "GOVERNANCE_DECISION_PARALYSIS": "Institutional decision paralysis pathway",
        "TRUSTWORTHINESS_COLLAPSE_RISK": "Confidence integrity collapse pathway",
        "AUDITABILITY_FAILURE_RISK": "Governance transparency failure pathway",
        "FRAGMENTATION_RISK_ELEVATED": "Progressive fragmentation pathway",
        "RECOVERABLE_GOVERNANCE_STRESS": "Containable stress with stabilization potential",
        "GOVERNANCE_FAILURE_CONTAINED": "Low-risk contained governance posture",
    }
    pathway = pathway_map.get(scenario_state, "Monitor failure signals")

    if not warnings:
        if scenario_state == "GOVERNANCE_FAILURE_CONTAINED":
            warnings = ["No elevated early-warning signals detected"]
        else:
            warnings = [f"Confidence direction: {hist.get('confidence_direction', '—')}"]

    return pathway, warnings[:6]


def _gcc_failure_scenario_drivers(
    scenario_state: str,
    regime: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    resilience: Dict[str, Any],
    failure: Dict[str, Any],
) -> List[str]:
    drivers: List[str] = []

    if scenario_state == "SYSTEMIC_GOVERNANCE_FAILURE_RISK":
        drivers.append(f"Constitutional pressure: {regime.get('regime_display', '—')}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
        drivers.append(f"Recovery probability: {resilience.get('recovery_probability', '—')}")
    elif scenario_state == "GOVERNANCE_DECISION_PARALYSIS":
        drivers.append(f"Contradictions: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Consensus: {coherence.get('cross_section_agreement', '—')}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Improvement state: blocked or stalled")
    elif scenario_state == "GOVERNANCE_FAILURE_CONTAINED":
        drivers.append(f"Failure risk: {failure.get('risk_display', '—')}")
        drivers.append(f"Stability containment active")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
        drivers.append(f"Recovery probability: {resilience.get('recovery_probability', '—')}")
    elif scenario_state == "TRUSTWORTHINESS_COLLAPSE_RISK":
        drivers.append(f"Integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(f"Overconfidence risk: {integrity.get('overconfidence_risk', '—')}")
        drivers.append(f"Trust discount: {_gcc_fmt_conf(integrity.get('confidence_discount'))}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
    elif scenario_state == "CONSTITUTIONAL_BREAKDOWN_RISK":
        drivers.append(f"Regime: {regime.get('regime_display', '—')}")
        drivers.append(f"Resilience: {resilience.get('resilience_display', '—')}")
        drivers.append(f"Failure risk: {failure.get('risk_display', '—')}")
        drivers.append(f"Recovery blocked: {resilience.get('resilience_state', '—')}")
    else:
        drivers.append(f"Tension: {tension.get('tension_display', '—')}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Fragility: {_gcc_fmt_conf(failure.get('fragility_score'))}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")

    return drivers[:6]


def _gcc_detect_governance_failure_scenario(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    improvement: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )

    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    failure_risk = failure.get("risk_state", "")
    coherence_state = coherence.get("coherence_state", "")
    reasoning = coherence.get("reasoning_chain_status", "")
    audit_state = audit.get("audit_state", "")
    resilience_state = resilience.get("resilience_state", "")
    fragility = float(failure.get("fragility_score", 0.0) or 0.0)
    consensus_state = consensus.get("consensus_state", "")

    def _match_systemic() -> bool:
        return (
            failure.get("risk_severity") == "CRITICAL"
            and reasoning == "BROKEN"
            and trust < 0.25
            and audit_state
            in (
                "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",
                "EVIDENCE_FRAGMENTED",
                "LOW_AUDITABILITY",
            )
            and (resilience_state == "GOVERNANCE_RECOVERY_BLOCKED" or fragility >= 0.85)
        )

    def _match_constitutional_breakdown() -> bool:
        return (
            regime.get("regime") == "CONSTITUTIONAL_STRESS"
            and snap["constitutional_safe"] is False
            and (
                stability.get("stability_state")
                in (
                    "CONSTITUTIONAL_DRIFT_RISK",
                    "GOVERNANCE_DETERIORATING",
                )
                or resilience_state
                in (
                    "GOVERNANCE_RECOVERY_BLOCKED",
                    "CONSTITUTIONAL_RECOVERY_UNLIKELY",
                )
            )
        )

    def _match_decision_paralysis() -> bool:
        return failure_risk == "DECISION_PARALYSIS_RISK" or (
            consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            and contradiction_count >= 2
            and decision.get("readiness_state") == "NOT_INSTITUTIONALLY_DISCUSSABLE"
        )

    def _match_trust_collapse() -> bool:
        return (
            integrity.get("integrity_state")
            in (
                "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
                "GOVERNANCE_OVERCONFIDENT",
            )
            and trust < 0.30
            and integrity.get("overconfidence_risk") in ("HIGH", "CRITICAL")
        )

    def _match_auditability_failure() -> bool:
        return (
            audit_state
            in (
                "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",
                "EVIDENCE_FRAGMENTED",
                "LOW_AUDITABILITY",
            )
            and audit.get("traceability_status") in ("NOT_TRACEABLE", "PARTIALLY_TRACEABLE")
            and audit.get("explainability_quality") in ("POOR", "LIMITED")
        )

    def _match_fragmentation_elevated() -> bool:
        return (
            consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            or coherence_state == "FRAGMENTED_REASONING_CHAIN"
            or contradiction_count >= 1
        ) and not _match_systemic()

    def _match_recoverable_stress() -> bool:
        return resilience_state in (
            "RECOVERABLE_FRAGMENTATION",
            "REVERSIBLE_GOVERNANCE_DETERIORATION",
            "MODERATE_GOVERNANCE_RESILIENCE",
        ) or (
            stability.get("stability_state") == "GOVERNANCE_DETERIORATING"
            and resilience.get("recovery_probability") in ("MODERATE", "HIGH")
        )

    def _match_contained() -> bool:
        return (
            failure_risk == "GOVERNANCE_STABLE"
            and contradiction_count == 0
            and coherence_state
            in (
                "HIGHLY_COHERENT_GOVERNANCE",
                "MODERATELY_COHERENT_GOVERNANCE",
                "LOGICALLY_CONSTRAINED_BUT_COHERENT",
            )
            and audit_state == "HIGH_AUDITABILITY"
        )

    matchers = {
        "SYSTEMIC_GOVERNANCE_FAILURE_RISK": _match_systemic,
        "CONSTITUTIONAL_BREAKDOWN_RISK": _match_constitutional_breakdown,
        "GOVERNANCE_DECISION_PARALYSIS": _match_decision_paralysis,
        "TRUSTWORTHINESS_COLLAPSE_RISK": _match_trust_collapse,
        "AUDITABILITY_FAILURE_RISK": _match_auditability_failure,
        "FRAGMENTATION_RISK_ELEVATED": _match_fragmentation_elevated,
        "RECOVERABLE_GOVERNANCE_STRESS": _match_recoverable_stress,
        "GOVERNANCE_FAILURE_CONTAINED": _match_contained,
    }

    scenario_state = "FRAGMENTATION_RISK_ELEVATED"
    for candidate in _GCC_FAILURE_SCENARIO_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            scenario_state = candidate
            break

    failure_prob = _gcc_failure_probability(scenario_state, resilience, improvement, stability)
    failure_severity = _gcc_failure_severity(scenario_state, failure)
    containment = _gcc_containment_strength(scenario_state, audit, stability, resilience, decision)
    drivers = _gcc_failure_scenario_drivers(
        scenario_state, regime, tension, integrity, audit, coherence, resilience, failure
    )
    pathway, warnings = _gcc_failure_pathway_warnings(
        scenario_state, hist, tension, integrity, audit, coherence, regime, decision
    )

    return {
        "scenario_state": scenario_state,
        "scenario_display": _GCC_FAILURE_SCENARIO_DISPLAY.get(
            scenario_state, scenario_state.replace("_", " ").title()
        ),
        "failure_probability": failure_prob,
        "failure_severity": failure_severity,
        "containment_strength": containment,
        "drivers": drivers,
        "interpretation": _GCC_FAILURE_SCENARIO_INTERPRETATION.get(scenario_state, ""),
        "containment_action": _GCC_FAILURE_SCENARIO_ACTION.get(
            scenario_state, "MONITOR_FAILURE_SIGNALS"
        ),
        "failure_pathway": pathway,
        "early_warnings": warnings,
    }


def _gcc_render_failure_scenario_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    improvement: Dict[str, Any],
) -> Dict[str, Any]:
    scenario = _gcc_detect_governance_failure_scenario(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
        improvement=improvement,
    )

    st.markdown("### Governance Institutional Failure Scenario Intelligence")
    st.caption(
        "Institutional stress testing — how governance could fail and what containment remains. "
        "**Read-only failure scenario view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failure Scenario State", scenario["scenario_display"])
    c2.metric("Failure Probability", scenario["failure_probability"])
    c3.metric("Failure Severity", scenario["failure_severity"])
    c4.metric("Containment Strength", scenario["containment_strength"])

    state = scenario["scenario_state"]
    severity = scenario["failure_severity"]
    if state == "SYSTEMIC_GOVERNANCE_FAILURE_RISK" or severity == "CRITICAL":
        st.error(scenario["interpretation"])
    elif state in (
        "CONSTITUTIONAL_BREAKDOWN_RISK",
        "GOVERNANCE_DECISION_PARALYSIS",
        "TRUSTWORTHINESS_COLLAPSE_RISK",
    ) or severity in ("SEVERE", "MATERIAL"):
        st.warning(scenario["interpretation"])
    elif state == "GOVERNANCE_FAILURE_CONTAINED":
        st.success(scenario["interpretation"])
    else:
        st.info(scenario["interpretation"])

    st.markdown("**Failure Drivers**")
    for driver in scenario["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Containment Action", scenario["containment_action"])
    a2.metric("Failure Pathway", scenario["failure_pathway"])

    st.markdown("**Early Warning Signals**")
    for warning in scenario["early_warnings"]:
        if state in ("SYSTEMIC_GOVERNANCE_FAILURE_RISK", "CONSTITUTIONAL_BREAKDOWN_RISK"):
            st.warning(f"• {warning}")
        else:
            st.markdown(f'<div class="gcc-block-item">• {warning}</div>', unsafe_allow_html=True)

    with st.expander("Failure scenario analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{scenario['scenario_state']}`")
        st.markdown(f"- **Resilience state:** `{resilience.get('resilience_state', '—')}`")
        st.markdown(f"- **Improvement state:** `{improvement.get('improvement_state', '—')}`")
        st.markdown(f"- **Failure risk state:** `{failure.get('risk_state', '—')}`")

    return scenario


_GCC_INTERVENTION_PRIORITY: Tuple[str, ...] = (
    "CONSTITUTIONAL_LOCK_REQUIRED",
    "ESCALATION_REQUIRED",
    "CONTAINMENT_RECOMMENDED",
    "INSTITUTIONAL_CAUTION_REQUIRED",
    "MONITORING_ELEVATED",
    "OBSERVATION_ONLY",
)

_GCC_INTERVENTION_DISPLAY: Dict[str, str] = {
    "CONSTITUTIONAL_LOCK_REQUIRED": "Constitutional Lock Required",
    "ESCALATION_REQUIRED": "Escalation Required",
    "CONTAINMENT_RECOMMENDED": "Containment Recommended",
    "INSTITUTIONAL_CAUTION_REQUIRED": "Institutional Caution Required",
    "MONITORING_ELEVATED": "Monitoring Elevated",
    "OBSERVATION_ONLY": "Observation Only",
}

_GCC_INTERVENTION_INTERPRETATION: Dict[str, str] = {
    "CONSTITUTIONAL_LOCK_REQUIRED": (
        "Governance containment posture remains defensive. Constitutional pressure, degraded "
        "trustworthiness, and systemic governance fragility justify preservation of institutional "
        "safeguards and delayed escalation."
    ),
    "ESCALATION_REQUIRED": (
        "Institutional deterioration is material. Governance failure risk is severe enough to "
        "justify operator escalation and heightened human review."
    ),
    "CONTAINMENT_RECOMMENDED": (
        "Governance deterioration appears manageable but material. Institutional containment and "
        "heightened observation appear warranted while stabilization signals are monitored."
    ),
    "INSTITUTIONAL_CAUTION_REQUIRED": (
        "Governance is weakening but deterioration remains manageable. Institutional caution and "
        "increased observability are appropriate until stability improves."
    ),
    "MONITORING_ELEVATED": (
        "Governance stress is visible but intervention is not yet required. Monitoring intensity "
        "should increase while containment signals are tracked."
    ),
    "OBSERVATION_ONLY": (
        "Governance remains institutionally stable. Current safeguards appear sufficient and no "
        "escalation posture is presently justified."
    ),
}

_GCC_INTERVENTION_ACTION: Dict[str, str] = {
    "CONSTITUTIONAL_LOCK_REQUIRED": "PRESERVE_CONSTITUTIONAL_LOCK",
    "ESCALATION_REQUIRED": "ESCALATE_HUMAN_REVIEW",
    "CONTAINMENT_RECOMMENDED": "CONTAIN_AND_MONITOR",
    "INSTITUTIONAL_CAUTION_REQUIRED": "INCREASE_OBSERVABILITY",
    "MONITORING_ELEVATED": "MONITOR_CONTAINMENT_SIGNALS",
    "OBSERVATION_ONLY": "CONTINUE_OBSERVATION",
}

_GCC_INTERVENTION_PROTOCOL: Dict[str, List[str]] = {
    "CONSTITUTIONAL_LOCK_REQUIRED": [
        "Preserve constitutional safeguards",
        "Delay escalation until stability improves",
        "Monitor trustworthiness deterioration",
        "Reassess once institutional posture stabilizes",
    ],
    "ESCALATION_REQUIRED": [
        "Escalate to human review",
        "Monitor governance coherence",
        "Observe trustworthiness deterioration",
        "Reassess containment posture daily",
    ],
    "CONTAINMENT_RECOMMENDED": [
        "Contain and monitor governance stress",
        "Monitor contradiction intensity",
        "Observe trustworthiness deterioration",
        "Delay escalation until stability improves",
    ],
    "INSTITUTIONAL_CAUTION_REQUIRED": [
        "Increase observability",
        "Monitor governance coherence",
        "Monitor contradiction intensity",
        "Reassess once institutional posture stabilizes",
    ],
    "MONITORING_ELEVATED": [
        "Monitor containment signals",
        "Monitor governance coherence",
        "Observe constitutional pressure",
    ],
    "OBSERVATION_ONLY": [
        "Continue observation only",
        "Monitor routine governance signals",
    ],
}


def _gcc_institutional_urgency(posture: str, scenario: Dict[str, Any]) -> str:
    if posture == "CONSTITUTIONAL_LOCK_REQUIRED":
        return "CRITICAL"
    if posture == "ESCALATION_REQUIRED":
        return "HIGH"
    if posture == "CONTAINMENT_RECOMMENDED":
        return "HIGH" if scenario.get("failure_severity") in ("SEVERE", "CRITICAL") else "MODERATE"
    if posture == "INSTITUTIONAL_CAUTION_REQUIRED":
        return "MODERATE"
    if posture == "MONITORING_ELEVATED":
        return "GUARDED"
    return "LOW"


def _gcc_escalation_threshold(posture: str, scenario: Dict[str, Any]) -> str:
    if posture == "CONSTITUTIONAL_LOCK_REQUIRED":
        return "IMMEDIATE"
    if posture == "ESCALATION_REQUIRED":
        return "IMMEDIATE"
    if posture == "CONTAINMENT_RECOMMENDED":
        return "MATERIAL"
    if posture == "INSTITUTIONAL_CAUTION_REQUIRED":
        return "HEIGHTENED"
    if posture == "MONITORING_ELEVATED":
        return "WATCH"
    if scenario.get("scenario_state") == "GOVERNANCE_FAILURE_CONTAINED":
        return "NONE"
    return "NONE"


def _gcc_containment_readiness(
    posture: str,
    scenario: Dict[str, Any],
    resilience: Dict[str, Any],
    audit: Dict[str, Any],
    stability: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    strength = scenario.get("containment_strength", "WEAK")
    if posture in ("CONSTITUTIONAL_LOCK_REQUIRED", "ESCALATION_REQUIRED"):
        return "NONE" if strength in ("NONE", "WEAK") else "WEAK"
    if strength == "STRONG":
        return "STRONG"
    if strength == "MODERATE":
        return "MODERATE"
    if strength == "LIMITED":
        return "LIMITED"

    score = (
        float(stability.get("stability_score", 0.0) or 0.0) * 0.25
        + float(audit.get("evidence_integrity_score", 0.0) or 0.0) * 0.25
        + float(decision.get("governance_maturity_score", 0.0) or 0.0) * 0.20
    )
    recovery = resilience.get("recovery_probability", "LOW")
    if recovery in ("HIGH", "VERY_HIGH"):
        score += 0.15
    elif recovery == "VERY_LOW":
        score -= 0.15

    if score >= 0.55:
        return "MODERATE"
    if score >= 0.30:
        return "LIMITED"
    return "WEAK"


def _gcc_intervention_drivers(
    posture: str,
    regime: Dict[str, Any],
    scenario: Dict[str, Any],
    resilience: Dict[str, Any],
    stability: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    coherence: Dict[str, Any],
) -> List[str]:
    drivers: List[str] = []

    if posture == "CONSTITUTIONAL_LOCK_REQUIRED":
        drivers.append(f"Constitutional pressure: {regime.get('regime_display', '—')}")
        drivers.append(f"Resilience: {resilience.get('resilience_display', '—')}")
        drivers.append(f"Failure scenario: {scenario.get('scenario_display', '—')}")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Containment strength: {scenario.get('containment_strength', '—')}")
    elif posture == "CONTAINMENT_RECOMMENDED":
        drivers.append(f"Failure scenario: {scenario.get('scenario_display', '—')}")
        drivers.append(f"Stability: {stability.get('stability_display', '—')}")
        drivers.append(f"Contradictions: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Recovery probability: {resilience.get('recovery_probability', '—')}")
    elif posture == "OBSERVATION_ONLY":
        drivers.append(f"Failure scenario: {scenario.get('scenario_display', '—')}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Stability score: {_gcc_fmt_conf(stability.get('stability_score'))}")
        drivers.append("Failure risk contained")
    elif posture == "ESCALATION_REQUIRED":
        drivers.append(f"Failure severity: {scenario.get('failure_severity', '—')}")
        drivers.append(f"Failure probability: {scenario.get('failure_probability', '—')}")
        drivers.append(f"Institutional urgency elevated")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
    else:
        drivers.append(f"Stability: {stability.get('stability_display', '—')}")
        drivers.append(f"Improvement blocked or limited")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Containment strength: {scenario.get('containment_strength', '—')}")

    return drivers[:6]


def _gcc_detect_governance_intervention(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    improvement: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    snap = _gcc_collect_governance_snapshot(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )

    scenario_state = scenario.get("scenario_state", "")
    failure_severity = scenario.get("failure_severity", "MINOR")
    failure_prob = scenario.get("failure_probability", "LOW")
    containment = scenario.get("containment_strength", "WEAK")
    resilience_state = resilience.get("resilience_state", "")
    improvement_state = improvement.get("improvement_state", "")
    stability_state = stability.get("stability_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)

    def _match_constitutional_lock() -> bool:
        return (
            scenario_state
            in (
                "SYSTEMIC_GOVERNANCE_FAILURE_RISK",
                "CONSTITUTIONAL_BREAKDOWN_RISK",
            )
            or (
                resilience_state == "GOVERNANCE_RECOVERY_BLOCKED"
                and snap["constitutional_safe"] is False
            )
            or (
                improvement_state == "IMPROVEMENT_TRAJECTORY_BLOCKED"
                and regime.get("regime") == "CONSTITUTIONAL_STRESS"
                and containment in ("NONE", "WEAK")
            )
        )

    def _match_escalation_required() -> bool:
        return (
            failure_severity in ("SEVERE", "CRITICAL")
            and scenario_state
            in (
                "GOVERNANCE_DECISION_PARALYSIS",
                "TRUSTWORTHINESS_COLLAPSE_RISK",
            )
        ) or (
            failure.get("risk_severity") == "CRITICAL"
            and failure_prob in ("HIGH", "VERY_HIGH")
            and decision.get("discussability") == "NOT_DISCUSSABLE"
        )

    def _match_containment_recommended() -> bool:
        return (
            scenario_state
            in (
                "FRAGMENTATION_RISK_ELEVATED",
                "AUDITABILITY_FAILURE_RISK",
                "RECOVERABLE_GOVERNANCE_STRESS",
            )
            or stability_state
            in (
                "GOVERNANCE_DETERIORATING",
                "REASONING_INSTABILITY_DETECTED",
            )
        ) and failure_prob in ("MODERATE", "HIGH", "VERY_HIGH")

    def _match_caution_required() -> bool:
        return (
            resilience_state
            in (
                "LOW_GOVERNANCE_RESILIENCE",
                "RECOVERABLE_FRAGMENTATION",
            )
            or improvement_state
            in (
                "INSTITUTIONAL_IMPROVEMENT_POSSIBLE",
                "GOVERNANCE_LEARNING_STALLED",
            )
            or (trust < 0.35 and contradiction_count >= 1)
        )

    def _match_monitoring_elevated() -> bool:
        return (
            scenario_state == "FRAGMENTATION_RISK_ELEVATED"
            or stability_state in ("INSTITUTIONAL_DRIFT_DETECTED", "CONFIDENCE_OSCILLATION_RISK")
            or failure_prob == "MODERATE"
        )

    def _match_observation_only() -> bool:
        return (
            scenario_state == "GOVERNANCE_FAILURE_CONTAINED"
            and stability_state in ("STABLE_GOVERNANCE_POSTURE", "GOVERNANCE_STABILITY_IMPROVING")
            and failure_prob in ("VERY_LOW", "LOW")
            and contradiction_count == 0
        )

    matchers = {
        "CONSTITUTIONAL_LOCK_REQUIRED": _match_constitutional_lock,
        "ESCALATION_REQUIRED": _match_escalation_required,
        "CONTAINMENT_RECOMMENDED": _match_containment_recommended,
        "INSTITUTIONAL_CAUTION_REQUIRED": _match_caution_required,
        "MONITORING_ELEVATED": _match_monitoring_elevated,
        "OBSERVATION_ONLY": _match_observation_only,
    }

    posture = "INSTITUTIONAL_CAUTION_REQUIRED"
    for candidate in _GCC_INTERVENTION_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            posture = candidate
            break

    urgency = _gcc_institutional_urgency(posture, scenario)
    escalation = _gcc_escalation_threshold(posture, scenario)
    readiness_level = _gcc_containment_readiness(
        posture, scenario, resilience, audit, stability, decision
    )
    drivers = _gcc_intervention_drivers(
        posture, regime, scenario, resilience, stability, tension, integrity, coherence
    )
    protocol_steps = _GCC_INTERVENTION_PROTOCOL.get(posture, ["Continue observation only"])

    return {
        "intervention_posture": posture,
        "intervention_display": _GCC_INTERVENTION_DISPLAY.get(
            posture, posture.replace("_", " ").title()
        ),
        "institutional_urgency": urgency,
        "escalation_threshold": escalation,
        "containment_readiness": readiness_level,
        "drivers": drivers,
        "interpretation": _GCC_INTERVENTION_INTERPRETATION.get(posture, ""),
        "institutional_action": _GCC_INTERVENTION_ACTION.get(posture, "CONTINUE_OBSERVATION"),
        "containment_protocol": protocol_steps,
    }


def _gcc_render_intervention_intelligence(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    consensus: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    failure: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    improvement: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    intervention = _gcc_detect_governance_intervention(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
        improvement=improvement,
        scenario=scenario,
    )

    st.markdown("### Governance Institutional Intervention & Containment Intelligence")
    st.caption(
        "Intervention posture analysis — what institutional containment and escalation posture is appropriate. "
        "**Read-only intervention view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intervention Posture", intervention["intervention_display"])
    c2.metric("Institutional Urgency", intervention["institutional_urgency"])
    c3.metric("Escalation Threshold", intervention["escalation_threshold"])
    c4.metric("Containment Readiness", intervention["containment_readiness"])

    posture = intervention["intervention_posture"]
    urgency = intervention["institutional_urgency"]
    if posture == "CONSTITUTIONAL_LOCK_REQUIRED" or urgency == "CRITICAL":
        st.error(intervention["interpretation"])
    elif posture in ("ESCALATION_REQUIRED", "CONTAINMENT_RECOMMENDED") or urgency == "HIGH":
        st.warning(intervention["interpretation"])
    elif posture == "OBSERVATION_ONLY":
        st.success(intervention["interpretation"])
    else:
        st.info(intervention["interpretation"])

    st.markdown("**Containment Drivers**")
    for driver in intervention["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Institutional Action", intervention["institutional_action"])
    a2.metric("Protocol Steps", len(intervention["containment_protocol"]))

    st.markdown("**Institutional Containment Protocol**")
    for step in intervention["containment_protocol"]:
        if posture in ("CONSTITUTIONAL_LOCK_REQUIRED", "ESCALATION_REQUIRED"):
            st.warning(f"• {step}")
        else:
            st.markdown(f'<div class="gcc-block-item">• {step}</div>', unsafe_allow_html=True)

    with st.expander("Intervention analysis detail", expanded=False):
        st.markdown(f"- **Internal posture:** `{intervention['intervention_posture']}`")
        st.markdown(f"- **Failure scenario:** `{scenario.get('scenario_state', '—')}`")
        st.markdown(f"- **Resilience state:** `{resilience.get('resilience_state', '—')}`")
        st.markdown(f"- **Improvement state:** `{improvement.get('improvement_state', '—')}`")

    return intervention


_GCC_INST_EVIDENCE_PRIORITY: Tuple[str, ...] = (
    "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED",
    "TRUSTWORTHINESS_MATERIALLY_IMPAIRED",
    "CONFIDENCE_OVEREXTENDED",
    "EVIDENCE_QUALITY_DEGRADING",
    "PARTIALLY_SUPPORTED_CONFIDENCE",
    "CONFIDENCE_SUPPORTED",
    "EVIDENCE_INTEGRITY_STRONG",
)

_GCC_INST_EVIDENCE_DISPLAY: Dict[str, str] = {
    "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED": "Institutional Confidence Unsupported",
    "TRUSTWORTHINESS_MATERIALLY_IMPAIRED": "Trustworthiness Materially Impaired",
    "CONFIDENCE_OVEREXTENDED": "Confidence Overextended",
    "EVIDENCE_QUALITY_DEGRADING": "Evidence Quality Degrading",
    "PARTIALLY_SUPPORTED_CONFIDENCE": "Partially Supported Confidence",
    "CONFIDENCE_SUPPORTED": "Confidence Supported",
    "EVIDENCE_INTEGRITY_STRONG": "Evidence Integrity Strong",
}

_GCC_INST_EVIDENCE_INTERPRETATION: Dict[str, str] = {
    "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED": (
        "Governance confidence currently appears insufficiently supported by institutional evidence. "
        "Trustworthiness, auditability, and governance defensibility remain constrained, reducing "
        "institutional confidence in governance conclusions."
    ),
    "TRUSTWORTHINESS_MATERIALLY_IMPAIRED": (
        "Institutional trustworthiness is materially impaired. Evidence quality is weak and "
        "governance confidence should be treated with significant caution."
    ),
    "CONFIDENCE_OVEREXTENDED": (
        "Governance confidence appears to exceed available evidence quality. Institutional reasoning "
        "may be overstated and defensibility is weakened."
    ),
    "EVIDENCE_QUALITY_DEGRADING": (
        "Evidence quality is degrading. Auditability and confidence integrity are weakening, "
        "reducing institutional reliability of governance outputs."
    ),
    "PARTIALLY_SUPPORTED_CONFIDENCE": (
        "Governance confidence appears partially supported by available evidence. Institutional trust "
        "remains conditional and continued observation is warranted."
    ),
    "CONFIDENCE_SUPPORTED": (
        "Governance confidence appears mostly supported by institutional evidence. Trustworthiness "
        "and defensibility remain acceptable under current conditions."
    ),
    "EVIDENCE_INTEGRITY_STRONG": (
        "Governance evidence integrity appears strong. Institutional trustworthiness, auditability, "
        "and confidence support remain aligned and governance outputs appear defensible."
    ),
}

_GCC_INST_EVIDENCE_ACTION: Dict[str, str] = {
    "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED": "PRESERVE_CONSTITUTIONAL_LOCK",
    "TRUSTWORTHINESS_MATERIALLY_IMPAIRED": "IMPROVE_CONFIDENCE_INTEGRITY",
    "CONFIDENCE_OVEREXTENDED": "REDUCE_CONTRADICTIONS",
    "EVIDENCE_QUALITY_DEGRADING": "IMPROVE_AUDITABILITY",
    "PARTIALLY_SUPPORTED_CONFIDENCE": "MONITOR_EVIDENCE_QUALITY",
    "CONFIDENCE_SUPPORTED": "CONTINUE_OBSERVATION",
    "EVIDENCE_INTEGRITY_STRONG": "CONTINUE_OBSERVATION",
}

_GCC_INST_EVIDENCE_PROTOCOL: Dict[str, List[str]] = {
    "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED": [
        "Monitor trustworthiness deterioration",
        "Preserve constitutional safeguards if confidence weakens",
        "Observe confidence inflation risk",
        "Reassess auditability deterioration",
    ],
    "TRUSTWORTHINESS_MATERIALLY_IMPAIRED": [
        "Improve governance evidence quality",
        "Monitor trustworthiness deterioration",
        "Monitor contradiction severity",
    ],
    "CONFIDENCE_OVEREXTENDED": [
        "Observe confidence inflation risk",
        "Monitor contradiction severity",
        "Reassess auditability deterioration",
    ],
    "EVIDENCE_QUALITY_DEGRADING": [
        "Reassess auditability deterioration",
        "Monitor trustworthiness deterioration",
        "Improve governance evidence quality",
    ],
    "PARTIALLY_SUPPORTED_CONFIDENCE": [
        "Monitor evidence quality",
        "Monitor contradiction severity",
        "Observe confidence inflation risk",
    ],
    "CONFIDENCE_SUPPORTED": [
        "Continue observation",
        "Monitor routine evidence signals",
    ],
    "EVIDENCE_INTEGRITY_STRONG": [
        "Continue observation",
        "Monitor routine evidence signals",
    ],
}


def _gcc_confidence_reliability(evidence_state: str, trust: float) -> str:
    if evidence_state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED":
        return "VERY_LOW"
    if evidence_state in ("TRUSTWORTHINESS_MATERIALLY_IMPAIRED", "CONFIDENCE_OVEREXTENDED"):
        return "LOW"
    if evidence_state in ("EVIDENCE_QUALITY_DEGRADING", "PARTIALLY_SUPPORTED_CONFIDENCE"):
        return "MODERATE"
    if evidence_state == "CONFIDENCE_SUPPORTED":
        return "HIGH"
    if evidence_state == "EVIDENCE_INTEGRITY_STRONG":
        return "VERY_HIGH"
    if trust < 0.25:
        return "VERY_LOW"
    if trust < 0.40:
        return "LOW"
    if trust < 0.55:
        return "MODERATE"
    if trust < 0.70:
        return "HIGH"
    return "VERY_HIGH"


def _gcc_trustworthiness_state(trust: float, integrity: Dict[str, Any]) -> str:
    integrity_state = integrity.get("integrity_state", "")
    if trust < 0.20 or integrity_state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS":
        return "BROKEN"
    if trust < 0.35 or integrity.get("overconfidence_risk") in ("HIGH", "CRITICAL"):
        return "WEAK"
    if trust < 0.50:
        return "LIMITED"
    if trust < 0.70:
        return "ACCEPTABLE"
    return "STRONG"


def _gcc_institutional_defensibility(
    evidence_state: str,
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    if evidence_state in (
        "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED",
        "TRUSTWORTHINESS_MATERIALLY_IMPAIRED",
    ):
        return "NONE" if evidence_state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED" else "WEAK"
    if evidence_state in ("CONFIDENCE_OVEREXTENDED", "EVIDENCE_QUALITY_DEGRADING"):
        return "WEAK"
    if evidence_state == "PARTIALLY_SUPPORTED_CONFIDENCE":
        return "LIMITED"
    if evidence_state == "CONFIDENCE_SUPPORTED":
        return "MODERATE"

    score = (
        float(audit.get("evidence_integrity_score", 0.0) or 0.0) * 0.35
        + float(coherence.get("logic_score", 0.0) or 0.0) * 0.30
        + float(decision.get("governance_maturity_score", 0.0) or 0.0) * 0.20
    )
    if evidence_state == "EVIDENCE_INTEGRITY_STRONG" or score >= 0.65:
        return "STRONG"
    if score >= 0.45:
        return "MODERATE"
    return "LIMITED"


def _gcc_inst_evidence_drivers(
    evidence_state: str,
    integrity: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    decision: Dict[str, Any],
) -> List[str]:
    drivers: List[str] = []

    if evidence_state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED":
        drivers.append("Governance confidence exceeds evidence quality")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
        drivers.append(
            f"Defensibility constrained by coherence: {coherence.get('coherence_display', '—')}"
        )
        drivers.append(f"Contradictions: {tension.get('contradiction_count', 0)}")
    elif evidence_state == "CONFIDENCE_SUPPORTED":
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
        drivers.append(f"Integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(f"Maturity: {_gcc_fmt_conf(decision.get('governance_maturity_score'))}")
    elif evidence_state == "EVIDENCE_INTEGRITY_STRONG":
        drivers.append("Governance evidence strong")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
        drivers.append("Institutional defensibility high")
    elif evidence_state == "CONFIDENCE_OVEREXTENDED":
        drivers.append(f"Integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(f"Overconfidence risk: {integrity.get('overconfidence_risk', '—')}")
        drivers.append(f"Trust discount: {_gcc_fmt_conf(integrity.get('confidence_discount'))}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
    else:
        drivers.append(f"Integrity: {integrity.get('integrity_display', '—')}")
        drivers.append(f"Auditability: {audit.get('audit_display', '—')}")
        drivers.append(f"Evidence score: {_gcc_fmt_conf(audit.get('evidence_integrity_score'))}")
        drivers.append(f"Contradiction count: {tension.get('contradiction_count', 0)}")

    return drivers[:6]


def _gcc_detect_governance_institutional_evidence(
    *,
    integrity: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    decision: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    intervention: Dict[str, Any],
) -> Dict[str, Any]:
    trust = float(integrity.get("discounted_trust_score", 0.0) or 0.0)
    raw_conf = float(integrity.get("raw_confidence_context", 0.0) or 0.0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    integrity_state = integrity.get("integrity_state", "")
    audit_state = audit.get("audit_state", "")
    overconf = integrity.get("overconfidence_risk", "NONE")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    evidence_score = float(audit.get("evidence_integrity_score", 0.0) or 0.0)
    logic_score = float(coherence.get("logic_score", 0.0) or 0.0)

    def _match_unsupported() -> bool:
        return (
            audit_state == "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE"
            or intervention.get("intervention_posture") == "CONSTITUTIONAL_LOCK_REQUIRED"
            or (
                trust < 0.25
                and evidence_score < 0.20
                and integrity_state
                in (
                    "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
                    "GOVERNANCE_OVERCONFIDENT",
                )
            )
        )

    def _match_impaired() -> bool:
        return trust < 0.35 and audit_state in (
            "CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",
            "EVIDENCE_FRAGMENTED",
            "LOW_AUDITABILITY",
            "SPARSE_EVIDENCE",
        )

    def _match_overextended() -> bool:
        return (
            integrity_state == "GOVERNANCE_OVERCONFIDENT"
            or overconf in ("HIGH", "CRITICAL")
            or (raw_conf >= 0.70 and trust < 0.45 and maturity < 0.30)
        )

    def _match_degrading() -> bool:
        return (
            audit_state in ("EVIDENCE_FRAGMENTED", "LOW_AUDITABILITY")
            and stability.get("stability_state")
            in (
                "GOVERNANCE_DETERIORATING",
                "REASONING_INSTABILITY_DETECTED",
            )
        ) or (
            integrity_state == "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS"
            and audit.get("explainability_quality") in ("POOR", "LIMITED")
        )

    def _match_partial() -> bool:
        return (
            trust >= 0.25
            and evidence_score >= 0.15
            and logic_score >= 0.20
            and contradiction_count <= 2
            and audit_state not in ("CONFIDENCE_UNSUPPORTED_BY_EVIDENCE",)
        )

    def _match_supported() -> bool:
        return (
            trust >= 0.45
            and evidence_score >= 0.35
            and logic_score >= 0.35
            and contradiction_count <= 1
            and integrity_state
            not in (
                "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
                "GOVERNANCE_OVERCONFIDENT",
            )
        )

    def _match_strong() -> bool:
        return (
            audit_state == "HIGH_AUDITABILITY"
            and trust >= 0.55
            and evidence_score >= 0.50
            and logic_score >= 0.50
            and contradiction_count == 0
            and coherence.get("coherence_state")
            in (
                "HIGHLY_COHERENT_GOVERNANCE",
                "MODERATELY_COHERENT_GOVERNANCE",
                "LOGICALLY_CONSTRAINED_BUT_COHERENT",
            )
            and resilience.get("resilience_state") == "HIGH_GOVERNANCE_RESILIENCE"
        )

    matchers = {
        "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED": _match_unsupported,
        "TRUSTWORTHINESS_MATERIALLY_IMPAIRED": _match_impaired,
        "CONFIDENCE_OVEREXTENDED": _match_overextended,
        "EVIDENCE_QUALITY_DEGRADING": _match_degrading,
        "PARTIALLY_SUPPORTED_CONFIDENCE": _match_partial,
        "CONFIDENCE_SUPPORTED": _match_supported,
        "EVIDENCE_INTEGRITY_STRONG": _match_strong,
    }

    evidence_state = "PARTIALLY_SUPPORTED_CONFIDENCE"
    for candidate in _GCC_INST_EVIDENCE_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            evidence_state = candidate
            break

    reliability = _gcc_confidence_reliability(evidence_state, trust)
    trust_state = _gcc_trustworthiness_state(trust, integrity)
    defensibility = _gcc_institutional_defensibility(evidence_state, audit, coherence, decision)
    drivers = _gcc_inst_evidence_drivers(
        evidence_state, integrity, audit, coherence, tension, decision
    )
    protocol = _GCC_INST_EVIDENCE_PROTOCOL.get(evidence_state, ["Monitor evidence quality"])

    return {
        "evidence_state": evidence_state,
        "evidence_display": _GCC_INST_EVIDENCE_DISPLAY.get(
            evidence_state, evidence_state.replace("_", " ").title()
        ),
        "confidence_reliability": reliability,
        "trustworthiness_state": trust_state,
        "institutional_defensibility": defensibility,
        "drivers": drivers,
        "interpretation": _GCC_INST_EVIDENCE_INTERPRETATION.get(evidence_state, ""),
        "integrity_action": _GCC_INST_EVIDENCE_ACTION.get(evidence_state, "CONTINUE_OBSERVATION"),
        "evidence_protocol": protocol,
    }


def _gcc_render_institutional_evidence_intelligence(
    *,
    integrity: Dict[str, Any],
    audit: Dict[str, Any],
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    decision: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    intervention: Dict[str, Any],
) -> Dict[str, Any]:
    evidence = _gcc_detect_governance_institutional_evidence(
        integrity=integrity,
        audit=audit,
        coherence=coherence,
        tension=tension,
        decision=decision,
        stability=stability,
        resilience=resilience,
        intervention=intervention,
    )

    st.markdown("### Governance Institutional Confidence & Evidence Integrity Intelligence")
    st.caption(
        "Evidence trustworthiness analysis — how defensible governance confidence and conclusions are. "
        "**Read-only integrity view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence Integrity State", evidence["evidence_display"])
    c2.metric("Confidence Reliability", evidence["confidence_reliability"])
    c3.metric("Trustworthiness State", evidence["trustworthiness_state"])
    c4.metric("Institutional Defensibility", evidence["institutional_defensibility"])

    state = evidence["evidence_state"]
    if state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED":
        st.error(evidence["interpretation"])
    elif state in (
        "TRUSTWORTHINESS_MATERIALLY_IMPAIRED",
        "CONFIDENCE_OVEREXTENDED",
        "EVIDENCE_QUALITY_DEGRADING",
    ):
        st.warning(evidence["interpretation"])
    elif state in ("CONFIDENCE_SUPPORTED", "EVIDENCE_INTEGRITY_STRONG"):
        st.success(evidence["interpretation"])
    else:
        st.info(evidence["interpretation"])

    st.markdown("**Evidence Drivers**")
    for driver in evidence["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Integrity Action", evidence["integrity_action"])
    a2.metric("Protocol Steps", len(evidence["evidence_protocol"]))

    st.markdown("**Evidence Integrity Protocol**")
    for step in evidence["evidence_protocol"]:
        if state in (
            "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED",
            "TRUSTWORTHINESS_MATERIALLY_IMPAIRED",
        ):
            st.warning(f"• {step}")
        else:
            st.markdown(f'<div class="gcc-block-item">• {step}</div>', unsafe_allow_html=True)

    with st.expander("Evidence integrity analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{evidence['evidence_state']}`")
        st.markdown(f"- **Integrity state:** `{integrity.get('integrity_state', '—')}`")
        st.markdown(f"- **Auditability state:** `{audit.get('audit_state', '—')}`")
        st.markdown(
            f"- **Intervention posture:** `{intervention.get('intervention_posture', '—')}`"
        )

    return evidence


_GCC_INST_CONSENSUS_PRIORITY: Tuple[str, ...] = (
    "INSTITUTIONAL_CONSENSUS_BROKEN",
    "GOVERNANCE_INTERNAL_CONFLICT",
    "FRAGMENTATION_ELEVATED",
    "PARTIAL_ALIGNMENT",
    "GOVERNANCE_CONSENSUS_FORMING",
    "INSTITUTIONAL_ALIGNMENT_STRONG",
)

_GCC_INST_CONSENSUS_DISPLAY: Dict[str, str] = {
    "INSTITUTIONAL_CONSENSUS_BROKEN": "Institutional Consensus Broken",
    "GOVERNANCE_INTERNAL_CONFLICT": "Governance Internal Conflict",
    "FRAGMENTATION_ELEVATED": "Fragmentation Elevated",
    "PARTIAL_ALIGNMENT": "Partial Alignment",
    "GOVERNANCE_CONSENSUS_FORMING": "Governance Consensus Forming",
    "INSTITUTIONAL_ALIGNMENT_STRONG": "Institutional Alignment Strong",
}

_GCC_INST_CONSENSUS_INTERPRETATION: Dict[str, str] = {
    "INSTITUTIONAL_CONSENSUS_BROKEN": (
        "Governance alignment appears materially impaired. Institutional disagreement, fragmented "
        "coherence, and elevated contradictions constrain governance consistency and reduce "
        "confidence in unified institutional posture."
    ),
    "GOVERNANCE_INTERNAL_CONFLICT": (
        "Governance internal conflict is present. Contradictions and misaligned reasoning reduce "
        "institutional agreement and escalation posture consistency."
    ),
    "FRAGMENTATION_ELEVATED": (
        "Governance fragmentation is elevated. Institutional coherence is weakening and alignment "
        "requires continued containment monitoring."
    ),
    "PARTIAL_ALIGNMENT": (
        "Governance appears partially aligned. Institutional reasoning remains mixed, though "
        "fragmentation appears manageable and coherence remains recoverable."
    ),
    "GOVERNANCE_CONSENSUS_FORMING": (
        "Governance consensus appears to be forming. Contradictions are declining and institutional "
        "agreement signals are stabilizing."
    ),
    "INSTITUTIONAL_ALIGNMENT_STRONG": (
        "Governance alignment appears institutionally strong. Coherence, trustworthiness, and "
        "escalation posture remain consistent across governance reasoning."
    ),
}

_GCC_INST_CONSENSUS_ACTION: Dict[str, str] = {
    "INSTITUTIONAL_CONSENSUS_BROKEN": "PRESERVE_CONSTITUTIONAL_LOCK",
    "GOVERNANCE_INTERNAL_CONFLICT": "IMPROVE_GOVERNANCE_COHERENCE",
    "FRAGMENTATION_ELEVATED": "REDUCE_FRAGMENTATION",
    "PARTIAL_ALIGNMENT": "MONITOR_ALIGNMENT_SIGNALS",
    "GOVERNANCE_CONSENSUS_FORMING": "STABILIZE_ESCALATION_POSTURE",
    "INSTITUTIONAL_ALIGNMENT_STRONG": "CONTINUE_OBSERVATION",
}

_GCC_INST_CONSENSUS_PROTOCOL: Dict[str, List[str]] = {
    "INSTITUTIONAL_CONSENSUS_BROKEN": [
        "Monitor contradiction severity",
        "Observe escalation inconsistency",
        "Reassess governance coherence",
        "Stabilize institutional posture",
    ],
    "GOVERNANCE_INTERNAL_CONFLICT": [
        "Improve governance coherence",
        "Monitor contradiction severity",
        "Observe escalation inconsistency",
    ],
    "FRAGMENTATION_ELEVATED": [
        "Reduce governance fragmentation",
        "Monitor contradiction severity",
        "Reassess governance coherence",
    ],
    "PARTIAL_ALIGNMENT": [
        "Monitor alignment signals",
        "Observe escalation inconsistency",
        "Reassess governance coherence",
    ],
    "GOVERNANCE_CONSENSUS_FORMING": [
        "Stabilize escalation posture",
        "Continue observation if alignment strengthens",
        "Monitor contradiction severity",
    ],
    "INSTITUTIONAL_ALIGNMENT_STRONG": [
        "Continue observation if alignment strengthens",
        "Monitor routine alignment signals",
    ],
}


def _gcc_inst_alignment_strength_label(
    consensus_state: str,
    alignment_state: str,
    cross_section: str,
) -> str:
    if alignment_state == "INSTITUTIONAL_CONSENSUS_BROKEN":
        return "NONE"
    if alignment_state in ("GOVERNANCE_INTERNAL_CONFLICT", "FRAGMENTATION_ELEVATED"):
        return "WEAK"
    if alignment_state == "PARTIAL_ALIGNMENT" or cross_section in ("LOW", "MODERATE"):
        return "LIMITED"
    if alignment_state == "GOVERNANCE_CONSENSUS_FORMING":
        return "MODERATE"
    if alignment_state == "INSTITUTIONAL_ALIGNMENT_STRONG":
        return "STRONG"
    if consensus_state in ("STRONG_CONSENSUS", "CONSTITUTIONAL_CONSENSUS"):
        return "STRONG"
    return "LIMITED"


def _gcc_fragmentation_severity(
    alignment_state: str,
    tension: Dict[str, Any],
    coherence: Dict[str, Any],
) -> str:
    count = int(tension.get("contradiction_count", 0) or 0)
    reasoning = coherence.get("reasoning_chain_status", "")

    if alignment_state == "INSTITUTIONAL_CONSENSUS_BROKEN" or count >= 3:
        return "SEVERE"
    if alignment_state in ("GOVERNANCE_INTERNAL_CONFLICT", "FRAGMENTATION_ELEVATED") or count >= 2:
        return "MATERIAL"
    if count >= 1 or reasoning in ("FRAGMENTED", "BROKEN"):
        return "MODERATE"
    if alignment_state == "PARTIAL_ALIGNMENT":
        return "LOW"
    return "MINIMAL"


def _gcc_governance_coherence_label(
    coherence: Dict[str, Any],
    alignment_state: str,
) -> str:
    reasoning = coherence.get("reasoning_chain_status", "")
    narrative = coherence.get("narrative_integrity", "")

    if alignment_state == "INSTITUTIONAL_CONSENSUS_BROKEN" or reasoning == "BROKEN":
        return "BROKEN"
    if alignment_state in ("GOVERNANCE_INTERNAL_CONFLICT", "FRAGMENTATION_ELEVATED"):
        return "WEAK"
    if alignment_state == "PARTIAL_ALIGNMENT" or narrative in ("WEAK", "PARTIAL"):
        return "LIMITED"
    if alignment_state == "GOVERNANCE_CONSENSUS_FORMING":
        return "STABLE"
    if alignment_state == "INSTITUTIONAL_ALIGNMENT_STRONG" or narrative == "HIGHLY_COHERENT":
        return "STRONG"
    return "LIMITED"


def _gcc_escalation_disagreement(
    decision: Dict[str, Any],
    intervention: Dict[str, Any],
) -> bool:
    escalation = decision.get("escalation_readiness", "NONE")
    posture = intervention.get("intervention_posture", "")
    discussable = decision.get("discussability", "NOT_DISCUSSABLE")

    if posture in ("CONSTITUTIONAL_LOCK_REQUIRED", "ESCALATION_REQUIRED") and escalation in (
        "NONE",
        "VERY_LOW",
    ):
        return True
    if posture == "OBSERVATION_ONLY" and escalation in ("MODERATE", "HIGH"):
        return True
    if discussable == "NOT_DISCUSSABLE" and posture == "ESCALATION_REQUIRED":
        return False
    return (
        posture in ("CONTAINMENT_RECOMMENDED", "ESCALATION_REQUIRED")
        and escalation in ("NONE", "VERY_LOW")
        and discussable in ("NOT_DISCUSSABLE", "INTERNAL_OBSERVATION_ONLY")
    )


def _gcc_inst_alignment_drivers(
    alignment_state: str,
    consensus: Dict[str, Any],
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    intervention: Dict[str, Any],
    escalation_disagreement: bool,
) -> List[str]:
    drivers: List[str] = []

    if alignment_state == "INSTITUTIONAL_CONSENSUS_BROKEN":
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Contradictions: {tension.get('contradiction_count', 0)}")
        if escalation_disagreement:
            drivers.append("Escalation disagreement visible")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
    elif alignment_state == "FRAGMENTATION_ELEVATED":
        drivers.append("Governance fragmentation rising")
        drivers.append(f"Contradictions: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Stability pressured: {coherence.get('cross_section_agreement', '—')}")
    elif alignment_state == "INSTITUTIONAL_ALIGNMENT_STRONG":
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Contradictions: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Escalation: {decision.get('escalation_readiness', '—')}")
        drivers.append(f"Intervention: {intervention.get('intervention_display', '—')}")
    else:
        drivers.append(f"Consensus: {consensus.get('consensus_display', '—')}")
        drivers.append(f"Tension: {tension.get('tension_display', '—')}")
        drivers.append(f"Cross-section: {coherence.get('cross_section_agreement', '—')}")
        if escalation_disagreement:
            drivers.append("Escalation disagreement visible")

    return drivers[:6]


def _gcc_detect_governance_institutional_alignment(
    *,
    consensus: Dict[str, Any],
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    evidence: Dict[str, Any],
    intervention: Dict[str, Any],
    hist: Dict[str, Any],
) -> Dict[str, Any]:
    consensus_state = consensus.get("consensus_state", "")
    coherence_state = coherence.get("coherence_state", "")
    reasoning = coherence.get("reasoning_chain_status", "")
    cross_section = coherence.get("cross_section_agreement", "MODERATE")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    tension_level = tension.get("tension_level", "NO_TENSION")
    escalation_disagreement = _gcc_escalation_disagreement(decision, intervention)
    direction = hist.get("confidence_direction", "stable")

    def _match_broken() -> bool:
        return (
            consensus_state == "CONFLICTED_GOVERNANCE"
            or (
                reasoning == "BROKEN"
                and contradiction_count >= 2
                and cross_section in ("VERY_LOW", "LOW")
            )
            or (
                coherence_state
                in (
                    "FRAGMENTED_REASONING_CHAIN",
                    "INTERNALLY_INCONSISTENT_GOVERNANCE",
                )
                and contradiction_count >= 3
            )
        )

    def _match_internal_conflict() -> bool:
        return (
            contradiction_count >= 2
            and (
                consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
                or escalation_disagreement
            )
            and tension_level in ("MODERATE_TENSION", "HIGH_TENSION", "CRITICAL_TENSION")
        )

    def _match_fragmentation_elevated() -> bool:
        return (
            consensus_state in ("FRAGMENTED_GOVERNANCE", "CONFLICTED_GOVERNANCE")
            or coherence_state == "FRAGMENTED_REASONING_CHAIN"
            or contradiction_count >= 2
        )

    def _match_partial() -> bool:
        return (
            cross_section in ("MODERATE", "LOW")
            or consensus_state == "MODERATE_CONSENSUS"
            or (
                contradiction_count <= 2
                and coherence_state
                in (
                    "MODERATELY_COHERENT_GOVERNANCE",
                    "LOW_COHERENCE_GOVERNANCE",
                )
            )
        )

    def _match_forming() -> bool:
        return (
            direction == "improving"
            and contradiction_count <= 1
            and reasoning not in ("BROKEN", "FRAGMENTED")
            and stability.get("stability_state")
            in (
                "GOVERNANCE_STABILITY_IMPROVING",
                "STABLE_GOVERNANCE_POSTURE",
            )
        )

    def _match_strong() -> bool:
        return (
            consensus_state
            in (
                "STRONG_CONSENSUS",
                "CONSTITUTIONAL_CONSENSUS",
                "PRE_RUNTIME_CONVERGENCE",
            )
            and contradiction_count == 0
            and coherence_state
            in (
                "HIGHLY_COHERENT_GOVERNANCE",
                "MODERATELY_COHERENT_GOVERNANCE",
                "LOGICALLY_CONSTRAINED_BUT_COHERENT",
            )
            and cross_section in ("HIGH", "VERY_HIGH")
            and not escalation_disagreement
            and evidence.get("evidence_state")
            in (
                "CONFIDENCE_SUPPORTED",
                "EVIDENCE_INTEGRITY_STRONG",
                "PARTIALLY_SUPPORTED_CONFIDENCE",
            )
        )

    matchers = {
        "INSTITUTIONAL_CONSENSUS_BROKEN": _match_broken,
        "GOVERNANCE_INTERNAL_CONFLICT": _match_internal_conflict,
        "FRAGMENTATION_ELEVATED": _match_fragmentation_elevated,
        "PARTIAL_ALIGNMENT": _match_partial,
        "GOVERNANCE_CONSENSUS_FORMING": _match_forming,
        "INSTITUTIONAL_ALIGNMENT_STRONG": _match_strong,
    }

    alignment_state = "PARTIAL_ALIGNMENT"
    for candidate in _GCC_INST_CONSENSUS_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            alignment_state = candidate
            break

    alignment_strength = _gcc_inst_alignment_strength_label(
        consensus_state, alignment_state, cross_section
    )
    fragmentation = _gcc_fragmentation_severity(alignment_state, tension, coherence)
    coherence_label = _gcc_governance_coherence_label(coherence, alignment_state)
    drivers = _gcc_inst_alignment_drivers(
        alignment_state,
        consensus,
        coherence,
        tension,
        integrity,
        decision,
        intervention,
        escalation_disagreement,
    )
    protocol = _GCC_INST_CONSENSUS_PROTOCOL.get(alignment_state, ["Monitor alignment signals"])

    return {
        "alignment_state": alignment_state,
        "alignment_display": _GCC_INST_CONSENSUS_DISPLAY.get(
            alignment_state, alignment_state.replace("_", " ").title()
        ),
        "alignment_strength": alignment_strength,
        "fragmentation_severity": fragmentation,
        "governance_coherence": coherence_label,
        "drivers": drivers,
        "interpretation": _GCC_INST_CONSENSUS_INTERPRETATION.get(alignment_state, ""),
        "alignment_action": _GCC_INST_CONSENSUS_ACTION.get(
            alignment_state, "MONITOR_ALIGNMENT_SIGNALS"
        ),
        "consensus_protocol": protocol,
        "escalation_disagreement": escalation_disagreement,
    }


def _gcc_render_institutional_alignment_intelligence(
    *,
    consensus: Dict[str, Any],
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    evidence: Dict[str, Any],
    intervention: Dict[str, Any],
    hist: Dict[str, Any],
) -> Dict[str, Any]:
    alignment = _gcc_detect_governance_institutional_alignment(
        consensus=consensus,
        coherence=coherence,
        tension=tension,
        integrity=integrity,
        decision=decision,
        stability=stability,
        resilience=resilience,
        evidence=evidence,
        intervention=intervention,
        hist=hist,
    )

    st.markdown("### Governance Institutional Consensus & Alignment Intelligence")
    st.caption(
        "Institutional alignment analysis — whether governance reasoning is internally consistent. "
        "**Read-only consensus view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consensus State", alignment["alignment_display"])
    c2.metric("Alignment Strength", alignment["alignment_strength"])
    c3.metric("Fragmentation Severity", alignment["fragmentation_severity"])
    c4.metric("Governance Coherence", alignment["governance_coherence"])

    state = alignment["alignment_state"]
    if state == "INSTITUTIONAL_CONSENSUS_BROKEN":
        st.error(alignment["interpretation"])
    elif state in ("GOVERNANCE_INTERNAL_CONFLICT", "FRAGMENTATION_ELEVATED"):
        st.warning(alignment["interpretation"])
    elif state == "INSTITUTIONAL_ALIGNMENT_STRONG":
        st.success(alignment["interpretation"])
    else:
        st.info(alignment["interpretation"])

    st.markdown("**Alignment Drivers**")
    for driver in alignment["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Alignment Action", alignment["alignment_action"])
    a2.metric("Protocol Steps", len(alignment["consensus_protocol"]))

    st.markdown("**Institutional Consensus Protocol**")
    for step in alignment["consensus_protocol"]:
        if state in ("INSTITUTIONAL_CONSENSUS_BROKEN", "GOVERNANCE_INTERNAL_CONFLICT"):
            st.warning(f"• {step}")
        else:
            st.markdown(f'<div class="gcc-block-item">• {step}</div>', unsafe_allow_html=True)

    with st.expander("Alignment analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{alignment['alignment_state']}`")
        st.markdown(f"- **Consensus state:** `{consensus.get('consensus_state', '—')}`")
        st.markdown(f"- **Coherence state:** `{coherence.get('coherence_state', '—')}`")
        st.markdown(
            f"- **Escalation disagreement:** `{'Yes' if alignment['escalation_disagreement'] else 'No'}`"
        )

    return alignment


_GCC_DECISION_QUALITY_PRIORITY: Tuple[str, ...] = (
    "INSTITUTIONAL_REASONING_BROKEN",
    "DECISION_QUALITY_MATERIALLY_IMPAIRED",
    "REASONING_INCONSISTENT",
    "DECISION_QUALITY_RECOVERABLE",
    "INSTITUTIONAL_REASONING_STABILIZING",
    "HIGH_QUALITY_GOVERNANCE_REASONING",
)

_GCC_DECISION_QUALITY_DISPLAY: Dict[str, str] = {
    "INSTITUTIONAL_REASONING_BROKEN": "Institutional Reasoning Broken",
    "DECISION_QUALITY_MATERIALLY_IMPAIRED": "Decision Quality Materially Impaired",
    "REASONING_INCONSISTENT": "Reasoning Inconsistent",
    "DECISION_QUALITY_RECOVERABLE": "Decision Quality Recoverable",
    "INSTITUTIONAL_REASONING_STABILIZING": "Institutional Reasoning Stabilizing",
    "HIGH_QUALITY_GOVERNANCE_REASONING": "High Quality Governance Reasoning",
}

_GCC_DECISION_QUALITY_INTERPRETATION: Dict[str, str] = {
    "INSTITUTIONAL_REASONING_BROKEN": (
        "Governance decision quality appears materially impaired. Fragmented coherence, weak evidence "
        "integrity, and elevated contradictions constrain institutional confidence in governance reasoning."
    ),
    "DECISION_QUALITY_MATERIALLY_IMPAIRED": (
        "Governance conclusions remain weak under institutional scrutiny. Reasoning quality and "
        "defensibility are materially constrained."
    ),
    "REASONING_INCONSISTENT": (
        "Governance reasoning appears inconsistent. Institutional logic and escalation posture remain "
        "misaligned across governance layers."
    ),
    "DECISION_QUALITY_RECOVERABLE": (
        "Governance decision quality remains weakened but recoverable. Institutional coherence and "
        "defensibility remain sufficient to justify continued observation."
    ),
    "INSTITUTIONAL_REASONING_STABILIZING": (
        "Governance reasoning appears to be stabilizing. Coherence and institutional quality signals "
        "suggest decision quality may be improving."
    ),
    "HIGH_QUALITY_GOVERNANCE_REASONING": (
        "Governance reasoning appears institutionally strong. Decision quality, evidence integrity, "
        "and institutional coherence remain aligned under scrutiny."
    ),
}

_GCC_DECISION_QUALITY_ACTION: Dict[str, str] = {
    "INSTITUTIONAL_REASONING_BROKEN": "PRESERVE_CONSTITUTIONAL_LOCK",
    "DECISION_QUALITY_MATERIALLY_IMPAIRED": "IMPROVE_REASONING_INTEGRITY",
    "REASONING_INCONSISTENT": "IMPROVE_GOVERNANCE_COHERENCE",
    "DECISION_QUALITY_RECOVERABLE": "MONITOR_REASONING_QUALITY",
    "INSTITUTIONAL_REASONING_STABILIZING": "CONTINUE_OBSERVATION",
    "HIGH_QUALITY_GOVERNANCE_REASONING": "CONTINUE_OBSERVATION",
}

_GCC_DECISION_QUALITY_PROTOCOL: Dict[str, List[str]] = {
    "INSTITUTIONAL_REASONING_BROKEN": [
        "Improve governance coherence",
        "Monitor contradiction severity",
        "Reassess institutional defensibility",
        "Monitor evidence integrity deterioration",
    ],
    "DECISION_QUALITY_MATERIALLY_IMPAIRED": [
        "Improve reasoning integrity",
        "Monitor evidence integrity deterioration",
        "Reassess institutional defensibility",
    ],
    "REASONING_INCONSISTENT": [
        "Observe reasoning inconsistency",
        "Improve governance coherence",
        "Monitor contradiction severity",
    ],
    "DECISION_QUALITY_RECOVERABLE": [
        "Monitor reasoning quality",
        "Continue observation if reasoning stabilizes",
        "Monitor contradiction severity",
    ],
    "INSTITUTIONAL_REASONING_STABILIZING": [
        "Continue observation if reasoning stabilizes",
        "Monitor reasoning quality",
    ],
    "HIGH_QUALITY_GOVERNANCE_REASONING": [
        "Continue observation",
        "Monitor routine reasoning signals",
    ],
}


def _gcc_reasoning_integrity_label(
    quality_state: str,
    coherence: Dict[str, Any],
    alignment: Dict[str, Any],
) -> str:
    reasoning = coherence.get("reasoning_chain_status", "")
    if quality_state == "INSTITUTIONAL_REASONING_BROKEN" or reasoning == "BROKEN":
        return "BROKEN"
    if quality_state in ("DECISION_QUALITY_MATERIALLY_IMPAIRED", "REASONING_INCONSISTENT"):
        return "WEAK"
    if (
        quality_state == "DECISION_QUALITY_RECOVERABLE"
        or alignment.get("governance_coherence") == "LIMITED"
    ):
        return "LIMITED"
    if quality_state == "INSTITUTIONAL_REASONING_STABILIZING":
        return "MODERATE"
    if quality_state == "HIGH_QUALITY_GOVERNANCE_REASONING":
        return "STRONG"
    return "LIMITED"


def _gcc_decision_defensibility(
    quality_state: str,
    evidence: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    base = evidence.get("institutional_defensibility", "LIMITED")
    if quality_state == "INSTITUTIONAL_REASONING_BROKEN":
        return "NONE"
    if quality_state == "DECISION_QUALITY_MATERIALLY_IMPAIRED":
        return "WEAK"
    if quality_state == "HIGH_QUALITY_GOVERNANCE_REASONING":
        return "STRONG"
    if float(decision.get("governance_maturity_score", 0.0) or 0.0) >= 0.50 and base == "MODERATE":
        return "MODERATE"
    return base


def _gcc_stability_under_scrutiny(
    quality_state: str,
    evidence: Dict[str, Any],
    stability: Dict[str, Any],
    alignment: Dict[str, Any],
) -> str:
    if quality_state == "INSTITUTIONAL_REASONING_BROKEN":
        return "UNSTABLE"
    if quality_state in ("DECISION_QUALITY_MATERIALLY_IMPAIRED", "REASONING_INCONSISTENT"):
        return "FRAGILE"
    if quality_state == "DECISION_QUALITY_RECOVERABLE":
        return "CONDITIONAL"
    if quality_state == "INSTITUTIONAL_REASONING_STABILIZING":
        return "STABLE"
    if quality_state == "HIGH_QUALITY_GOVERNANCE_REASONING":
        return "RESILIENT"
    if evidence.get("confidence_reliability") in ("VERY_LOW", "LOW"):
        return "FRAGILE"
    if float(stability.get("stability_score", 0.0) or 0.0) >= 0.55:
        return "STABLE"
    return "CONDITIONAL"


def _gcc_decision_quality_drivers(
    quality_state: str,
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    decision: Dict[str, Any],
    stability: Dict[str, Any],
) -> List[str]:
    drivers: List[str] = []

    if quality_state == "INSTITUTIONAL_REASONING_BROKEN":
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Contradictions: {tension.get('contradiction_count', 0)}")
        drivers.append(f"Trustworthiness: {_gcc_fmt_conf(integrity.get('discounted_trust_score'))}")
        drivers.append(f"Evidence: {evidence.get('evidence_display', '—')}")
        drivers.append(f"Defensibility: {evidence.get('institutional_defensibility', '—')}")
    elif quality_state == "REASONING_INCONSISTENT":
        if alignment.get("escalation_disagreement"):
            drivers.append("Escalation posture inconsistent")
        drivers.append(f"Alignment: {alignment.get('alignment_display', '—')}")
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Consensus alignment: {alignment.get('alignment_strength', '—')}")
    elif quality_state == "HIGH_QUALITY_GOVERNANCE_REASONING":
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Evidence: {evidence.get('evidence_display', '—')}")
        drivers.append(f"Alignment: {alignment.get('alignment_display', '—')}")
        drivers.append(f"Maturity: {_gcc_fmt_conf(decision.get('governance_maturity_score'))}")
    else:
        drivers.append(f"Coherence: {coherence.get('coherence_display', '—')}")
        drivers.append(f"Evidence reliability: {evidence.get('confidence_reliability', '—')}")
        drivers.append(f"Alignment: {alignment.get('alignment_display', '—')}")
        drivers.append(f"Stability score: {_gcc_fmt_conf(stability.get('stability_score'))}")

    return drivers[:6]


def _gcc_detect_governance_decision_quality(
    *,
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    intervention: Dict[str, Any],
    improvement: Dict[str, Any],
    hist: Dict[str, Any],
) -> Dict[str, Any]:
    reasoning_chain = coherence.get("reasoning_chain_status", "")
    coherence_state = coherence.get("coherence_state", "")
    alignment_state = alignment.get("alignment_state", "")
    evidence_state = evidence.get("evidence_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    logic_score = float(coherence.get("logic_score", 0.0) or 0.0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    direction = hist.get("confidence_direction", "stable")
    escalation_disagreement = alignment.get("escalation_disagreement", False)

    def _match_broken() -> bool:
        return (
            alignment_state == "INSTITUTIONAL_CONSENSUS_BROKEN"
            or (
                reasoning_chain == "BROKEN"
                and evidence_state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED"
                and contradiction_count >= 2
            )
            or (
                coherence_state
                in (
                    "FRAGMENTED_REASONING_CHAIN",
                    "INTERNALLY_INCONSISTENT_GOVERNANCE",
                )
                and evidence.get("institutional_defensibility") in ("NONE", "WEAK")
            )
        )

    def _match_impaired() -> bool:
        return evidence_state in (
            "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED",
            "TRUSTWORTHINESS_MATERIALLY_IMPAIRED",
        ) or (logic_score < 0.25 and evidence.get("confidence_reliability") in ("VERY_LOW", "LOW"))

    def _match_inconsistent() -> bool:
        return (
            escalation_disagreement
            or alignment_state in ("GOVERNANCE_INTERNAL_CONFLICT", "FRAGMENTATION_ELEVATED")
            or coherence_state == "INTERNALLY_INCONSISTENT_GOVERNANCE"
        )

    def _match_recoverable() -> bool:
        return (
            resilience.get("resilience_state")
            in (
                "RECOVERABLE_FRAGMENTATION",
                "REVERSIBLE_GOVERNANCE_DETERIORATION",
                "MODERATE_GOVERNANCE_RESILIENCE",
            )
            and alignment_state in ("PARTIAL_ALIGNMENT", "FRAGMENTATION_ELEVATED")
            and evidence_state
            in (
                "PARTIALLY_SUPPORTED_CONFIDENCE",
                "EVIDENCE_QUALITY_DEGRADING",
            )
        )

    def _match_stabilizing() -> bool:
        return (
            alignment_state == "GOVERNANCE_CONSENSUS_FORMING"
            or improvement.get("improvement_state") == "GOVERNANCE_ADAPTATION_EMERGING"
            or (
                direction == "improving"
                and contradiction_count <= 1
                and reasoning_chain not in ("BROKEN",)
            )
        )

    def _match_high_quality() -> bool:
        return (
            alignment_state == "INSTITUTIONAL_ALIGNMENT_STRONG"
            and evidence_state in ("CONFIDENCE_SUPPORTED", "EVIDENCE_INTEGRITY_STRONG")
            and logic_score >= 0.50
            and contradiction_count == 0
            and maturity >= 0.40
            and intervention.get("intervention_posture") == "OBSERVATION_ONLY"
        )

    matchers = {
        "INSTITUTIONAL_REASONING_BROKEN": _match_broken,
        "DECISION_QUALITY_MATERIALLY_IMPAIRED": _match_impaired,
        "REASONING_INCONSISTENT": _match_inconsistent,
        "DECISION_QUALITY_RECOVERABLE": _match_recoverable,
        "INSTITUTIONAL_REASONING_STABILIZING": _match_stabilizing,
        "HIGH_QUALITY_GOVERNANCE_REASONING": _match_high_quality,
    }

    quality_state = "DECISION_QUALITY_RECOVERABLE"
    for candidate in _GCC_DECISION_QUALITY_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            quality_state = candidate
            break

    reasoning_integrity = _gcc_reasoning_integrity_label(quality_state, coherence, alignment)
    defensibility = _gcc_decision_defensibility(quality_state, evidence, decision)
    scrutiny = _gcc_stability_under_scrutiny(quality_state, evidence, stability, alignment)
    drivers = _gcc_decision_quality_drivers(
        quality_state, coherence, tension, integrity, evidence, alignment, decision, stability
    )
    protocol = _GCC_DECISION_QUALITY_PROTOCOL.get(quality_state, ["Monitor reasoning quality"])

    return {
        "quality_state": quality_state,
        "quality_display": _GCC_DECISION_QUALITY_DISPLAY.get(
            quality_state, quality_state.replace("_", " ").title()
        ),
        "reasoning_integrity": reasoning_integrity,
        "institutional_defensibility": defensibility,
        "stability_under_scrutiny": scrutiny,
        "drivers": drivers,
        "interpretation": _GCC_DECISION_QUALITY_INTERPRETATION.get(quality_state, ""),
        "reasoning_action": _GCC_DECISION_QUALITY_ACTION.get(
            quality_state, "MONITOR_REASONING_QUALITY"
        ),
        "decision_protocol": protocol,
    }


def _gcc_render_decision_quality_intelligence(
    *,
    coherence: Dict[str, Any],
    tension: Dict[str, Any],
    integrity: Dict[str, Any],
    decision: Dict[str, Any],
    stability: Dict[str, Any],
    resilience: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    intervention: Dict[str, Any],
    improvement: Dict[str, Any],
    hist: Dict[str, Any],
) -> None:
    quality = _gcc_detect_governance_decision_quality(
        coherence=coherence,
        tension=tension,
        integrity=integrity,
        decision=decision,
        stability=stability,
        resilience=resilience,
        evidence=evidence,
        alignment=alignment,
        intervention=intervention,
        improvement=improvement,
        hist=hist,
    )

    st.markdown("### Governance Institutional Decision Quality & Reasoning Integrity Intelligence")
    st.caption(
        "Decision quality analysis — whether governance conclusions are institutionally sound under scrutiny. "
        "**Read-only reasoning view. Not runtime enablement.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision Quality State", quality["quality_display"])
    c2.metric("Reasoning Integrity", quality["reasoning_integrity"])
    c3.metric("Institutional Defensibility", quality["institutional_defensibility"])
    c4.metric("Stability Under Scrutiny", quality["stability_under_scrutiny"])

    state = quality["quality_state"]
    if state == "INSTITUTIONAL_REASONING_BROKEN":
        st.error(quality["interpretation"])
    elif state in ("DECISION_QUALITY_MATERIALLY_IMPAIRED", "REASONING_INCONSISTENT"):
        st.warning(quality["interpretation"])
    elif state == "HIGH_QUALITY_GOVERNANCE_REASONING":
        st.success(quality["interpretation"])
    else:
        st.info(quality["interpretation"])

    st.markdown("**Decision Drivers**")
    for driver in quality["drivers"]:
        st.markdown(f'<div class="gcc-hist-metric">• {driver}</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    a1.metric("Recommended Reasoning Action", quality["reasoning_action"])
    a2.metric("Protocol Steps", len(quality["decision_protocol"]))

    st.markdown("**Institutional Decision Integrity Protocol**")
    for step in quality["decision_protocol"]:
        if state in ("INSTITUTIONAL_REASONING_BROKEN", "DECISION_QUALITY_MATERIALLY_IMPAIRED"):
            st.warning(f"• {step}")
        else:
            st.markdown(f'<div class="gcc-block-item">• {step}</div>', unsafe_allow_html=True)

    with st.expander("Decision quality analysis detail", expanded=False):
        st.markdown(f"- **Internal state:** `{quality['quality_state']}`")
        st.markdown(f"- **Coherence state:** `{coherence.get('coherence_state', '—')}`")
        st.markdown(f"- **Evidence state:** `{evidence.get('evidence_state', '—')}`")
        st.markdown(f"- **Alignment state:** `{alignment.get('alignment_state', '—')}`")


_GCC_EXECUTIVE_VERDICT_PRIORITY: Tuple[str, ...] = (
    "CONSTITUTIONALLY_LOCKED_GOVERNANCE",
    "SYSTEMIC_GOVERNANCE_INSTABILITY",
    "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS",
    "GOVERNANCE_STABILIZING",
    "GOVERNANCE_OPERATIONALLY_STABLE",
)

_GCC_EXECUTIVE_VERDICT_DISPLAY: Dict[str, str] = {
    "CONSTITUTIONALLY_LOCKED_GOVERNANCE": "Constitutionally Locked Governance",
    "SYSTEMIC_GOVERNANCE_INSTABILITY": "Systemic Governance Instability",
    "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS": "Governance Under Institutional Stress",
    "GOVERNANCE_STABILIZING": "Governance Stabilizing",
    "GOVERNANCE_OPERATIONALLY_STABLE": "Governance Operationally Stable",
}

_GCC_EXECUTIVE_VERDICT_MEMO: Dict[str, str] = {
    "CONSTITUTIONALLY_LOCKED_GOVERNANCE": (
        "Governance remains constitutionally constrained. Fragmented institutional alignment, "
        "weak trustworthiness, and impaired reasoning quality continue to prevent runtime governance "
        "readiness. Constitutional safeguards remain appropriate and escalation is not presently justified."
    ),
    "SYSTEMIC_GOVERNANCE_INSTABILITY": (
        "Governance exhibits systemic institutional instability. Fragmented coherence, weak evidence "
        "integrity, and defensive intervention posture constrain operator confidence. Containment "
        "and observation remain the appropriate institutional response."
    ),
    "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS": (
        "Governance is under institutional stress. Contradictions, weakened alignment, and constrained "
        "defensibility warrant heightened monitoring while constitutional safeguards remain in place."
    ),
    "GOVERNANCE_STABILIZING": (
        "Governance shows stabilizing signals. Institutional quality is improving, though runtime "
        "readiness has not yet been demonstrated. Continued observation remains appropriate."
    ),
    "GOVERNANCE_OPERATIONALLY_STABLE": (
        "Governance appears operationally stable. Coherence, trustworthiness, and alignment remain "
        "sufficiently strong for continued institutional observation."
    ),
}

_GCC_EXECUTIVE_OPERATOR_ACTION: Dict[str, str] = {
    "CONSTITUTIONALLY_LOCKED_GOVERNANCE": "MAINTAIN_CONSTITUTIONAL_LOCK",
    "SYSTEMIC_GOVERNANCE_INSTABILITY": "REDUCE_FRAGMENTATION",
    "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS": "MONITOR_GOVERNANCE_STRESS",
    "GOVERNANCE_STABILIZING": "CONTINUE_OBSERVATION",
    "GOVERNANCE_OPERATIONALLY_STABLE": "CONTINUE_OBSERVATION",
}

_GCC_SEVERITY_PRIORITY: Tuple[str, ...] = (
    "CRITICAL_LOCK",
    "HIGH_RISK",
    "ELEVATED",
    "WATCH",
    "NORMAL",
)

_GCC_SEVERITY_DISPLAY: Dict[str, str] = {
    "CRITICAL_LOCK": "Critical Lock",
    "HIGH_RISK": "High Risk",
    "ELEVATED": "Elevated",
    "WATCH": "Watch",
    "NORMAL": "Normal",
}

_GCC_ATTENTION_MEMO: Dict[str, str] = {
    "CRITICAL_LOCK": (
        "Governance remains constitutionally locked. Runtime posture is blocked and no immediate "
        "intervention beyond normal governance refresh is required. Maintain constitutional "
        "safeguards and avoid escalation fatigue."
    ),
    "HIGH_RISK": (
        "Governance quality remains materially impaired and deserves elevated operator awareness. "
        "Intraday review is appropriate while governance fragmentation and trust deterioration "
        "are monitored."
    ),
    "ELEVATED": (
        "Governance stress is visible and warrants active monitoring. Routine daily review remains "
        "appropriate while constitutional safeguards stay in place."
    ),
    "WATCH": (
        "Governance shows mild deterioration signals. Operator awareness is appropriate; routine "
        "monitoring remains sufficient unless stress indicators worsen."
    ),
    "NORMAL": (
        "Governance posture remains stable and institutionally healthy. Routine monitoring "
        "remains sufficient."
    ),
}

_GCC_DIRECTION_PRIORITY: Tuple[str, ...] = (
    "RAPIDLY_DETERIORATING",
    "DETERIORATING",
    "STRESSED_BUT_STABLE",
    "STABILIZING",
    "IMPROVING",
    "OPERATIONALLY_STABLE",
)

_GCC_DIRECTION_DISPLAY: Dict[str, str] = {
    "RAPIDLY_DETERIORATING": "Rapidly Deteriorating",
    "DETERIORATING": "Deteriorating",
    "STRESSED_BUT_STABLE": "Stressed but Stable",
    "STABILIZING": "Stabilizing",
    "IMPROVING": "Improving",
    "OPERATIONALLY_STABLE": "Operationally Stable",
}

_GCC_TREND_MEMO: Dict[str, str] = {
    "RAPIDLY_DETERIORATING": (
        "Governance quality appears to be deteriorating rapidly. Fragmentation, trust "
        "deterioration, and contradiction persistence justify closer observation while "
        "constitutional safeguards remain active."
    ),
    "DETERIORATING": (
        "Governance trajectory is weakening. Institutional quality, coherence, or trust "
        "signals are declining and warrant elevated monitoring."
    ),
    "STRESSED_BUT_STABLE": (
        "Governance remains impaired but directional deterioration is not accelerating. "
        "Constitutional safeguards remain appropriate and routine governance refresh remains "
        "sufficient."
    ),
    "STABILIZING": (
        "Governance shows stabilizing directional signals. Contradictions may be declining "
        "and institutional posture appears to be recovering gradually."
    ),
    "IMPROVING": (
        "Governance trajectory appears favorable. Institutional coherence and trust signals "
        "show gradual improvement."
    ),
    "OPERATIONALLY_STABLE": (
        "Governance direction remains stable with low drift. Institutional quality appears "
        "durable and routine observation remains sufficient."
    ),
}

_GCC_COCKPIT_CONFIDENCE_PRIORITY: Tuple[str, ...] = (
    "VERY_LOW",
    "LOW",
    "GUARDED",
    "MODERATE",
    "HIGH",
    "VERY_HIGH",
)

_GCC_COCKPIT_CONFIDENCE_DISPLAY: Dict[str, str] = {
    "VERY_LOW": "Very Low",
    "LOW": "Low",
    "GUARDED": "Guarded",
    "MODERATE": "Moderate",
    "HIGH": "High",
    "VERY_HIGH": "Very High",
}

_GCC_COCKPIT_CONFIDENCE_MEMO: Dict[str, str] = {
    "VERY_LOW": (
        "Governance interpretation confidence remains weak. Fragmented reasoning, impaired "
        "evidence integrity, and institutional inconsistency reduce confidence in the "
        "cockpit's conclusions."
    ),
    "LOW": (
        "Governance cockpit interpretation appears fragile. Weak trust signals and constrained "
        "evidence warrant caution when acting on governance conclusions."
    ),
    "GUARDED": (
        "Governance interpretation remains conditionally reliable. Institutional signals remain "
        "usable, though continued observation and caution are warranted."
    ),
    "MODERATE": (
        "Governance cockpit interpretation appears reasonably stable. Evidence and directional "
        "signals are sufficiently aligned for routine institutional use."
    ),
    "HIGH": (
        "Governance cockpit interpretation appears institutionally reliable. Evidence integrity, "
        "coherence, and directional consistency remain aligned."
    ),
    "VERY_HIGH": (
        "Governance cockpit appears institutionally trustworthy. Interpretation is stable, "
        "signal reliability is strong, and governance conclusions are well supported."
    ),
}

_GCC_PLAYBOOK_DISCIPLINE_MEMO: Dict[str, str] = {
    "MAINTAIN_CONSTITUTIONAL_LOCK": (
        "Governance remains impaired but stable. Operators should maintain constitutional "
        "safeguards, avoid premature escalation, and continue disciplined observation during "
        "scheduled refresh windows."
    ),
    "REDUCE_FRAGMENTATION": (
        "Governance fragmentation warrants elevated discipline. Observation intensity should "
        "increase while alignment and consensus signals are reassessed for stabilization."
    ),
    "IMPROVE_CONFIDENCE_INTEGRITY": (
        "Evidence and confidence integrity require operator attention. Continue observation "
        "while trustworthiness and auditability signals are monitored for recovery."
    ),
    "MONITOR_GOVERNANCE_STRESS": (
        "Governance remains under institutional stress. Maintain heightened monitoring while "
        "constitutional safeguards stay in place and escalation remains deferred."
    ),
    "CONTINUE_DISCIPLINED_OBSERVATION": (
        "Governance posture supports continued disciplined observation. Routine monitoring "
        "remains appropriate unless stress or drift indicators worsen."
    ),
    "PREPARE_ESCALATION_REVIEW": (
        "Governance deterioration warrants elevated discipline. Observation intensity should "
        "increase while escalation readiness and fragmentation risks are reassessed."
    ),
}

_GCC_PERSISTENCE_PRIORITY: Tuple[str, ...] = (
    "ENTRENCHED_INSTABILITY",
    "PERSISTENT_LOCK",
    "TRANSITIONING",
    "STABILIZING",
    "SHORT_TERM_VARIATION",
    "OPERATIONALLY_DURABLE",
)

_GCC_PERSISTENCE_DISPLAY: Dict[str, str] = {
    "ENTRENCHED_INSTABILITY": "Entrenched Instability",
    "PERSISTENT_LOCK": "Persistent Lock",
    "TRANSITIONING": "Transitioning",
    "STABILIZING": "Stabilizing",
    "SHORT_TERM_VARIATION": "Short-Term Variation",
    "OPERATIONALLY_DURABLE": "Operationally Durable",
}

_GCC_TEMPORAL_MEMO: Dict[str, str] = {
    "ENTRENCHED_INSTABILITY": (
        "Governance deterioration appears persistent and institutionally difficult to resolve. "
        "Elevated observation and escalation readiness remain appropriate."
    ),
    "PERSISTENT_LOCK": (
        "Governance remains constitutionally constrained but directionally stable. Recovery "
        "appears gradual and operators should maintain disciplined observation without "
        "escalation fatigue."
    ),
    "TRANSITIONING": (
        "Governance posture appears to be in transition. Directional movement is visible and "
        "operators should monitor regime shifts during scheduled review windows."
    ),
    "STABILIZING": (
        "Governance posture appears to be improving gradually. Institutional recovery signals "
        "remain early but observable."
    ),
    "SHORT_TERM_VARIATION": (
        "Governance disturbance appears limited in duration. Routine observation remains "
        "appropriate unless persistence indicators worsen."
    ),
    "OPERATIONALLY_DURABLE": (
        "Governance posture appears operationally durable. Institutional consistency remains "
        "stable and routine monitoring remains sufficient."
    ),
}

_GCC_DELTA_PRIORITY: Tuple[str, ...] = (
    "REGIME_SHIFT_EMERGING",
    "MATERIAL_DETERIORATION",
    "MATERIAL_IMPROVEMENT",
    "MIXED_TRANSITION",
    "STABLE_NO_MATERIAL_CHANGE",
)

_GCC_DELTA_DISPLAY: Dict[str, str] = {
    "REGIME_SHIFT_EMERGING": "Regime Shift Emerging",
    "MATERIAL_DETERIORATION": "Material Deterioration",
    "MATERIAL_IMPROVEMENT": "Material Improvement",
    "MIXED_TRANSITION": "Mixed Transition",
    "STABLE_NO_MATERIAL_CHANGE": "Stable - No Material Change",
}

_GCC_DELTA_CHANGE_MEMO: Dict[str, str] = {
    "REGIME_SHIFT_EMERGING": (
        "Governance appears to be entering a material institutional transition. Directional "
        "movement is significant and operators should monitor regime shift signals closely."
    ),
    "MATERIAL_DETERIORATION": (
        "Governance quality appears to be weakening materially. Elevated observation is "
        "warranted as trust, coherence, or institutional consistency deteriorate."
    ),
    "MATERIAL_IMPROVEMENT": (
        "Governance quality appears to be improving materially. Institutional repair signals "
        "are visible though continued observation remains appropriate."
    ),
    "MIXED_TRANSITION": (
        "Governance signals remain mixed. Institutional change is observable but insufficiently "
        "coherent to justify strong directional conclusions."
    ),
    "STABLE_NO_MATERIAL_CHANGE": (
        "Governance posture remains materially unchanged. Institutional direction, constitutional "
        "safeguards, and governance trajectory remain stable without evidence of regime transition."
    ),
}

_GCC_POSITIVE_CHANGE_DISPLAY: Dict[str, str] = {
    "NONE": "None",
    "TRUST_IMPROVEMENT": "Trust Improvement",
    "ALIGNMENT_IMPROVEMENT": "Alignment Improvement",
    "COHERENCE_STABILIZATION": "Coherence Stabilization",
    "CONTRADICTION_REDUCTION": "Contradiction Reduction",
    "EVIDENCE_STRENGTHENING": "Evidence Strengthening",
    "GOVERNANCE_STABILIZATION": "Governance Stabilization",
}

_GCC_NEGATIVE_CHANGE_DISPLAY: Dict[str, str] = {
    "NONE": "None",
    "TRUST_DETERIORATION": "Trust Deterioration",
    "ALIGNMENT_FRAGMENTATION": "Alignment Fragmentation",
    "COHERENCE_BREAKDOWN": "Coherence Breakdown",
    "CONTRADICTION_ACCELERATION": "Contradiction Acceleration",
    "EVIDENCE_DEGRADATION": "Evidence Degradation",
    "GOVERNANCE_INSTABILITY": "Governance Instability",
}

_GCC_FORWARD_OUTLOOK_PRIORITY: Tuple[str, ...] = (
    "STRUCTURAL_DETERIORATION_RISK",
    "PERSISTENT_CONSTRAINT",
    "MIXED_FORWARD_PATH",
    "STABILIZATION_PATH",
    "OPERATIONALLY_STABLE_PATH",
)

_GCC_FORWARD_OUTLOOK_DISPLAY: Dict[str, str] = {
    "STRUCTURAL_DETERIORATION_RISK": "Structural Deterioration Risk",
    "PERSISTENT_CONSTRAINT": "Persistent Constraint",
    "MIXED_FORWARD_PATH": "Mixed Forward Path",
    "STABILIZATION_PATH": "Stabilization Path",
    "OPERATIONALLY_STABLE_PATH": "Operationally Stable Path",
}

_GCC_FORWARD_FORECAST_MEMO: Dict[str, str] = {
    "STRUCTURAL_DETERIORATION_RISK": (
        "Governance deterioration risk appears elevated. Increased observation of fragmentation, "
        "trust deterioration, and institutional instability is warranted."
    ),
    "PERSISTENT_CONSTRAINT": (
        "Governance is expected to remain constitutionally constrained in the near term. "
        "Institutional deterioration does not appear to be accelerating, and disciplined "
        "observation remains appropriate."
    ),
    "MIXED_FORWARD_PATH": (
        "Governance forward trajectory remains uncertain. Mixed stabilization and deterioration "
        "signals warrant guarded observation without strong directional assumptions."
    ),
    "STABILIZATION_PATH": (
        "Governance posture shows signs of gradual stabilization. Institutional recovery signals "
        "remain early but observable."
    ),
    "OPERATIONALLY_STABLE_PATH": (
        "Governance appears likely to remain operationally stable. Institutional durability "
        "remains sufficient for routine forward observation."
    ),
}

_GCC_SCENARIO_PRIORITY: Tuple[str, ...] = (
    "CONSTITUTIONAL_CONSTRAINT_SCENARIO",
    "FRAGMENTATION_STRESS_SCENARIO",
    "REGIME_TRANSITION_SCENARIO",
    "STABILIZATION_SCENARIO",
    "OPERATIONALLY_STABLE_SCENARIO",
)

_GCC_SCENARIO_DISPLAY: Dict[str, str] = {
    "CONSTITUTIONAL_CONSTRAINT_SCENARIO": "Constitutional Constraint",
    "FRAGMENTATION_STRESS_SCENARIO": "Fragmentation Stress",
    "REGIME_TRANSITION_SCENARIO": "Regime Transition",
    "STABILIZATION_SCENARIO": "Stabilization",
    "OPERATIONALLY_STABLE_SCENARIO": "Operationally Stable",
}

_GCC_NEXT_REGIME_DISPLAY: Dict[str, str] = {
    "CONSTITUTIONAL_LOCK_PERSISTS": "Constitutional Lock Persists",
    "FRAGMENTATION_ELEVATES": "Fragmentation Elevates",
    "GOVERNANCE_STABILIZES": "Governance Stabilizes",
    "MIXED_TRANSITION": "Mixed Transition",
    "OPERATIONAL_STABILITY": "Operational Stability",
}

_GCC_SCENARIO_MEMO: Dict[str, str] = {
    "CONSTITUTIONAL_CONSTRAINT_SCENARIO": (
        "Governance is expected to remain constitutionally constrained in the near term. "
        "Operators should monitor contradiction persistence and trust drift while avoiding "
        "escalation fatigue."
    ),
    "FRAGMENTATION_STRESS_SCENARIO": (
        "Governance fragmentation stress remains material. Operators should monitor alignment, "
        "coherence, and contradiction persistence for further deterioration."
    ),
    "REGIME_TRANSITION_SCENARIO": (
        "Governance appears to be entering a transition phase. Institutional deterioration and "
        "stabilization signals remain mixed and continued observation is warranted."
    ),
    "STABILIZATION_SCENARIO": (
        "Governance stabilization signals are gradually improving. Institutional repair remains "
        "early but increasingly observable."
    ),
    "OPERATIONALLY_STABLE_SCENARIO": (
        "Governance appears operationally stable. Institutional posture remains durable and "
        "material regime transition appears unlikely."
    ),
}

_GCC_DECISION_BRIEF_PRIORITY: Tuple[str, ...] = (
    "LOCKED_OBSERVE_ONLY",
    "LOCKED_HEIGHTENED_MONITORING",
    "GOVERNANCE_REPAIR_REQUIRED",
    "TRANSITION_WATCH",
    "STABLE_CONTINUE_MONITORING",
)

_GCC_DECISION_BRIEF_DISPLAY: Dict[str, str] = {
    "LOCKED_OBSERVE_ONLY": "Locked Observe Only",
    "LOCKED_HEIGHTENED_MONITORING": "Locked Heightened Monitoring",
    "GOVERNANCE_REPAIR_REQUIRED": "Governance Repair Required",
    "TRANSITION_WATCH": "Transition Watch",
    "STABLE_CONTINUE_MONITORING": "Stable Continue Monitoring",
}

_GCC_GOVERNANCE_MODE_DISPLAY: Dict[str, str] = {
    "CONSTITUTIONAL_LOCK_MODE": "Constitutional Lock Mode",
    "CONTAINMENT_MODE": "Containment Mode",
    "REPAIR_MODE": "Repair Mode",
    "TRANSITION_MODE": "Transition Mode",
    "OBSERVATION_MODE": "Observation Mode",
}

_GCC_DECISION_INSTRUCTION: Dict[str, str] = {
    "LOCKED_OBSERVE_ONLY": "MAINTAIN_LOCK_AND_OBSERVE",
    "LOCKED_HEIGHTENED_MONITORING": "MONITOR_TRIGGER_CONDITIONS",
    "GOVERNANCE_REPAIR_REQUIRED": "REVIEW_GOVERNANCE_REPAIR",
    "TRANSITION_WATCH": "PREPARE_ESCALATION_REVIEW",
    "STABLE_CONTINUE_MONITORING": "CONTINUE_ROUTINE_MONITORING",
}

_GCC_DECISION_BRIEF_MEMO: Dict[str, str] = {
    "LOCKED_OBSERVE_ONLY": (
        "Governance remains constitutionally locked but directionally stable. Operators should "
        "maintain safeguards, avoid runtime enablement, and continue observation at the next refresh."
    ),
    "LOCKED_HEIGHTENED_MONITORING": (
        "Governance remains locked with elevated stress signals. Operators should monitor trigger "
        "conditions and reassess if trust, coherence, or contradiction signals deteriorate."
    ),
    "GOVERNANCE_REPAIR_REQUIRED": (
        "Governance quality requires institutional repair. Runtime enablement remains inappropriate "
        "while confidence, coherence, or alignment signals remain impaired."
    ),
    "TRANSITION_WATCH": (
        "Governance may be entering a regime transition. Operators should monitor trigger conditions "
        "and maintain escalation readiness without premature action."
    ),
    "STABLE_CONTINUE_MONITORING": (
        "Governance appears stable. Routine monitoring remains sufficient and no escalation posture "
        "is required."
    ),
}


def _gcc_build_governance_intelligence_stack(
    *,
    readiness: Dict[str, Any],
    admission: Dict[str, Any],
    eligibility: Dict[str, Any],
    recommendation: Dict[str, Any],
    review: Dict[str, Any],
    verdict: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    dossier_record: Dict[str, Any],
) -> Dict[str, Any]:
    common = dict(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    hist = _gcc_analyze_governance_history(**common)
    regime = _gcc_detect_governance_regime(**common, hist=hist)
    forecast = _gcc_detect_governance_forecast(**common, hist=hist, regime=regime)
    tension = _gcc_detect_governance_tensions(**common, hist=hist, regime=regime, forecast=forecast)
    consensus = _gcc_detect_governance_consensus(
        **common, hist=hist, regime=regime, forecast=forecast, tension=tension
    )
    integrity = _gcc_detect_confidence_integrity(
        **common, hist=hist, regime=regime, forecast=forecast, tension=tension, consensus=consensus
    )
    decision = _gcc_detect_decision_readiness(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
    )
    failure = _gcc_detect_governance_failure_modes(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
    )
    audit = _gcc_detect_governance_auditability(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
    )
    coherence = _gcc_detect_governance_coherence(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
    )
    stability = _gcc_detect_governance_stability(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
    )
    resilience = _gcc_detect_governance_resilience(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
    )
    improvement = _gcc_detect_governance_improvement(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
    )
    scenario = _gcc_detect_governance_failure_scenario(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
        improvement=improvement,
    )
    intervention = _gcc_detect_governance_intervention(
        **common,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
        improvement=improvement,
        scenario=scenario,
    )
    evidence = _gcc_detect_governance_institutional_evidence(
        integrity=integrity,
        audit=audit,
        coherence=coherence,
        tension=tension,
        decision=decision,
        stability=stability,
        resilience=resilience,
        intervention=intervention,
    )
    alignment = _gcc_detect_governance_institutional_alignment(
        consensus=consensus,
        coherence=coherence,
        tension=tension,
        integrity=integrity,
        decision=decision,
        stability=stability,
        resilience=resilience,
        evidence=evidence,
        intervention=intervention,
        hist=hist,
    )
    quality = _gcc_detect_governance_decision_quality(
        coherence=coherence,
        tension=tension,
        integrity=integrity,
        decision=decision,
        stability=stability,
        resilience=resilience,
        evidence=evidence,
        alignment=alignment,
        intervention=intervention,
        improvement=improvement,
        hist=hist,
    )
    return {
        "hist": hist,
        "regime": regime,
        "forecast": forecast,
        "tension": tension,
        "consensus": consensus,
        "integrity": integrity,
        "decision": decision,
        "failure": failure,
        "audit": audit,
        "coherence": coherence,
        "stability": stability,
        "resilience": resilience,
        "improvement": improvement,
        "scenario": scenario,
        "intervention": intervention,
        "evidence": evidence,
        "alignment": alignment,
        "quality": quality,
    }


def _gcc_executive_stability_summary(quality: Dict[str, Any], stability: Dict[str, Any]) -> str:
    scrutiny = quality.get("stability_under_scrutiny", "CONDITIONAL")
    mapping = {
        "UNSTABLE": "BROKEN",
        "FRAGILE": "WEAK",
        "CONDITIONAL": "CONDITIONAL",
        "STABLE": "STABLE",
        "RESILIENT": "STRONG",
    }
    return mapping.get(scrutiny, "CONDITIONAL")


def _gcc_executive_trust_summary(evidence: Dict[str, Any]) -> str:
    trust = evidence.get("trustworthiness_state", "LIMITED")
    if trust == "BROKEN":
        return "BROKEN"
    if trust == "WEAK":
        return "WEAK"
    if trust == "LIMITED":
        return "LIMITED"
    if trust == "ACCEPTABLE":
        return "ACCEPTABLE"
    return "STRONG"


def _gcc_executive_alignment_summary(alignment: Dict[str, Any]) -> str:
    strength = alignment.get("alignment_strength", "LIMITED")
    state = alignment.get("alignment_state", "")
    if state == "INSTITUTIONAL_CONSENSUS_BROKEN" or strength == "NONE":
        return "FRACTURED"
    if strength == "WEAK":
        return "WEAK"
    if strength in ("LIMITED", "MODERATE"):
        return "PARTIAL"
    if strength == "STRONG":
        return "STRONG"
    return "PARTIAL"


def _gcc_constitutional_runtime_posture(
    dossier_summary: Dict[str, Any],
    intervention: Dict[str, Any],
    decision: Dict[str, Any],
    regime: Dict[str, Any],
) -> str:
    mutation_allowed = bool(_gcc_get(dossier_summary, "runtime_mutation_allowed", False))
    posture = intervention.get("intervention_posture", "")
    if not mutation_allowed or posture == "CONSTITUTIONAL_LOCK_REQUIRED":
        return "BLOCKED"
    if regime.get("regime") == "CONSTITUTIONAL_STRESS":
        return "CONSTRAINED"
    if posture in ("CONTAINMENT_RECOMMENDED", "INSTITUTIONAL_CAUTION_REQUIRED"):
        return "GUARDED"
    if posture in ("MONITORING_ELEVATED", "OBSERVATION_ONLY"):
        return "OBSERVED"
    if mutation_allowed and decision.get("readiness_state") in (
        "GOVERNANCE_REVIEW_ELIGIBLE",
        "OPERATOR_COMMITTEE_REVIEW_ELIGIBLE",
    ):
        return "READY"
    return "OBSERVED"


def _gcc_executive_top_risks(
    scenario: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    quality: Dict[str, Any],
    decision: Dict[str, Any],
    regime: Dict[str, Any],
    tension: Dict[str, Any],
) -> List[str]:
    risks: List[str] = []
    for warning in scenario.get("early_warnings") or []:
        if warning not in risks:
            risks.append(warning)
    if evidence.get("evidence_state") == "CONFIDENCE_OVEREXTENDED":
        risks.append("Confidence inflation risk")
    if alignment.get("alignment_state") in (
        "FRAGMENTATION_ELEVATED",
        "GOVERNANCE_INTERNAL_CONFLICT",
        "INSTITUTIONAL_CONSENSUS_BROKEN",
    ):
        risks.append("Governance fragmentation")
    if float(decision.get("governance_maturity_score", 0.0) or 0.0) < 0.20:
        risks.append("Institutional immaturity")
    if regime.get("regime") == "CONSTITUTIONAL_STRESS":
        risks.append("Constitutional pressure persistence")
    if quality.get("institutional_defensibility") in ("NONE", "WEAK"):
        risks.append("Weak decision defensibility")
    if (
        int(tension.get("contradiction_count", 0) or 0) >= 2
        and "Contradiction persistence" not in risks
    ):
        risks.append("Contradiction persistence")
    return risks[:5]


def _gcc_detect_executive_verdict(
    *,
    dossier_summary: Dict[str, Any],
    regime: Dict[str, Any],
    tension: Dict[str, Any],
    decision: Dict[str, Any],
    scenario: Dict[str, Any],
    intervention: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    quality: Dict[str, Any],
    improvement: Dict[str, Any],
    stability: Dict[str, Any],
) -> Dict[str, Any]:
    mutation_allowed = bool(_gcc_get(dossier_summary, "runtime_mutation_allowed", False))
    intervention_posture = intervention.get("intervention_posture", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)

    def _match_locked() -> bool:
        return intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED" or (
            not mutation_allowed and regime.get("regime") == "CONSTITUTIONAL_STRESS"
        )

    def _match_systemic() -> bool:
        return (
            scenario.get("scenario_state") == "SYSTEMIC_GOVERNANCE_FAILURE_RISK"
            or quality.get("quality_state") == "INSTITUTIONAL_REASONING_BROKEN"
            or alignment.get("alignment_state") == "INSTITUTIONAL_CONSENSUS_BROKEN"
        )

    def _match_stress() -> bool:
        return (
            intervention_posture
            in (
                "CONTAINMENT_RECOMMENDED",
                "INSTITUTIONAL_CAUTION_REQUIRED",
                "ESCALATION_REQUIRED",
            )
            or stability.get("stability_state")
            in (
                "GOVERNANCE_DETERIORATING",
                "REASONING_INSTABILITY_DETECTED",
            )
            or contradiction_count >= 2
        )

    def _match_stabilizing() -> bool:
        return (
            improvement.get("improvement_state")
            in (
                "GOVERNANCE_ADAPTATION_EMERGING",
                "GOVERNANCE_MATURITY_IMPROVING",
            )
            or stability.get("stability_state") == "GOVERNANCE_STABILITY_IMPROVING"
            or alignment.get("alignment_state") == "GOVERNANCE_CONSENSUS_FORMING"
        )

    def _match_stable() -> bool:
        return (
            intervention_posture == "OBSERVATION_ONLY"
            and quality.get("quality_state") == "HIGH_QUALITY_GOVERNANCE_REASONING"
            and alignment.get("alignment_state") == "INSTITUTIONAL_ALIGNMENT_STRONG"
            and evidence.get("evidence_state")
            in ("CONFIDENCE_SUPPORTED", "EVIDENCE_INTEGRITY_STRONG")
        )

    matchers = {
        "CONSTITUTIONALLY_LOCKED_GOVERNANCE": _match_locked,
        "SYSTEMIC_GOVERNANCE_INSTABILITY": _match_systemic,
        "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS": _match_stress,
        "GOVERNANCE_STABILIZING": _match_stabilizing,
        "GOVERNANCE_OPERATIONALLY_STABLE": _match_stable,
    }

    verdict_state = "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS"
    for candidate in _GCC_EXECUTIVE_VERDICT_PRIORITY:
        fn = matchers.get(candidate)
        if fn and fn():
            verdict_state = candidate
            break

    stability_summary = _gcc_executive_stability_summary(quality, stability)
    trust_summary = _gcc_executive_trust_summary(evidence)
    alignment_summary = _gcc_executive_alignment_summary(alignment)
    runtime_posture = _gcc_constitutional_runtime_posture(
        dossier_summary, intervention, decision, regime
    )
    top_risks = _gcc_executive_top_risks(
        scenario, evidence, alignment, quality, decision, regime, tension
    )
    why_drivers = [
        f"Intervention: {intervention.get('intervention_display', '—')}",
        f"Failure scenario: {scenario.get('scenario_display', '—')}",
        f"Decision quality: {quality.get('quality_display', '—')}",
        f"Evidence integrity: {evidence.get('evidence_display', '—')}",
        f"Alignment: {alignment.get('alignment_display', '—')}",
    ]

    return {
        "verdict_state": verdict_state,
        "verdict_display": _GCC_EXECUTIVE_VERDICT_DISPLAY.get(
            verdict_state, verdict_state.replace("_", " ").title()
        ),
        "stability_summary": stability_summary,
        "trust_summary": trust_summary,
        "alignment_summary": alignment_summary,
        "runtime_posture": runtime_posture,
        "top_risks": top_risks,
        "executive_memo": _GCC_EXECUTIVE_VERDICT_MEMO.get(verdict_state, ""),
        "operator_action": _GCC_EXECUTIVE_OPERATOR_ACTION.get(
            verdict_state, "CONTINUE_OBSERVATION"
        ),
        "why_drivers": why_drivers,
    }


def _gcc_detect_governance_attention(
    *,
    executive: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    regime: Dict[str, Any],
    tension: Dict[str, Any],
    decision: Dict[str, Any],
    scenario: Dict[str, Any],
    intervention: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    quality: Dict[str, Any],
    failure: Dict[str, Any],
    integrity: Dict[str, Any],
) -> Dict[str, Any]:
    verdict_state = executive["verdict_state"]
    runtime_posture = executive["runtime_posture"]
    trust_summary = executive["trust_summary"]
    stability_summary = executive["stability_summary"]
    intervention_posture = intervention.get("intervention_posture", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    mutation_allowed = bool(_gcc_get(dossier_summary, "runtime_mutation_allowed", False))

    def _match_critical_lock() -> bool:
        return (
            verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
            or (runtime_posture == "BLOCKED" and (maturity < 0.25 or not mutation_allowed))
        )

    def _match_high_risk() -> bool:
        return (
            verdict_state == "SYSTEMIC_GOVERNANCE_INSTABILITY"
            or scenario.get("scenario_state") == "SYSTEMIC_GOVERNANCE_FAILURE_RISK"
            or quality.get("quality_state") == "INSTITUTIONAL_REASONING_BROKEN"
            or (trust_summary in ("BROKEN", "WEAK") and contradiction_count >= 2)
            or (
                intervention_posture == "ESCALATION_REQUIRED"
                and failure.get("risk_severity") in ("SEVERE", "CRITICAL")
            )
        )

    def _match_elevated() -> bool:
        return (
            verdict_state == "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS"
            or intervention_posture
            in (
                "CONTAINMENT_RECOMMENDED",
                "INSTITUTIONAL_CAUTION_REQUIRED",
            )
            or trust_summary in ("BROKEN", "WEAK", "LIMITED")
            or stability_summary in ("BROKEN", "WEAK", "CONDITIONAL")
            or integrity.get("integrity_state")
            in (
                "GOVERNANCE_CONFIDENCE_WEAK",
                "GOVERNANCE_CONFIDENCE_UNDERCUT_BY_CONTRADICTIONS",
                "GOVERNANCE_OVERCONFIDENT",
            )
        )

    def _match_watch() -> bool:
        return (
            verdict_state == "GOVERNANCE_STABILIZING"
            or intervention_posture == "MONITORING_ELEVATED"
            or stability_summary == "CONDITIONAL"
            or contradiction_count == 1
        )

    def _match_normal() -> bool:
        return verdict_state == "GOVERNANCE_OPERATIONALLY_STABLE" or (
            intervention_posture == "OBSERVATION_ONLY"
            and trust_summary in ("ACCEPTABLE", "STRONG")
            and runtime_posture in ("OBSERVED", "READY")
        )

    severity_matchers = {
        "CRITICAL_LOCK": _match_critical_lock,
        "HIGH_RISK": _match_high_risk,
        "ELEVATED": _match_elevated,
        "WATCH": _match_watch,
        "NORMAL": _match_normal,
    }

    severity = "ELEVATED"
    for candidate in _GCC_SEVERITY_PRIORITY:
        fn = severity_matchers.get(candidate)
        if fn and fn():
            severity = candidate
            break

    attention_map = {
        "CRITICAL_LOCK": "HIGH",
        "HIGH_RISK": "IMMEDIATE",
        "ELEVATED": "MODERATE",
        "WATCH": "LOW",
        "NORMAL": "PASSIVE",
    }
    if intervention_posture == "ESCALATION_REQUIRED" and severity == "HIGH_RISK":
        attention = "IMMEDIATE"
    else:
        attention = attention_map.get(severity, "MODERATE")

    cadence_map = {
        "CRITICAL_LOCK": "NEXT_REFRESH",
        "HIGH_RISK": "INTRADAY",
        "ELEVATED": "DAILY",
        "WATCH": "DAILY",
        "NORMAL": "DAILY",
    }
    review_cadence = cadence_map.get(severity, "DAILY")
    if intervention_posture == "ESCALATION_REQUIRED":
        review_cadence = "IMMEDIATE_REVIEW"
    elif severity == "HIGH_RISK" and scenario.get("failure_probability") in ("HIGH", "VERY_HIGH"):
        review_cadence = "CONTINUOUS_MONITORING"

    escalation_threshold = intervention.get("escalation_threshold", "NONE")
    esc_cls = str(_gcc_get(dossier_summary, "human_escalation_classification") or "").upper()
    if intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED" or severity == "CRITICAL_LOCK":
        escalation_urgency = "LOW"
    elif intervention_posture == "ESCALATION_REQUIRED" or escalation_threshold == "IMMEDIATE":
        escalation_urgency = "IMMEDIATE"
    elif (
        escalation_threshold == "MATERIAL"
        or alignment.get("alignment_state") == "INSTITUTIONAL_CONSENSUS_BROKEN"
    ):
        escalation_urgency = "MATERIAL"
    elif escalation_threshold in ("HEIGHTENED", "WATCH") or trust_summary in ("BROKEN", "WEAK"):
        escalation_urgency = "GUARDED"
    elif esc_cls == "NO_ESCALATION" and severity in ("NORMAL", "WATCH"):
        escalation_urgency = "NONE"
    else:
        escalation_urgency = "LOW"

    return {
        "severity": severity,
        "severity_display": _GCC_SEVERITY_DISPLAY.get(severity, severity.replace("_", " ").title()),
        "attention_level": attention,
        "review_cadence": review_cadence,
        "escalation_urgency": escalation_urgency,
        "attention_memo": _GCC_ATTENTION_MEMO.get(severity, ""),
    }


def _gcc_detect_governance_trend(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    forecast: Dict[str, Any],
    tension: Dict[str, Any],
    stability: Dict[str, Any],
    improvement: Dict[str, Any],
    resilience: Dict[str, Any],
    scenario: Dict[str, Any],
    intervention: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    quality: Dict[str, Any],
    failure: Dict[str, Any],
) -> Dict[str, Any]:
    verdict_state = executive["verdict_state"]
    severity = attention["severity"]
    trust_summary = executive["trust_summary"]
    direction = hist.get("confidence_direction", "stable")
    momentum = hist.get("institutional_momentum", "NONE")
    stability_state = stability.get("stability_state", "")
    improvement_state = improvement.get("improvement_state", "")
    resilience_state = resilience.get("resilience_state", "")
    intervention_posture = intervention.get("intervention_posture", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    drift_severity = stability.get("drift_severity", "LOW")
    volatility = stability.get("volatility_level", "LOW")
    regression_risk = forecast.get("regression_risk", "NONE")
    trajectory = forecast.get("trajectory", "")

    constitutional_lock_stable = (
        (
            severity == "CRITICAL_LOCK"
            or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
        )
        and direction in ("stable", "dormant")
        and momentum != "HIGH"
    )

    def _match_rapidly_deteriorating() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            (direction == "deteriorating" and momentum == "HIGH")
            or (stability_state == "GOVERNANCE_DETERIORATING" and direction == "deteriorating")
            or drift_severity == "CRITICAL"
            or volatility == "EXTREME"
            or (
                regression_risk == "HIGH"
                and direction == "deteriorating"
                and contradiction_count >= 2
            )
        )

    def _match_deteriorating() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            direction == "deteriorating"
            or (
                stability_state == "GOVERNANCE_DETERIORATING"
                and direction not in ("stable", "dormant")
            )
            or (
                stability_state == "REASONING_INSTABILITY_DETECTED" and direction == "deteriorating"
            )
            or (
                trajectory == "GOVERNANCE_REGRESSION_RISK"
                and direction in ("deteriorating", "mixed")
            )
            or (
                improvement_state
                in (
                    "IMPROVEMENT_TRAJECTORY_BLOCKED",
                    "GOVERNANCE_LEARNING_STALLED",
                )
                and direction in ("deteriorating", "mixed")
            )
            or (
                stability_state == "GOVERNANCE_VOLATILITY_ELEVATED"
                and direction in ("deteriorating", "mixed")
            )
        )

    def _match_stressed_but_stable() -> bool:
        if constitutional_lock_stable:
            return True
        return (
            verdict_state == "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS"
            and direction in ("stable", "dormant")
            and stability_state
            not in (
                "GOVERNANCE_DETERIORATING",
                "REASONING_INSTABILITY_DETECTED",
            )
        )

    def _match_stabilizing() -> bool:
        return (
            verdict_state == "GOVERNANCE_STABILIZING"
            or improvement_state
            in (
                "GOVERNANCE_ADAPTATION_EMERGING",
                "CONTRADICTIONS_DECLINING",
            )
            or stability_state == "GOVERNANCE_STABILITY_IMPROVING"
            or (
                direction == "improving"
                and regime.get("regime") == "CONSTITUTIONAL_STRESS"
                and contradiction_count <= 1
            )
        )

    def _match_improving() -> bool:
        return (
            direction == "improving"
            and improvement_state
            in (
                "GOVERNANCE_MATURITY_IMPROVING",
                "GOVERNANCE_EVOLUTION_STRENGTHENING",
                "CONTRADICTIONS_DECLINING",
            )
        ) or trajectory in ("GOVERNANCE_IMPROVING", "GOVERNANCE_ACCELERATING")

    def _match_operationally_stable() -> bool:
        return verdict_state == "GOVERNANCE_OPERATIONALLY_STABLE" or (
            stability_state == "STABLE_GOVERNANCE_POSTURE"
            and direction in ("stable", "improving")
            and contradiction_count == 0
            and trust_summary in ("ACCEPTABLE", "STRONG")
        )

    direction_matchers = {
        "RAPIDLY_DETERIORATING": _match_rapidly_deteriorating,
        "DETERIORATING": _match_deteriorating,
        "STRESSED_BUT_STABLE": _match_stressed_but_stable,
        "STABILIZING": _match_stabilizing,
        "IMPROVING": _match_improving,
        "OPERATIONALLY_STABLE": _match_operationally_stable,
    }

    governance_direction = "STRESSED_BUT_STABLE"
    for candidate in _GCC_DIRECTION_PRIORITY:
        fn = direction_matchers.get(candidate)
        if fn and fn():
            governance_direction = candidate
            break

    if momentum == "HIGH" or drift_severity == "CRITICAL" or volatility == "EXTREME":
        trend_velocity = "RAPID"
    elif momentum == "MODERATE" or drift_severity == "HIGH" or volatility == "HIGH":
        trend_velocity = "FAST"
    elif momentum == "LOW" or drift_severity == "MODERATE" or volatility == "MODERATE":
        trend_velocity = "MODERATE"
    elif direction in ("stable", "dormant") and momentum == "NONE":
        trend_velocity = "MINIMAL"
    else:
        trend_velocity = "SLOW"

    drift_map = {
        "CRITICAL": "SEVERE",
        "HIGH": "MATERIAL",
        "MODERATE": "GUARDED",
        "LOW": "LOW",
        "NONE": "NONE",
    }
    drift_risk = drift_map.get(drift_severity, "LOW")
    if drift_risk == "LOW" and (
        contradiction_count >= 2
        or trust_summary in ("BROKEN", "WEAK")
        or alignment.get("alignment_state")
        in (
            "FRAGMENTATION_ELEVATED",
            "GOVERNANCE_INTERNAL_CONFLICT",
        )
    ):
        drift_risk = "GUARDED"
    if (
        quality.get("quality_state") == "INSTITUTIONAL_REASONING_BROKEN"
        and scenario.get("scenario_state") == "SYSTEMIC_GOVERNANCE_FAILURE_RISK"
    ):
        drift_risk = "SEVERE"

    if scenario.get("scenario_state") in (
        "SYSTEMIC_GOVERNANCE_FAILURE_RISK",
        "CONSTITUTIONAL_BREAKDOWN_RISK",
    ) and governance_direction in ("RAPIDLY_DETERIORATING", "DETERIORATING"):
        structural_outlook = "COLLAPSING"
    elif resilience_state in (
        "GOVERNANCE_RECOVERY_BLOCKED",
        "CONSTITUTIONAL_RECOVERY_UNLIKELY",
    ) or failure.get("risk_severity") in ("CRITICAL", "HIGH"):
        structural_outlook = "FRAGILE"
    elif (
        resilience_state
        in (
            "RECOVERABLE_FRAGMENTATION",
            "REVERSIBLE_GOVERNANCE_DETERIORATION",
            "MODERATE_GOVERNANCE_RESILIENCE",
        )
        or improvement_state == "INSTITUTIONAL_IMPROVEMENT_POSSIBLE"
    ):
        structural_outlook = "RECOVERABLE"
    elif (
        stability_state == "STABLE_GOVERNANCE_POSTURE"
        or resilience_state == "HIGH_GOVERNANCE_RESILIENCE"
    ):
        structural_outlook = "STABLE"
    elif governance_direction == "STRESSED_BUT_STABLE" or severity == "CRITICAL_LOCK":
        structural_outlook = "CONDITIONAL"
    else:
        structural_outlook = "CONDITIONAL"

    if governance_direction == "STRESSED_BUT_STABLE":
        if direction in ("stable", "dormant") and momentum == "NONE":
            trend_velocity = "MINIMAL"
        elif direction in ("stable", "dormant"):
            trend_velocity = "SLOW"
        if constitutional_lock_stable:
            drift_risk = "LOW"
        elif drift_risk in ("SEVERE", "MATERIAL"):
            drift_risk = "GUARDED" if contradiction_count >= 2 else "LOW"
        if structural_outlook in ("COLLAPSING", "FRAGILE"):
            structural_outlook = "RECOVERABLE"

    return {
        "governance_direction": governance_direction,
        "direction_display": _GCC_DIRECTION_DISPLAY.get(
            governance_direction, governance_direction.replace("_", " ").title()
        ),
        "trend_velocity": trend_velocity,
        "drift_risk": drift_risk,
        "structural_outlook": structural_outlook,
        "trend_memo": _GCC_TREND_MEMO.get(governance_direction, ""),
    }


def _gcc_cockpit_signal_conflicts(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    trend: Dict[str, Any],
    evidence: Dict[str, Any],
) -> int:
    conflicts = 0
    verdict = executive["verdict_state"]
    severity = attention["severity"]
    direction = trend["governance_direction"]
    velocity = trend["trend_velocity"]

    if verdict == "GOVERNANCE_OPERATIONALLY_STABLE" and direction in (
        "DETERIORATING",
        "RAPIDLY_DETERIORATING",
    ):
        conflicts += 1
    if verdict == "GOVERNANCE_STABILIZING" and direction == "RAPIDLY_DETERIORATING":
        conflicts += 1
    if severity == "NORMAL" and direction in ("DETERIORATING", "RAPIDLY_DETERIORATING"):
        conflicts += 1
    if (
        severity == "HIGH_RISK"
        and direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
    ):
        conflicts += 1
    if executive["runtime_posture"] == "READY" and direction not in (
        "IMPROVING",
        "OPERATIONALLY_STABLE",
    ):
        conflicts += 1
    if (
        executive["trust_summary"] in ("STRONG", "ACCEPTABLE")
        and evidence.get("evidence_state") == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED"
    ):
        conflicts += 1
    if executive["alignment_summary"] == "STRONG" and direction in (
        "DETERIORATING",
        "RAPIDLY_DETERIORATING",
    ):
        conflicts += 1
    return conflicts


def _gcc_detect_governance_signal_confidence(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    trend: Dict[str, Any],
    hist: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    quality: Dict[str, Any],
    decision: Dict[str, Any],
    tension: Dict[str, Any],
    stability: Dict[str, Any],
    integrity: Dict[str, Any],
    coherence: Dict[str, Any],
) -> Dict[str, Any]:
    trust_summary = executive["trust_summary"]
    stability_summary = executive["stability_summary"]
    alignment_summary = executive["alignment_summary"]
    evidence_state = evidence.get("evidence_state", "")
    quality_state = quality.get("quality_state", "")
    alignment_state = alignment.get("alignment_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    velocity = trend["trend_velocity"]
    drift_risk = trend["drift_risk"]
    gov_direction = trend["governance_direction"]
    volatility = stability.get("volatility_level", "LOW")
    conflicts = _gcc_cockpit_signal_conflicts(
        executive=executive, attention=attention, trend=trend, evidence=evidence
    )

    interpretation_coherent = (
        gov_direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
        and drift_risk in ("LOW", "NONE")
        and conflicts == 0
    )

    def _match_very_low() -> bool:
        if interpretation_coherent:
            return False
        return (
            conflicts >= 2
            or (
                quality_state == "INSTITUTIONAL_REASONING_BROKEN"
                and alignment_state
                in (
                    "INSTITUTIONAL_CONSENSUS_BROKEN",
                    "GOVERNANCE_INTERNAL_CONFLICT",
                )
            )
            or (
                evidence_state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED"
                and coherence.get("coherence_state")
                in (
                    "FRAGMENTED_REASONING_CHAIN",
                    "INTERNALLY_INCONSISTENT_GOVERNANCE",
                )
                and contradiction_count >= 2
            )
        )

    def _match_low() -> bool:
        if interpretation_coherent:
            return False
        return (
            trust_summary in ("BROKEN", "WEAK")
            or quality_state == "INSTITUTIONAL_REASONING_BROKEN"
            or evidence_state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED"
            or conflicts == 1
        )

    def _match_guarded() -> bool:
        return (
            interpretation_coherent
            or (
                gov_direction == "STRESSED_BUT_STABLE"
                and velocity in ("MINIMAL", "SLOW")
                and conflicts <= 1
            )
            or (
                trust_summary in ("LIMITED", "WEAK", "BROKEN")
                and drift_risk in ("LOW", "NONE", "GUARDED")
                and velocity in ("MINIMAL", "SLOW", "MODERATE")
            )
        )

    def _match_moderate() -> bool:
        return (
            trust_summary in ("ACCEPTABLE", "LIMITED")
            and quality_state
            not in (
                "INSTITUTIONAL_REASONING_BROKEN",
                "DECISION_QUALITY_MATERIALLY_IMPAIRED",
            )
            and gov_direction not in ("RAPIDLY_DETERIORATING",)
            and conflicts == 0
        )

    def _match_high() -> bool:
        return (
            trust_summary in ("ACCEPTABLE", "STRONG")
            and alignment_summary in ("ALIGNED", "STRONG", "PARTIAL")
            and evidence_state in ("CONFIDENCE_SUPPORTED", "EVIDENCE_INTEGRITY_STRONG")
            and quality_state == "HIGH_QUALITY_GOVERNANCE_REASONING"
        )

    def _match_very_high() -> bool:
        return (
            executive["verdict_state"] == "GOVERNANCE_OPERATIONALLY_STABLE"
            and trust_summary == "STRONG"
            and gov_direction == "OPERATIONALLY_STABLE"
            and drift_risk in ("LOW", "NONE")
            and conflicts == 0
        )

    confidence_matchers = {
        "VERY_LOW": _match_very_low,
        "LOW": _match_low,
        "GUARDED": _match_guarded,
        "MODERATE": _match_moderate,
        "HIGH": _match_high,
        "VERY_HIGH": _match_very_high,
    }

    cockpit_confidence = "GUARDED"
    for candidate in _GCC_COCKPIT_CONFIDENCE_PRIORITY:
        fn = confidence_matchers.get(candidate)
        if fn and fn():
            cockpit_confidence = candidate
            break

    reliability_map = {
        "BROKEN": "BROKEN",
        "WEAK": "WEAK",
        "LIMITED": "GUARDED",
        "ACCEPTABLE": "ACCEPTABLE",
        "STRONG": "STRONG",
    }
    reliability = reliability_map.get(trust_summary, "GUARDED")
    if quality_state == "INSTITUTIONAL_REASONING_BROKEN":
        reliability = "BROKEN"
    elif quality_state == "DECISION_QUALITY_MATERIALLY_IMPAIRED" and reliability not in ("BROKEN",):
        reliability = "WEAK"
    elif evidence_state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED" and reliability == "STRONG":
        reliability = "GUARDED"
    if interpretation_coherent and reliability in ("BROKEN", "WEAK"):
        reliability = "GUARDED"

    if (
        gov_direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
        and attention["severity"] == "CRITICAL_LOCK"
    ):
        false_alarm_risk = "LOW"
    elif conflicts >= 2:
        false_alarm_risk = "HIGH"
    elif conflicts == 1:
        false_alarm_risk = "MATERIAL"
    elif attention["severity"] == "HIGH_RISK" and gov_direction == "STRESSED_BUT_STABLE":
        false_alarm_risk = "GUARDED"
    elif velocity == "RAPID" and gov_direction == "RAPIDLY_DETERIORATING":
        false_alarm_risk = "GUARDED"
    else:
        false_alarm_risk = "LOW"

    if volatility == "EXTREME" or velocity == "RAPID":
        interpretation_stability = "VOLATILE"
    elif velocity == "FAST" or drift_risk in ("SEVERE", "MATERIAL"):
        interpretation_stability = "UNSTABLE"
    elif gov_direction == "OPERATIONALLY_STABLE" and drift_risk in ("LOW", "NONE"):
        interpretation_stability = "DURABLE"
    elif interpretation_coherent or gov_direction == "STRESSED_BUT_STABLE":
        interpretation_stability = "CONDITIONAL"
    elif velocity in ("MINIMAL", "SLOW") and conflicts == 0:
        interpretation_stability = "STABLE"
    else:
        interpretation_stability = "CONDITIONAL"

    return {
        "cockpit_confidence": cockpit_confidence,
        "confidence_display": _GCC_COCKPIT_CONFIDENCE_DISPLAY.get(
            cockpit_confidence, cockpit_confidence.replace("_", " ").title()
        ),
        "reliability_level": reliability,
        "false_alarm_risk": false_alarm_risk,
        "interpretation_stability": interpretation_stability,
        "confidence_memo": _GCC_COCKPIT_CONFIDENCE_MEMO.get(cockpit_confidence, ""),
    }


def _gcc_detect_governance_operator_playbook(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    trend: Dict[str, Any],
    signal_conf: Dict[str, Any],
    dossier_summary: Dict[str, Any],
    intervention: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    quality: Dict[str, Any],
    tension: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    verdict_state = executive["verdict_state"]
    severity = attention["severity"]
    review_cadence = attention["review_cadence"]
    escalation_urgency = attention["escalation_urgency"]
    runtime_posture = executive["runtime_posture"]
    trust_summary = executive["trust_summary"]
    gov_direction = trend["governance_direction"]
    velocity = trend["trend_velocity"]
    drift_risk = trend["drift_risk"]
    intervention_posture = intervention.get("intervention_posture", "")
    evidence_state = evidence.get("evidence_state", "")
    alignment_state = alignment.get("alignment_state", "")
    quality_state = quality.get("quality_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    mutation_allowed = bool(_gcc_get(dossier_summary, "runtime_mutation_allowed", False))

    constitutional_lock_stable = (
        (
            severity == "CRITICAL_LOCK"
            or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
            or runtime_posture == "BLOCKED"
        )
        and gov_direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
    )

    if constitutional_lock_stable:
        priority_action = "MAINTAIN_CONSTITUTIONAL_LOCK"
    elif (
        intervention_posture == "ESCALATION_REQUIRED"
        or escalation_urgency == "IMMEDIATE"
        or (severity == "HIGH_RISK" and gov_direction in ("DETERIORATING", "RAPIDLY_DETERIORATING"))
    ):
        priority_action = "PREPARE_ESCALATION_REVIEW"
    elif (
        verdict_state == "SYSTEMIC_GOVERNANCE_INSTABILITY"
        or gov_direction in ("DETERIORATING", "RAPIDLY_DETERIORATING")
        or (
            alignment_state
            in (
                "FRAGMENTATION_ELEVATED",
                "GOVERNANCE_INTERNAL_CONFLICT",
                "INSTITUTIONAL_CONSENSUS_BROKEN",
            )
            and gov_direction not in ("STRESSED_BUT_STABLE",)
        )
    ):
        priority_action = "REDUCE_FRAGMENTATION"
    elif evidence_state in (
        "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED",
        "CONFIDENCE_OVEREXTENDED",
        "EVIDENCE_QUALITY_DEGRADING",
    ) or trust_summary in ("BROKEN", "WEAK"):
        priority_action = "IMPROVE_CONFIDENCE_INTEGRITY"
    elif (
        severity == "CRITICAL_LOCK"
        or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
        or runtime_posture == "BLOCKED"
    ):
        priority_action = "MAINTAIN_CONSTITUTIONAL_LOCK"
    elif (
        severity in ("ELEVATED", "WATCH")
        or verdict_state == "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS"
    ):
        priority_action = "MONITOR_GOVERNANCE_STRESS"
    elif verdict_state == "GOVERNANCE_OPERATIONALLY_STABLE":
        priority_action = "CONTINUE_DISCIPLINED_OBSERVATION"
    else:
        priority_action = executive.get("operator_action", "CONTINUE_DISCIPLINED_OBSERVATION")
        if priority_action == "CONTINUE_OBSERVATION":
            priority_action = "CONTINUE_DISCIPLINED_OBSERVATION"

    immediate: List[str] = []
    if priority_action == "MAINTAIN_CONSTITUTIONAL_LOCK":
        immediate.extend(
            [
                "Maintain constitutional safeguards",
                f"Continue {review_cadence.replace('_', ' ').lower()} review cadence",
                "Monitor contradiction persistence",
                "Observe trustworthiness trend",
            ]
        )
    elif priority_action == "REDUCE_FRAGMENTATION":
        immediate.extend(
            [
                "Increase governance observation",
                "Review fragmentation signals",
                "Reassess alignment and consensus posture",
                "Review escalation threshold",
            ]
        )
    elif priority_action == "IMPROVE_CONFIDENCE_INTEGRITY":
        immediate.extend(
            [
                "Monitor evidence integrity signals",
                "Review confidence reliability posture",
                "Observe trustworthiness deterioration",
                "Maintain constitutional safeguards",
            ]
        )
    elif priority_action == "PREPARE_ESCALATION_REVIEW":
        immediate.extend(
            [
                "Increase governance observation",
                "Review escalation threshold and dossier posture",
                "Reassess evidence deterioration",
                "Monitor fragmentation and contradiction signals",
            ]
        )
    elif priority_action == "MONITOR_GOVERNANCE_STRESS":
        immediate.extend(
            [
                "Maintain heightened governance monitoring",
                "Track contradiction and coherence signals",
                "Observe intervention posture changes",
            ]
        )
    else:
        immediate.extend(
            [
                "Continue routine governance observation",
                "Monitor directional and drift signals",
                "Review governance refresh outputs",
            ]
        )

    if contradiction_count >= 2 and "Monitor contradiction persistence" not in immediate:
        immediate.append("Monitor contradiction persistence")
    if velocity in ("FAST", "RAPID") and "Track drift acceleration" not in immediate:
        immediate.append("Track drift acceleration")
    immediate = immediate[:6]

    deferred: List[str] = []
    if alignment_state in (
        "FRAGMENTATION_ELEVATED",
        "GOVERNANCE_INTERNAL_CONFLICT",
        "GOVERNANCE_CONSENSUS_FORMING",
    ):
        deferred.append("Reassess governance coherence")
    if trust_summary in ("BROKEN", "WEAK", "LIMITED"):
        deferred.append("Monitor confidence recovery")
    if escalation_urgency not in ("NONE", "LOW"):
        deferred.append("Revisit escalation posture")
    if alignment_summary := executive.get("alignment_summary"):
        if alignment_summary in ("WEAK", "PARTIAL", "FRACTURED"):
            deferred.append("Review alignment stabilization")
    if gov_direction in ("STABILIZING", "STRESSED_BUT_STABLE"):
        deferred.append("Reassess institutional maturity trajectory")
    if not deferred:
        deferred.append("Review governance ladder progression")
    deferred = deferred[:5]

    blocked: List[str] = []
    if not mutation_allowed or runtime_posture == "BLOCKED":
        blocked.extend(
            [
                "Runtime enablement",
                "Governance relaxation",
                "Policy override",
                "Autonomy escalation",
                "Runtime mutation",
            ]
        )
    elif runtime_posture in ("CONSTRAINED", "GUARDED"):
        blocked.extend(
            [
                "Runtime enablement",
                "Policy override",
                "Autonomy escalation",
            ]
        )
    elif intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED":
        blocked.extend(
            [
                "Runtime mutation",
                "Governance relaxation",
            ]
        )
    if not blocked:
        blocked.append("Premature runtime enablement without review")
    blocked = blocked[:6]

    monitoring: List[str] = []
    if contradiction_count >= 1:
        monitoring.append("Contradiction severity")
    if executive.get("stability_summary") in ("BROKEN", "WEAK", "CONDITIONAL"):
        monitoring.append("Governance coherence")
    if trust_summary in ("BROKEN", "WEAK", "LIMITED"):
        monitoring.append("Trustworthiness trend")
    if evidence_state in (
        "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED",
        "EVIDENCE_QUALITY_DEGRADING",
        "CONFIDENCE_OVEREXTENDED",
    ):
        monitoring.append("Evidence integrity deterioration")
    if drift_risk not in ("LOW", "NONE"):
        monitoring.append("Drift acceleration")
    if alignment_state in (
        "FRAGMENTATION_ELEVATED",
        "GOVERNANCE_INTERNAL_CONFLICT",
        "INSTITUTIONAL_CONSENSUS_BROKEN",
    ):
        monitoring.append("Alignment fragmentation")
    for risk in executive.get("top_risks") or []:
        label = risk.split(" risk")[0].strip()
        if label and label not in monitoring and len(monitoring) < 6:
            monitoring.append(label)
    if not monitoring:
        monitoring.extend(
            [
                "Governance coherence",
                "Trustworthiness trend",
                "Evidence integrity",
            ]
        )
    monitoring = monitoring[:6]

    discipline_memo = _GCC_PLAYBOOK_DISCIPLINE_MEMO.get(
        priority_action,
        _GCC_PLAYBOOK_DISCIPLINE_MEMO["CONTINUE_DISCIPLINED_OBSERVATION"],
    )

    return {
        "priority_action": priority_action,
        "immediate_actions": immediate,
        "deferred_actions": deferred,
        "blocked_actions": blocked,
        "monitoring_priorities": monitoring,
        "discipline_memo": discipline_memo,
    }


def _gcc_detect_governance_temporal(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    trend: Dict[str, Any],
    playbook: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    stability: Dict[str, Any],
    improvement: Dict[str, Any],
    resilience: Dict[str, Any],
    intervention: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    verdict_state = executive["verdict_state"]
    severity = attention["severity"]
    review_cadence = attention["review_cadence"]
    attention_level = attention["attention_level"]
    gov_direction = trend["governance_direction"]
    velocity = trend["trend_velocity"]
    drift_risk = trend["drift_risk"]
    structural_outlook = trend["structural_outlook"]
    runtime_posture = executive["runtime_posture"]
    intervention_posture = intervention.get("intervention_posture", "")
    resilience_state = resilience.get("resilience_state", "")
    recovery_prob = resilience.get("recovery_probability", "LOW")
    improvement_state = improvement.get("improvement_state", "")
    stability_state = stability.get("stability_state", "")
    hist_direction = hist.get("confidence_direction", "stable")
    hist_momentum = hist.get("institutional_momentum", "NONE")
    transition_n = len(hist.get("transitions") or [])
    has_transitions = bool(hist.get("has_transitions"))
    maturity = float(decision.get("governance_maturity_score", 0.0) or 0.0)
    regime_key = regime.get("regime", "")

    constitutional_lock_stable = (
        (
            severity == "CRITICAL_LOCK"
            or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
        )
        and gov_direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
    )

    def _match_entrenched() -> bool:
        return (
            gov_direction in ("DETERIORATING", "RAPIDLY_DETERIORATING")
            and velocity in ("FAST", "RAPID")
            and resilience_state
            in (
                "GOVERNANCE_RECOVERY_BLOCKED",
                "CONSTITUTIONAL_RECOVERY_UNLIKELY",
            )
            and improvement_state
            in (
                "IMPROVEMENT_TRAJECTORY_BLOCKED",
                "GOVERNANCE_LEARNING_STALLED",
            )
        )

    def _match_persistent_lock() -> bool:
        return constitutional_lock_stable or (
            playbook.get("priority_action") == "MAINTAIN_CONSTITUTIONAL_LOCK"
            and gov_direction == "STRESSED_BUT_STABLE"
            and velocity in ("MINIMAL", "SLOW")
        )

    def _match_transitioning() -> bool:
        return (
            hist_direction in ("mixed", "deteriorating", "improving")
            and hist_momentum in ("MODERATE", "HIGH")
        ) or (
            gov_direction == "TRANSITIONING"
            or (transition_n >= 2 and velocity in ("MODERATE", "FAST"))
        )

    def _match_stabilizing() -> bool:
        return (
            gov_direction == "STABILIZING"
            or verdict_state == "GOVERNANCE_STABILIZING"
            or improvement_state
            in (
                "GOVERNANCE_ADAPTATION_EMERGING",
                "CONTRADICTIONS_DECLINING",
                "GOVERNANCE_MATURITY_IMPROVING",
            )
        )

    def _match_short_term() -> bool:
        return (
            severity in ("WATCH", "NORMAL")
            and velocity in ("MINIMAL", "SLOW")
            and transition_n <= 1
            and gov_direction not in ("DETERIORATING", "RAPIDLY_DETERIORATING")
        )

    def _match_durable() -> bool:
        return (
            verdict_state == "GOVERNANCE_OPERATIONALLY_STABLE"
            or gov_direction == "OPERATIONALLY_STABLE"
        )

    persistence_matchers = {
        "ENTRENCHED_INSTABILITY": _match_entrenched,
        "PERSISTENT_LOCK": _match_persistent_lock,
        "TRANSITIONING": _match_transitioning,
        "STABILIZING": _match_stabilizing,
        "SHORT_TERM_VARIATION": _match_short_term,
        "OPERATIONALLY_DURABLE": _match_durable,
    }

    governance_persistence = "PERSISTENT_LOCK"
    for candidate in _GCC_PERSISTENCE_PRIORITY:
        fn = persistence_matchers.get(candidate)
        if fn and fn():
            governance_persistence = candidate
            break

    if (
        hist_direction == "dormant"
        and hist_momentum == "NONE"
        and not has_transitions
        and regime_key == "CONSTITUTIONAL_STRESS"
    ):
        regime_duration = "PERSISTENT"
    elif hist_direction in ("stable", "dormant") and hist_momentum == "NONE" and transition_n == 0:
        regime_duration = "ESTABLISHED"
    elif transition_n >= 4 or hist_momentum == "HIGH":
        regime_duration = "DEVELOPING"
    elif transition_n >= 2:
        regime_duration = "SHORT"
    elif transition_n == 1:
        regime_duration = "SHORT"
    elif not hist.get("has_history"):
        regime_duration = "NEW"
    elif maturity >= 0.40 and stability_state == "STABLE_GOVERNANCE_POSTURE":
        regime_duration = "STRUCTURAL"
    elif hist_direction in ("stable", "dormant"):
        regime_duration = "ESTABLISHED"
    else:
        regime_duration = "DEVELOPING"

    if velocity == "RAPID" or (gov_direction == "RAPIDLY_DETERIORATING" and velocity == "FAST"):
        momentum_strength = "ACCELERATING"
    elif hist_momentum == "HIGH" or velocity == "FAST":
        momentum_strength = "STRONG"
    elif hist_momentum == "MODERATE" or velocity == "MODERATE":
        momentum_strength = "MODERATE"
    elif hist_momentum == "LOW" or velocity == "SLOW":
        momentum_strength = "WEAK"
    else:
        momentum_strength = "NONE"

    if constitutional_lock_stable or (
        structural_outlook == "RECOVERABLE" and gov_direction == "STRESSED_BUT_STABLE"
    ):
        recovery_horizon = "LONG"
    elif structural_outlook == "COLLAPSING" or recovery_prob == "VERY_LOW":
        recovery_horizon = "EXTENDED"
    elif recovery_prob in ("MODERATE", "HIGH") or gov_direction == "STABILIZING":
        recovery_horizon = "MEDIUM"
    elif gov_direction == "OPERATIONALLY_STABLE" or recovery_prob == "HIGH":
        recovery_horizon = "SHORT"
    elif recovery_prob == "LOW" and gov_direction == "STRESSED_BUT_STABLE":
        recovery_horizon = "LONG"
    else:
        recovery_horizon = "UNKNOWN"

    if constitutional_lock_stable and review_cadence == "NEXT_REFRESH":
        fatigue_risk = "LOW"
    elif attention_level == "IMMEDIATE" and review_cadence in (
        "IMMEDIATE_REVIEW",
        "CONTINUOUS_MONITORING",
    ):
        fatigue_risk = "HIGH"
    elif attention_level in ("HIGH", "IMMEDIATE") and velocity in ("FAST", "RAPID"):
        fatigue_risk = "MATERIAL"
    elif attention_level == "MODERATE" or review_cadence == "INTRADAY":
        fatigue_risk = "GUARDED"
    elif severity in ("WATCH", "NORMAL") and velocity in ("MINIMAL", "SLOW"):
        fatigue_risk = "NONE"
    else:
        fatigue_risk = "LOW"

    return {
        "governance_persistence": governance_persistence,
        "persistence_display": _GCC_PERSISTENCE_DISPLAY.get(
            governance_persistence, governance_persistence.replace("_", " ").title()
        ),
        "regime_duration": regime_duration,
        "momentum_strength": momentum_strength,
        "recovery_horizon": recovery_horizon,
        "fatigue_risk": fatigue_risk,
        "temporal_memo": _GCC_TEMPORAL_MEMO.get(governance_persistence, ""),
    }


def _gcc_detect_governance_delta(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    trend: Dict[str, Any],
    temporal: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
    tension: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    quality: Dict[str, Any],
    coherence: Dict[str, Any],
    improvement: Dict[str, Any],
    stability: Dict[str, Any],
    intervention: Dict[str, Any],
) -> Dict[str, Any]:
    gov_direction = trend["governance_direction"]
    velocity = trend["trend_velocity"]
    drift_risk = trend["drift_risk"]
    trust_summary = executive["trust_summary"]
    hist_direction = hist.get("confidence_direction", "stable")
    hist_momentum = hist.get("institutional_momentum", "NONE")
    transition_n = len(hist.get("transitions") or [])
    persistence = temporal.get("governance_persistence", "")
    momentum_strength = temporal.get("momentum_strength", "NONE")
    intervention_posture = intervention.get("intervention_posture", "")
    severity = attention["severity"]
    verdict_state = executive["verdict_state"]
    improvement_state = improvement.get("improvement_state", "")
    stability_state = stability.get("stability_state", "")
    evidence_state = evidence.get("evidence_state", "")
    alignment_state = alignment.get("alignment_state", "")
    coherence_state = coherence.get("coherence_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)

    constitutional_lock_stable = (
        (
            severity == "CRITICAL_LOCK"
            or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
        )
        and gov_direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
    )

    def _match_regime_shift() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            persistence == "TRANSITIONING"
            or (
                velocity in ("FAST", "RAPID")
                and gov_direction in ("DETERIORATING", "RAPIDLY_DETERIORATING", "STABILIZING")
            )
            or (hist_direction == "mixed" and hist_momentum in ("MODERATE", "HIGH"))
            or momentum_strength in ("STRONG", "ACCELERATING")
        )

    def _match_material_deterioration() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            gov_direction in ("DETERIORATING", "RAPIDLY_DETERIORATING")
            or hist_direction == "deteriorating"
            or stability_state == "GOVERNANCE_DETERIORATING"
        )

    def _match_material_improvement() -> bool:
        return (
            gov_direction in ("IMPROVING", "STABILIZING")
            or hist_direction == "improving"
            or improvement_state
            in (
                "CONTRADICTIONS_DECLINING",
                "GOVERNANCE_MATURITY_IMPROVING",
                "GOVERNANCE_EVOLUTION_STRENGTHENING",
            )
        )

    def _match_mixed() -> bool:
        if constitutional_lock_stable:
            return False
        return hist_direction == "mixed" or (
            improvement_state
            in ("GOVERNANCE_ADAPTATION_EMERGING", "INSTITUTIONAL_IMPROVEMENT_POSSIBLE")
            and trust_summary in ("BROKEN", "WEAK")
        )

    def _match_stable() -> bool:
        return (
            constitutional_lock_stable
            or (
                gov_direction == "STRESSED_BUT_STABLE"
                and velocity in ("MINIMAL", "SLOW")
                and drift_risk in ("LOW", "NONE")
                and hist_direction in ("stable", "dormant")
                and momentum_strength == "NONE"
            )
            or (
                gov_direction == "OPERATIONALLY_STABLE"
                and velocity in ("MINIMAL", "SLOW")
                and transition_n == 0
            )
        )

    delta_matchers = {
        "REGIME_SHIFT_EMERGING": _match_regime_shift,
        "MATERIAL_DETERIORATION": _match_material_deterioration,
        "MATERIAL_IMPROVEMENT": _match_material_improvement,
        "MIXED_TRANSITION": _match_mixed,
        "STABLE_NO_MATERIAL_CHANGE": _match_stable,
    }

    delta_state = "STABLE_NO_MATERIAL_CHANGE"
    for candidate in _GCC_DELTA_PRIORITY:
        fn = delta_matchers.get(candidate)
        if fn and fn():
            delta_state = candidate
            break

    positive_scores: Dict[str, int] = {
        "NONE": 0,
        "TRUST_IMPROVEMENT": 0,
        "ALIGNMENT_IMPROVEMENT": 0,
        "COHERENCE_STABILIZATION": 0,
        "CONTRADICTION_REDUCTION": 0,
        "EVIDENCE_STRENGTHENING": 0,
        "GOVERNANCE_STABILIZATION": 0,
    }
    negative_scores: Dict[str, int] = {
        "NONE": 0,
        "TRUST_DETERIORATION": 0,
        "ALIGNMENT_FRAGMENTATION": 0,
        "COHERENCE_BREAKDOWN": 0,
        "CONTRADICTION_ACCELERATION": 0,
        "EVIDENCE_DEGRADATION": 0,
        "GOVERNANCE_INSTABILITY": 0,
    }

    if improvement_state == "CONTRADICTIONS_DECLINING":
        positive_scores["CONTRADICTION_REDUCTION"] += 3
    if gov_direction in ("STABILIZING", "STRESSED_BUT_STABLE") and velocity in ("MINIMAL", "SLOW"):
        positive_scores["GOVERNANCE_STABILIZATION"] += 2
    if hist_direction == "improving":
        positive_scores["TRUST_IMPROVEMENT"] += 2
    if alignment_state == "GOVERNANCE_CONSENSUS_FORMING":
        positive_scores["ALIGNMENT_IMPROVEMENT"] += 2
    if coherence_state in ("HIGHLY_COHERENT_GOVERNANCE", "MODERATELY_COHERENT_GOVERNANCE"):
        positive_scores["COHERENCE_STABILIZATION"] += 2
    if evidence_state in ("CONFIDENCE_SUPPORTED", "EVIDENCE_INTEGRITY_STRONG"):
        positive_scores["EVIDENCE_STRENGTHENING"] += 2

    if trust_summary in ("BROKEN", "WEAK"):
        negative_scores["TRUST_DETERIORATION"] += 3
    if alignment_state in (
        "FRAGMENTATION_ELEVATED",
        "GOVERNANCE_INTERNAL_CONFLICT",
        "INSTITUTIONAL_CONSENSUS_BROKEN",
    ):
        negative_scores["ALIGNMENT_FRAGMENTATION"] += 3
    if coherence_state in (
        "FRAGMENTED_REASONING_CHAIN",
        "INTERNALLY_INCONSISTENT_GOVERNANCE",
        "LOW_COHERENCE_GOVERNANCE",
    ):
        negative_scores["COHERENCE_BREAKDOWN"] += 3
    if (
        contradiction_count >= 2
        and hist_direction == "deteriorating"
        and velocity in ("FAST", "RAPID", "MODERATE")
    ):
        negative_scores["CONTRADICTION_ACCELERATION"] += 3
    elif contradiction_count >= 2 and delta_state != "STABLE_NO_MATERIAL_CHANGE":
        negative_scores["CONTRADICTION_ACCELERATION"] += 1
    if evidence_state in (
        "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED",
        "EVIDENCE_QUALITY_DEGRADING",
    ):
        negative_scores["EVIDENCE_DEGRADATION"] += 2
    if gov_direction in ("DETERIORATING", "RAPIDLY_DETERIORATING"):
        negative_scores["GOVERNANCE_INSTABILITY"] += 3

    if delta_state == "STABLE_NO_MATERIAL_CHANGE":
        largest_positive = "NONE"
        largest_negative = "NONE"
    else:
        pos_key = max(positive_scores, key=positive_scores.get)
        neg_key = max(negative_scores, key=negative_scores.get)
        largest_positive = pos_key if positive_scores[pos_key] > 0 else "NONE"
        largest_negative = neg_key if negative_scores[neg_key] > 0 else "NONE"

    if constitutional_lock_stable or delta_state == "STABLE_NO_MATERIAL_CHANGE":
        regime_shift_probability = "LOW"
    elif delta_state == "REGIME_SHIFT_EMERGING":
        regime_shift_probability = "HIGH"
    elif delta_state == "MIXED_TRANSITION":
        regime_shift_probability = "GUARDED"
    elif delta_state == "MATERIAL_DETERIORATION" and velocity in ("FAST", "RAPID"):
        regime_shift_probability = "MATERIAL"
    elif persistence == "TRANSITIONING" or transition_n >= 2:
        regime_shift_probability = "GUARDED"
    elif momentum_strength in ("STRONG", "ACCELERATING"):
        regime_shift_probability = "MATERIAL"
    else:
        regime_shift_probability = "LOW"

    drivers: List[str] = []
    if delta_state == "STABLE_NO_MATERIAL_CHANGE":
        drivers.extend(
            [
                "Governance direction unchanged",
                "Drift minimal",
                "Constitutional posture stable",
                "No material escalation signals",
            ]
        )
    elif delta_state == "REGIME_SHIFT_EMERGING":
        if coherence_state in ("FRAGMENTED_REASONING_CHAIN", "LOW_COHERENCE_GOVERNANCE"):
            drivers.append("Governance coherence deteriorating")
        if trust_summary in ("BROKEN", "WEAK"):
            drivers.append("Trust weakening")
        if contradiction_count >= 2 and velocity not in ("MINIMAL", "SLOW"):
            drivers.append("Contradictions accelerating")
        if alignment_state in ("FRAGMENTATION_ELEVATED", "GOVERNANCE_INTERNAL_CONFLICT"):
            drivers.append("Alignment fragmenting")
    elif delta_state == "MATERIAL_DETERIORATION":
        if trust_summary in ("BROKEN", "WEAK"):
            drivers.append("Trustworthiness deteriorating")
        if coherence_state in ("FRAGMENTED_REASONING_CHAIN", "INTERNALLY_INCONSISTENT_GOVERNANCE"):
            drivers.append("Governance coherence weakening")
        if evidence_state == "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED":
            drivers.append("Evidence integrity constrained")
        if contradiction_count >= 1:
            drivers.append("Contradiction persistence elevated")
    elif delta_state == "MATERIAL_IMPROVEMENT":
        if improvement_state == "CONTRADICTIONS_DECLINING":
            drivers.append("Contradictions declining")
        if hist_direction == "improving":
            drivers.append("Trust stabilizing")
        if coherence_state not in ("FRAGMENTED_REASONING_CHAIN", "LOW_COHERENCE_GOVERNANCE"):
            drivers.append("Governance coherence improving")
        if evidence_state in ("CONFIDENCE_SUPPORTED", "PARTIALLY_SUPPORTED_CONFIDENCE"):
            drivers.append("Confidence integrity strengthening")
    elif delta_state == "MIXED_TRANSITION":
        drivers.append("Mixed historical confidence direction")
        if trust_summary in ("BROKEN", "WEAK", "LIMITED"):
            drivers.append("Trust signals remain constrained")
        if improvement_state in (
            "GOVERNANCE_ADAPTATION_EMERGING",
            "INSTITUTIONAL_IMPROVEMENT_POSSIBLE",
        ):
            drivers.append("Partial improvement signals emerging")
        drivers.append("Institutional ambiguity persists")

    for change in (hist.get("changes") or [])[:2]:
        if change not in drivers and len(drivers) < 6:
            drivers.append(change.rstrip("."))
    drivers = drivers[:6]

    return {
        "delta_state": delta_state,
        "delta_display": _GCC_DELTA_DISPLAY.get(delta_state, delta_state.replace("_", " ").title()),
        "largest_positive_change": largest_positive,
        "positive_display": _GCC_POSITIVE_CHANGE_DISPLAY.get(
            largest_positive, largest_positive.replace("_", " ").title()
        ),
        "largest_negative_change": largest_negative,
        "negative_display": _GCC_NEGATIVE_CHANGE_DISPLAY.get(
            largest_negative, largest_negative.replace("_", " ").title()
        ),
        "regime_shift_probability": regime_shift_probability,
        "change_drivers": drivers,
        "change_memo": _GCC_DELTA_CHANGE_MEMO.get(delta_state, ""),
    }


def _gcc_detect_governance_forward_forecast(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    trend: Dict[str, Any],
    temporal: Dict[str, Any],
    delta: Dict[str, Any],
    forecast: Dict[str, Any],
    scenario: Dict[str, Any],
    resilience: Dict[str, Any],
    intervention: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    tension: Dict[str, Any],
    hist: Dict[str, Any],
    regime: Dict[str, Any],
) -> Dict[str, Any]:
    gov_direction = trend["governance_direction"]
    velocity = trend["trend_velocity"]
    drift_risk = trend["drift_risk"]
    structural_outlook = trend["structural_outlook"]
    trust_summary = executive["trust_summary"]
    runtime_posture = executive["runtime_posture"]
    intervention_posture = intervention.get("intervention_posture", "")
    severity = attention["severity"]
    verdict_state = executive["verdict_state"]
    delta_state = delta.get("delta_state", "")
    regime_shift = delta.get("regime_shift_probability", "LOW")
    persistence = temporal.get("governance_persistence", "")
    momentum_strength = temporal.get("momentum_strength", "NONE")
    regression_risk = forecast.get("regression_risk", "NONE")
    trajectory = forecast.get("trajectory", "")
    resilience_state = resilience.get("resilience_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    hist_direction = hist.get("confidence_direction", "stable")

    constitutional_lock_stable = (
        (
            severity == "CRITICAL_LOCK"
            or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
        )
        and gov_direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
        and delta_state == "STABLE_NO_MATERIAL_CHANGE"
    )

    def _match_structural_deterioration() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            gov_direction in ("DETERIORATING", "RAPIDLY_DETERIORATING")
            or persistence == "ENTRENCHED_INSTABILITY"
            or structural_outlook == "COLLAPSING"
            or (regression_risk == "HIGH" and velocity in ("FAST", "RAPID"))
            or scenario.get("scenario_state") == "SYSTEMIC_GOVERNANCE_FAILURE_RISK"
        )

    def _match_persistent_constraint() -> bool:
        return (
            constitutional_lock_stable
            or (
                persistence == "PERSISTENT_LOCK"
                and gov_direction == "STRESSED_BUT_STABLE"
                and velocity in ("MINIMAL", "SLOW")
            )
            or (
                trajectory == "CONSTITUTIONALLY_CONSTRAINED"
                and runtime_posture == "BLOCKED"
                and delta_state == "STABLE_NO_MATERIAL_CHANGE"
            )
        )

    def _match_mixed_forward() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            delta_state == "MIXED_TRANSITION"
            or hist_direction == "mixed"
            or trajectory == "GOVERNANCE_REGRESSION_RISK"
            or regime_shift == "GUARDED"
        )

    def _match_stabilization() -> bool:
        return (
            gov_direction in ("STABILIZING", "IMPROVING")
            or persistence == "STABILIZING"
            or trajectory in ("GOVERNANCE_IMPROVING", "GOVERNANCE_ACCELERATING")
        )

    def _match_operationally_stable() -> bool:
        return (
            gov_direction == "OPERATIONALLY_STABLE"
            or verdict_state == "GOVERNANCE_OPERATIONALLY_STABLE"
            or persistence == "OPERATIONALLY_DURABLE"
        )

    outlook_matchers = {
        "STRUCTURAL_DETERIORATION_RISK": _match_structural_deterioration,
        "PERSISTENT_CONSTRAINT": _match_persistent_constraint,
        "MIXED_FORWARD_PATH": _match_mixed_forward,
        "STABILIZATION_PATH": _match_stabilization,
        "OPERATIONALLY_STABLE_PATH": _match_operationally_stable,
    }

    forward_outlook = "PERSISTENT_CONSTRAINT"
    for candidate in _GCC_FORWARD_OUTLOOK_PRIORITY:
        fn = outlook_matchers.get(candidate)
        if fn and fn():
            forward_outlook = candidate
            break

    if constitutional_lock_stable or forward_outlook == "PERSISTENT_CONSTRAINT":
        transition_probability = "LOW"
    elif forward_outlook == "STRUCTURAL_DETERIORATION_RISK":
        transition_probability = "HIGH"
    elif forward_outlook == "MIXED_FORWARD_PATH" or regime_shift == "GUARDED":
        transition_probability = "GUARDED"
    elif regime_shift == "MATERIAL":
        transition_probability = "MATERIAL"
    elif delta_state == "REGIME_SHIFT_EMERGING":
        transition_probability = "HIGH"
    else:
        transition_probability = regime_shift if regime_shift != "NONE" else "LOW"

    if forward_outlook == "STRUCTURAL_DETERIORATION_RISK" or velocity == "RAPID":
        stability_risk_forecast = "SEVERE"
    elif gov_direction == "DETERIORATING" or drift_risk == "SEVERE":
        stability_risk_forecast = "MATERIAL"
    elif drift_risk == "MATERIAL" or regression_risk == "MODERATE":
        stability_risk_forecast = "GUARDED"
    elif constitutional_lock_stable or (
        forward_outlook == "PERSISTENT_CONSTRAINT" and velocity in ("MINIMAL", "SLOW")
    ):
        stability_risk_forecast = "LOW"
    elif drift_risk in ("LOW", "NONE") and delta_state == "STABLE_NO_MATERIAL_CHANGE":
        stability_risk_forecast = "MINIMAL"
    else:
        stability_risk_forecast = "GUARDED"

    if forward_outlook == "STRUCTURAL_DETERIORATION_RISK" or transition_probability == "HIGH":
        warning_sensitivity = "IMMEDIATE" if velocity == "RAPID" else "HIGH"
    elif forward_outlook == "MIXED_FORWARD_PATH":
        warning_sensitivity = "GUARDED"
    elif constitutional_lock_stable or forward_outlook == "PERSISTENT_CONSTRAINT":
        warning_sensitivity = "LOW"
    elif forward_outlook == "OPERATIONALLY_STABLE_PATH":
        warning_sensitivity = "PASSIVE"
    elif forward_outlook == "STABILIZATION_PATH":
        warning_sensitivity = "LOW"
    else:
        warning_sensitivity = "GUARDED"

    warning_drivers: List[str] = []
    if forward_outlook == "PERSISTENT_CONSTRAINT":
        warning_drivers.extend(
            [
                "Monitor contradiction persistence",
                "Watch trustworthiness drift",
                "Observe governance coherence stability",
                "Reassess regime transition signals",
            ]
        )
    elif forward_outlook == "STRUCTURAL_DETERIORATION_RISK":
        warning_drivers.extend(
            [
                "Monitor trust deterioration",
                "Watch governance fragmentation acceleration",
                "Observe contradiction intensity",
                "Reassess escalation readiness",
            ]
        )
    elif forward_outlook == "STABILIZATION_PATH":
        warning_drivers.extend(
            [
                "Observe contradiction reduction",
                "Monitor trust recovery",
                "Watch evidence integrity stabilization",
                "Reassess governance coherence",
            ]
        )
    elif forward_outlook == "MIXED_FORWARD_PATH":
        warning_drivers.extend(
            [
                "Monitor mixed directional signals",
                "Watch trust and alignment divergence",
                "Observe regime transition indicators",
            ]
        )
    elif forward_outlook == "OPERATIONALLY_STABLE_PATH":
        warning_drivers.extend(
            [
                "Monitor routine drift signals",
                "Watch constitutional posture stability",
            ]
        )
    else:
        warning_drivers.append("Monitor governance directional signals")

    for warning in (scenario.get("early_warnings") or [])[:2]:
        if warning not in warning_drivers and len(warning_drivers) < 6:
            warning_drivers.append(warning)
    if (
        trust_summary in ("BROKEN", "WEAK")
        and "Watch trustworthiness drift" not in warning_drivers
        and len(warning_drivers) < 6
    ):
        warning_drivers.append("Watch trustworthiness drift")
    if (
        alignment.get("alignment_state")
        in (
            "FRAGMENTATION_ELEVATED",
            "GOVERNANCE_INTERNAL_CONFLICT",
        )
        and len(warning_drivers) < 6
    ):
        warning_drivers.append("Watch alignment fragmentation")
    warning_drivers = warning_drivers[:6]

    return {
        "forward_outlook": forward_outlook,
        "outlook_display": _GCC_FORWARD_OUTLOOK_DISPLAY.get(
            forward_outlook, forward_outlook.replace("_", " ").title()
        ),
        "regime_transition_probability": transition_probability,
        "stability_risk_forecast": stability_risk_forecast,
        "early_warning_sensitivity": warning_sensitivity,
        "early_warning_drivers": warning_drivers,
        "forecast_memo": _GCC_FORWARD_FORECAST_MEMO.get(forward_outlook, ""),
    }


def _gcc_detect_governance_scenario_matrix(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    trend: Dict[str, Any],
    temporal: Dict[str, Any],
    delta: Dict[str, Any],
    forward: Dict[str, Any],
    scenario: Dict[str, Any],
    intervention: Dict[str, Any],
    evidence: Dict[str, Any],
    alignment: Dict[str, Any],
    coherence: Dict[str, Any],
    improvement: Dict[str, Any],
    tension: Dict[str, Any],
) -> Dict[str, Any]:
    verdict_state = executive["verdict_state"]
    severity = attention["severity"]
    runtime_posture = executive["runtime_posture"]
    trust_summary = executive["trust_summary"]
    gov_direction = trend["governance_direction"]
    velocity = trend["trend_velocity"]
    drift_risk = trend["drift_risk"]
    structural_outlook = trend["structural_outlook"]
    intervention_posture = intervention.get("intervention_posture", "")
    delta_state = delta.get("delta_state", "")
    forward_outlook = forward.get("forward_outlook", "")
    transition_prob = forward.get("regime_transition_probability", "LOW")
    scenario_state = scenario.get("scenario_state", "")
    alignment_state = alignment.get("alignment_state", "")
    coherence_state = coherence.get("coherence_state", "")
    improvement_state = improvement.get("improvement_state", "")
    contradiction_count = int(tension.get("contradiction_count", 0) or 0)
    persistence = temporal.get("governance_persistence", "")

    constitutional_lock_stable = (
        (
            severity == "CRITICAL_LOCK"
            or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
            or runtime_posture == "BLOCKED"
        )
        and gov_direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
        and delta_state == "STABLE_NO_MATERIAL_CHANGE"
    )

    def _match_constitutional_constraint() -> bool:
        return (
            constitutional_lock_stable
            or (forward_outlook == "PERSISTENT_CONSTRAINT" and runtime_posture == "BLOCKED")
            or (
                intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
                and gov_direction == "STRESSED_BUT_STABLE"
                and velocity in ("MINIMAL", "SLOW")
            )
        )

    def _match_fragmentation_stress() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            alignment_state
            in (
                "FRAGMENTATION_ELEVATED",
                "GOVERNANCE_INTERNAL_CONFLICT",
                "INSTITUTIONAL_CONSENSUS_BROKEN",
            )
            or scenario_state == "FRAGMENTATION_RISK_ELEVATED"
            or (
                contradiction_count >= 2
                and coherence_state
                in (
                    "FRAGMENTED_REASONING_CHAIN",
                    "LOW_COHERENCE_GOVERNANCE",
                )
            )
        )

    def _match_regime_transition() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            delta_state == "REGIME_SHIFT_EMERGING"
            or forward_outlook == "MIXED_FORWARD_PATH"
            or delta_state == "MIXED_TRANSITION"
            or persistence == "TRANSITIONING"
        )

    def _match_stabilization() -> bool:
        return (
            forward_outlook == "STABILIZATION_PATH"
            or gov_direction in ("STABILIZING", "IMPROVING")
            or improvement_state
            in (
                "CONTRADICTIONS_DECLINING",
                "GOVERNANCE_MATURITY_IMPROVING",
                "GOVERNANCE_ADAPTATION_EMERGING",
            )
            or scenario_state == "RECOVERABLE_GOVERNANCE_STRESS"
        )

    def _match_operationally_stable() -> bool:
        return (
            verdict_state == "GOVERNANCE_OPERATIONALLY_STABLE"
            or gov_direction == "OPERATIONALLY_STABLE"
            or forward_outlook == "OPERATIONALLY_STABLE_PATH"
        )

    scenario_matchers = {
        "CONSTITUTIONAL_CONSTRAINT_SCENARIO": _match_constitutional_constraint,
        "FRAGMENTATION_STRESS_SCENARIO": _match_fragmentation_stress,
        "REGIME_TRANSITION_SCENARIO": _match_regime_transition,
        "STABILIZATION_SCENARIO": _match_stabilization,
        "OPERATIONALLY_STABLE_SCENARIO": _match_operationally_stable,
    }

    governance_scenario = "CONSTITUTIONAL_CONSTRAINT_SCENARIO"
    for candidate in _GCC_SCENARIO_PRIORITY:
        fn = scenario_matchers.get(candidate)
        if fn and fn():
            governance_scenario = candidate
            break

    next_regime_map = {
        "CONSTITUTIONAL_CONSTRAINT_SCENARIO": "CONSTITUTIONAL_LOCK_PERSISTS",
        "FRAGMENTATION_STRESS_SCENARIO": "FRAGMENTATION_ELEVATES",
        "REGIME_TRANSITION_SCENARIO": "MIXED_TRANSITION",
        "STABILIZATION_SCENARIO": "GOVERNANCE_STABILIZES",
        "OPERATIONALLY_STABLE_SCENARIO": "OPERATIONAL_STABILITY",
    }
    most_likely_next_regime = next_regime_map.get(governance_scenario, "MIXED_TRANSITION")

    if constitutional_lock_stable or governance_scenario == "CONSTITUTIONAL_CONSTRAINT_SCENARIO":
        deterioration_trigger_risk = "LOW"
    elif (
        forward_outlook == "STRUCTURAL_DETERIORATION_RISK"
        or delta_state == "MATERIAL_DETERIORATION"
    ):
        deterioration_trigger_risk = "HIGH"
    elif drift_risk in ("SEVERE", "MATERIAL") or transition_prob == "HIGH":
        deterioration_trigger_risk = "MATERIAL"
    elif contradiction_count >= 2 and velocity in ("MODERATE", "FAST", "RAPID"):
        deterioration_trigger_risk = "GUARDED"
    elif trust_summary in ("BROKEN", "WEAK") and delta_state != "STABLE_NO_MATERIAL_CHANGE":
        deterioration_trigger_risk = "GUARDED"
    else:
        deterioration_trigger_risk = "LOW"

    if governance_scenario == "OPERATIONALLY_STABLE_SCENARIO":
        stabilization_trigger_probability = "HIGH"
    elif governance_scenario == "STABILIZATION_SCENARIO":
        stabilization_trigger_probability = "MODERATE"
    elif (
        constitutional_lock_stable
        or structural_outlook == "RECOVERABLE"
        or scenario_state == "RECOVERABLE_GOVERNANCE_STRESS"
    ):
        stabilization_trigger_probability = "GUARDED"
    elif improvement_state in (
        "INSTITUTIONAL_IMPROVEMENT_POSSIBLE",
        "GOVERNANCE_ADAPTATION_EMERGING",
    ):
        stabilization_trigger_probability = "GUARDED"
    elif forward_outlook == "STRUCTURAL_DETERIORATION_RISK":
        stabilization_trigger_probability = "NONE"
    else:
        stabilization_trigger_probability = "LOW"

    deterioration_triggers: List[str] = []
    if contradiction_count >= 1:
        deterioration_triggers.append("Rising contradiction intensity")
    if trust_summary in ("BROKEN", "WEAK"):
        deterioration_triggers.append("Trust deterioration")
    if alignment_state in (
        "FRAGMENTATION_ELEVATED",
        "GOVERNANCE_INTERNAL_CONFLICT",
        "INSTITUTIONAL_CONSENSUS_BROKEN",
    ):
        deterioration_triggers.append("Governance fragmentation acceleration")
    if coherence_state in (
        "FRAGMENTED_REASONING_CHAIN",
        "INTERNALLY_INCONSISTENT_GOVERNANCE",
        "LOW_COHERENCE_GOVERNANCE",
    ):
        deterioration_triggers.append("Coherence weakening")
    if evidence.get("evidence_state") in (
        "INSTITUTIONAL_CONFIDENCE_UNSUPPORTED",
        "EVIDENCE_QUALITY_DEGRADING",
    ):
        deterioration_triggers.append("Evidence integrity degradation")
    if attention.get("escalation_urgency") in ("MATERIAL", "IMMEDIATE"):
        deterioration_triggers.append("Escalation disagreement growth")
    if not deterioration_triggers:
        deterioration_triggers.append("No active deterioration triggers identified")
    deterioration_triggers = deterioration_triggers[:5]

    stabilization_triggers: List[str] = []
    if improvement_state == "CONTRADICTIONS_DECLINING" or contradiction_count <= 1:
        stabilization_triggers.append("Contradiction reduction")
    if trust_summary in ("ACCEPTABLE", "STRONG") or improvement_state in (
        "GOVERNANCE_MATURITY_IMPROVING",
    ):
        stabilization_triggers.append("Trust stabilization")
    if coherence_state in (
        "MODERATELY_COHERENT_GOVERNANCE",
        "HIGHLY_COHERENT_GOVERNANCE",
        "LOGICALLY_CONSTRAINED_BUT_COHERENT",
    ):
        stabilization_triggers.append("Governance coherence improvement")
    if evidence.get("evidence_state") in (
        "CONFIDENCE_SUPPORTED",
        "EVIDENCE_INTEGRITY_STRONG",
        "PARTIALLY_SUPPORTED_CONFIDENCE",
    ):
        stabilization_triggers.append("Evidence integrity strengthening")
    if alignment_state == "GOVERNANCE_CONSENSUS_FORMING":
        stabilization_triggers.append("Alignment stabilization")
    if structural_outlook == "RECOVERABLE":
        stabilization_triggers.append("Recoverable institutional posture")
    if governance_scenario == "CONSTITUTIONAL_CONSTRAINT_SCENARIO":
        for trigger in (
            "Contradiction reduction",
            "Trust stabilization",
            "Governance coherence improvement",
        ):
            if trigger not in stabilization_triggers:
                stabilization_triggers.insert(0, trigger)
    if not stabilization_triggers:
        stabilization_triggers.extend(
            [
                "Contradiction reduction",
                "Trust stabilization",
                "Governance coherence improvement",
            ]
        )
    stabilization_triggers = stabilization_triggers[:5]

    return {
        "governance_scenario": governance_scenario,
        "scenario_display": _GCC_SCENARIO_DISPLAY.get(
            governance_scenario, governance_scenario.replace("_", " ").title()
        ),
        "most_likely_next_regime": most_likely_next_regime,
        "next_regime_display": _GCC_NEXT_REGIME_DISPLAY.get(
            most_likely_next_regime, most_likely_next_regime.replace("_", " ").title()
        ),
        "deterioration_trigger_risk": deterioration_trigger_risk,
        "stabilization_trigger_probability": stabilization_trigger_probability,
        "deterioration_triggers": deterioration_triggers,
        "stabilization_triggers": stabilization_triggers,
        "scenario_memo": _GCC_SCENARIO_MEMO.get(governance_scenario, ""),
    }


def _gcc_detect_governance_decision_brief(
    *,
    executive: Dict[str, Any],
    attention: Dict[str, Any],
    trend: Dict[str, Any],
    signal_conf: Dict[str, Any],
    playbook: Dict[str, Any],
    temporal: Dict[str, Any],
    delta: Dict[str, Any],
    forward: Dict[str, Any],
    scenario_matrix: Dict[str, Any],
    intervention: Dict[str, Any],
) -> Dict[str, Any]:
    verdict_state = executive["verdict_state"]
    severity = attention["severity"]
    runtime_posture = executive["runtime_posture"]
    trust_summary = executive["trust_summary"]
    gov_direction = trend["governance_direction"]
    velocity = trend["trend_velocity"]
    drift_risk = trend["drift_risk"]
    intervention_posture = intervention.get("intervention_posture", "")
    delta_state = delta.get("delta_state", "")
    forward_outlook = forward.get("forward_outlook", "")
    priority_action = playbook.get("priority_action", "")
    deterioration_risk = scenario_matrix.get("deterioration_trigger_risk", "LOW")
    governance_scenario = scenario_matrix.get("governance_scenario", "")
    attention_level = attention.get("attention_level", "MODERATE")
    reliability = signal_conf.get("reliability_level", "GUARDED")

    constitutional_lock_stable = (
        (
            severity == "CRITICAL_LOCK"
            or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
            or runtime_posture == "BLOCKED"
        )
        and gov_direction == "STRESSED_BUT_STABLE"
        and velocity in ("MINIMAL", "SLOW")
        and delta_state == "STABLE_NO_MATERIAL_CHANGE"
        and deterioration_risk == "LOW"
    )

    def _match_locked_observe_only() -> bool:
        return constitutional_lock_stable or (
            priority_action == "MAINTAIN_CONSTITUTIONAL_LOCK"
            and gov_direction == "STRESSED_BUT_STABLE"
            and velocity in ("MINIMAL", "SLOW")
            and deterioration_risk == "LOW"
            and forward_outlook == "PERSISTENT_CONSTRAINT"
        )

    def _match_locked_heightened() -> bool:
        if _match_locked_observe_only():
            return False
        return (
            severity == "CRITICAL_LOCK"
            or runtime_posture == "BLOCKED"
            or verdict_state == "CONSTITUTIONALLY_LOCKED_GOVERNANCE"
            or intervention_posture == "CONSTITUTIONAL_LOCK_REQUIRED"
        ) and (
            attention_level in ("HIGH", "IMMEDIATE")
            or severity == "HIGH_RISK"
            or deterioration_risk in ("GUARDED", "MATERIAL", "HIGH")
            or forward.get("early_warning_sensitivity") in ("GUARDED", "HIGH", "IMMEDIATE")
        )

    def _match_repair_required() -> bool:
        if constitutional_lock_stable:
            return False
        return (
            priority_action
            in (
                "IMPROVE_CONFIDENCE_INTEGRITY",
                "REDUCE_FRAGMENTATION",
            )
            or reliability in ("BROKEN", "WEAK")
            or trust_summary in ("BROKEN", "WEAK")
            or governance_scenario == "FRAGMENTATION_STRESS_SCENARIO"
        )

    def _match_transition_watch() -> bool:
        return (
            delta_state in ("REGIME_SHIFT_EMERGING", "MIXED_TRANSITION")
            or forward_outlook == "MIXED_FORWARD_PATH"
            or governance_scenario == "REGIME_TRANSITION_SCENARIO"
            or priority_action == "PREPARE_ESCALATION_REVIEW"
            or forward.get("regime_transition_probability") in ("GUARDED", "MATERIAL", "HIGH")
        )

    def _match_stable_continue() -> bool:
        return (
            verdict_state == "GOVERNANCE_OPERATIONALLY_STABLE"
            or gov_direction == "OPERATIONALLY_STABLE"
            or (severity == "NORMAL" and delta_state == "STABLE_NO_MATERIAL_CHANGE")
        )

    brief_matchers = {
        "LOCKED_OBSERVE_ONLY": _match_locked_observe_only,
        "LOCKED_HEIGHTENED_MONITORING": _match_locked_heightened,
        "GOVERNANCE_REPAIR_REQUIRED": _match_repair_required,
        "TRANSITION_WATCH": _match_transition_watch,
        "STABLE_CONTINUE_MONITORING": _match_stable_continue,
    }

    final_brief = "LOCKED_OBSERVE_ONLY"
    for candidate in _GCC_DECISION_BRIEF_PRIORITY:
        fn = brief_matchers.get(candidate)
        if fn and fn():
            final_brief = candidate
            break

    if final_brief in ("LOCKED_OBSERVE_ONLY", "LOCKED_HEIGHTENED_MONITORING"):
        governance_mode = "CONSTITUTIONAL_LOCK_MODE"
    elif governance_scenario == "FRAGMENTATION_STRESS_SCENARIO":
        governance_mode = "CONTAINMENT_MODE"
    elif final_brief == "GOVERNANCE_REPAIR_REQUIRED":
        governance_mode = "REPAIR_MODE"
    elif final_brief == "TRANSITION_WATCH":
        governance_mode = "TRANSITION_MODE"
    else:
        governance_mode = "OBSERVATION_MODE"

    immediate_instruction = _GCC_DECISION_INSTRUCTION.get(
        final_brief, "CONTINUE_ROUTINE_MONITORING"
    )

    monitoring = playbook.get("monitoring_priorities") or []
    det_triggers = scenario_matrix.get("deterioration_triggers") or []
    if trust_summary in ("BROKEN", "WEAK") and "Trustworthiness drift" not in monitoring:
        primary_watch = "Trustworthiness drift"
    elif "Trust deterioration" in det_triggers:
        primary_watch = "Trustworthiness drift"
    elif "Rising contradiction intensity" in det_triggers:
        primary_watch = "Contradiction persistence"
    elif "Governance fragmentation acceleration" in det_triggers:
        primary_watch = "Alignment fragmentation"
    elif "Coherence weakening" in det_triggers:
        primary_watch = "Governance coherence weakening"
    elif forward.get("regime_transition_probability") in ("GUARDED", "MATERIAL", "HIGH"):
        primary_watch = "Regime transition signals"
    elif monitoring:
        primary_watch = monitoring[0]
    else:
        primary_watch = "Governance directional signals"

    blocked = playbook.get("blocked_actions") or []
    if runtime_posture == "BLOCKED" or final_brief.startswith("LOCKED"):
        primary_blocked = "Runtime enablement"
    elif blocked:
        primary_blocked = blocked[0]
    else:
        primary_blocked = "Premature runtime enablement"

    return {
        "final_brief": final_brief,
        "brief_display": _GCC_DECISION_BRIEF_DISPLAY.get(
            final_brief, final_brief.replace("_", " ").title()
        ),
        "governance_mode": governance_mode,
        "mode_display": _GCC_GOVERNANCE_MODE_DISPLAY.get(
            governance_mode, governance_mode.replace("_", " ").title()
        ),
        "immediate_instruction": immediate_instruction,
        "primary_watch_condition": primary_watch,
        "primary_blocked_condition": primary_blocked,
        "decision_memo": _GCC_DECISION_BRIEF_MEMO.get(final_brief, ""),
    }


def _gcc_render_decision_brief_card(brief: Dict[str, Any]) -> None:
    st.markdown("#### Operator Decision Brief")
    st.caption("Executive compression — final governance posture in one view. **Read-only.**")

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Final Operator Brief", brief["brief_display"])
    r1c2.metric("Governance Mode", brief["mode_display"])
    r1c3.metric("Immediate Instruction", brief["immediate_instruction"])

    r2c1, r2c2 = st.columns(2)
    r2c1.metric("Watch Condition", brief["primary_watch_condition"])
    r2c2.metric("Blocked Condition", brief["primary_blocked_condition"])

    final_brief = brief["final_brief"]
    if final_brief == "LOCKED_OBSERVE_ONLY":
        st.warning(brief["decision_memo"])
    elif final_brief in ("LOCKED_HEIGHTENED_MONITORING", "TRANSITION_WATCH"):
        st.warning(brief["decision_memo"])
    elif final_brief == "GOVERNANCE_REPAIR_REQUIRED":
        st.error(brief["decision_memo"])
    elif final_brief == "STABLE_CONTINUE_MONITORING":
        st.success(brief["decision_memo"])
    else:
        st.info(brief["decision_memo"])

    st.markdown("---")


def _gcc_render_executive_summary_intelligence(
    *,
    stack: Dict[str, Any],
    dossier_summary: Dict[str, Any],
) -> None:
    executive = _gcc_detect_executive_verdict(
        dossier_summary=dossier_summary,
        regime=stack["regime"],
        tension=stack["tension"],
        decision=stack["decision"],
        scenario=stack["scenario"],
        intervention=stack["intervention"],
        evidence=stack["evidence"],
        alignment=stack["alignment"],
        quality=stack["quality"],
        improvement=stack["improvement"],
        stability=stack["stability"],
    )

    st.markdown("### Governance Executive Summary & Institutional Verdict Intelligence")
    st.caption(
        "Operator cockpit — synthesized governance posture at a glance. "
        "**Read-only executive view. Drill down in sections below.**"
    )

    attention = _gcc_detect_governance_attention(
        executive=executive,
        dossier_summary=dossier_summary,
        regime=stack["regime"],
        tension=stack["tension"],
        decision=stack["decision"],
        scenario=stack["scenario"],
        intervention=stack["intervention"],
        evidence=stack["evidence"],
        alignment=stack["alignment"],
        quality=stack["quality"],
        failure=stack["failure"],
        integrity=stack["integrity"],
    )
    trend = _gcc_detect_governance_trend(
        executive=executive,
        attention=attention,
        hist=stack["hist"],
        regime=stack["regime"],
        forecast=stack["forecast"],
        tension=stack["tension"],
        stability=stack["stability"],
        improvement=stack["improvement"],
        resilience=stack["resilience"],
        scenario=stack["scenario"],
        intervention=stack["intervention"],
        evidence=stack["evidence"],
        alignment=stack["alignment"],
        quality=stack["quality"],
        failure=stack["failure"],
    )
    signal_conf = _gcc_detect_governance_signal_confidence(
        executive=executive,
        attention=attention,
        trend=trend,
        hist=stack["hist"],
        evidence=stack["evidence"],
        alignment=stack["alignment"],
        quality=stack["quality"],
        decision=stack["decision"],
        tension=stack["tension"],
        stability=stack["stability"],
        integrity=stack["integrity"],
        coherence=stack["coherence"],
    )
    playbook = _gcc_detect_governance_operator_playbook(
        executive=executive,
        attention=attention,
        trend=trend,
        signal_conf=signal_conf,
        dossier_summary=dossier_summary,
        intervention=stack["intervention"],
        evidence=stack["evidence"],
        alignment=stack["alignment"],
        quality=stack["quality"],
        tension=stack["tension"],
        scenario=stack["scenario"],
    )
    temporal = _gcc_detect_governance_temporal(
        executive=executive,
        attention=attention,
        trend=trend,
        playbook=playbook,
        hist=stack["hist"],
        regime=stack["regime"],
        stability=stack["stability"],
        improvement=stack["improvement"],
        resilience=stack["resilience"],
        intervention=stack["intervention"],
        decision=stack["decision"],
    )
    delta = _gcc_detect_governance_delta(
        executive=executive,
        attention=attention,
        trend=trend,
        temporal=temporal,
        hist=stack["hist"],
        regime=stack["regime"],
        tension=stack["tension"],
        evidence=stack["evidence"],
        alignment=stack["alignment"],
        quality=stack["quality"],
        coherence=stack["coherence"],
        improvement=stack["improvement"],
        stability=stack["stability"],
        intervention=stack["intervention"],
    )
    forward = _gcc_detect_governance_forward_forecast(
        executive=executive,
        attention=attention,
        trend=trend,
        temporal=temporal,
        delta=delta,
        forecast=stack["forecast"],
        scenario=stack["scenario"],
        resilience=stack["resilience"],
        intervention=stack["intervention"],
        evidence=stack["evidence"],
        alignment=stack["alignment"],
        tension=stack["tension"],
        hist=stack["hist"],
        regime=stack["regime"],
    )
    scenario_matrix = _gcc_detect_governance_scenario_matrix(
        executive=executive,
        attention=attention,
        trend=trend,
        temporal=temporal,
        delta=delta,
        forward=forward,
        scenario=stack["scenario"],
        intervention=stack["intervention"],
        evidence=stack["evidence"],
        alignment=stack["alignment"],
        coherence=stack["coherence"],
        improvement=stack["improvement"],
        tension=stack["tension"],
    )
    decision_brief = _gcc_detect_governance_decision_brief(
        executive=executive,
        attention=attention,
        trend=trend,
        signal_conf=signal_conf,
        playbook=playbook,
        temporal=temporal,
        delta=delta,
        forward=forward,
        scenario_matrix=scenario_matrix,
        intervention=stack["intervention"],
    )
    _gcc_render_decision_brief_card(decision_brief)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Institutional Verdict", executive["verdict_display"])
    c2.metric("Stability", executive["stability_summary"])
    c3.metric("Trust", executive["trust_summary"])
    c4.metric("Alignment", executive["alignment_summary"])
    c5.metric("Runtime Posture", executive["runtime_posture"])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Severity", attention["severity_display"])
    s2.metric("Operator Attention", attention["attention_level"])
    s3.metric("Review Cadence", attention["review_cadence"])
    s4.metric("Escalation Urgency", attention["escalation_urgency"])

    severity = attention["severity"]
    if severity == "CRITICAL_LOCK":
        st.warning(attention["attention_memo"])
    elif severity == "HIGH_RISK":
        st.error(attention["attention_memo"])
    elif severity == "ELEVATED":
        st.warning(attention["attention_memo"])
    elif severity == "NORMAL":
        st.success(attention["attention_memo"])
    else:
        st.info(attention["attention_memo"])

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Governance Direction", trend["direction_display"])
    t2.metric("Trend Velocity", trend["trend_velocity"])
    t3.metric("Drift Risk", trend["drift_risk"])
    t4.metric("Structural Outlook", trend["structural_outlook"])

    gov_dir = trend["governance_direction"]
    if gov_dir == "RAPIDLY_DETERIORATING":
        st.error(trend["trend_memo"])
    elif gov_dir == "DETERIORATING":
        st.warning(trend["trend_memo"])
    elif gov_dir == "STRESSED_BUT_STABLE":
        st.info(trend["trend_memo"])
    elif gov_dir == "OPERATIONALLY_STABLE":
        st.success(trend["trend_memo"])
    else:
        st.info(trend["trend_memo"])

    cf1, cf2, cf3, cf4 = st.columns(4)
    cf1.metric("Cockpit Confidence", signal_conf["confidence_display"])
    cf2.metric("Reliability Level", signal_conf["reliability_level"])
    cf3.metric("False Alarm Risk", signal_conf["false_alarm_risk"])
    cf4.metric("Interpretation Stability", signal_conf["interpretation_stability"])

    conf_level = signal_conf["cockpit_confidence"]
    if conf_level in ("VERY_LOW", "LOW"):
        st.error(signal_conf["confidence_memo"])
    elif conf_level == "GUARDED":
        st.info(signal_conf["confidence_memo"])
    elif conf_level in ("HIGH", "VERY_HIGH"):
        st.success(signal_conf["confidence_memo"])
    else:
        st.info(signal_conf["confidence_memo"])

    st.metric("Priority Operator Action", playbook["priority_action"])

    pb_left, pb_right = st.columns(2)
    with pb_left:
        st.markdown("**Immediate Actions**")
        for action in playbook["immediate_actions"]:
            st.markdown(f"- {action}")
        st.markdown("**Monitoring Priorities**")
        for target in playbook["monitoring_priorities"]:
            st.markdown(f"- {target}")
    with pb_right:
        st.markdown("**Deferred Actions**")
        for action in playbook["deferred_actions"]:
            st.markdown(f"- {action}")
        st.markdown("**Blocked Actions**")
        for action in playbook["blocked_actions"]:
            st.markdown(f"- {action}")

    if playbook["priority_action"] in ("PREPARE_ESCALATION_REVIEW", "REDUCE_FRAGMENTATION"):
        st.warning(playbook["discipline_memo"])
    elif playbook["priority_action"] == "MAINTAIN_CONSTITUTIONAL_LOCK":
        st.info(playbook["discipline_memo"])
    else:
        st.info(playbook["discipline_memo"])

    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
    tm1.metric("Governance Persistence", temporal["persistence_display"])
    tm2.metric("Regime Duration", temporal["regime_duration"])
    tm3.metric("Momentum Strength", temporal["momentum_strength"])
    tm4.metric("Recovery Horizon", temporal["recovery_horizon"])
    tm5.metric("Governance Fatigue Risk", temporal["fatigue_risk"])

    persistence = temporal["governance_persistence"]
    if persistence == "ENTRENCHED_INSTABILITY":
        st.error(temporal["temporal_memo"])
    elif persistence in ("PERSISTENT_LOCK", "TRANSITIONING"):
        st.info(temporal["temporal_memo"])
    elif persistence == "OPERATIONALLY_DURABLE":
        st.success(temporal["temporal_memo"])
    else:
        st.info(temporal["temporal_memo"])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Governance Delta State", delta["delta_display"])
    d2.metric("Largest Positive Change", delta["positive_display"])
    d3.metric("Largest Negative Change", delta["negative_display"])
    d4.metric("Regime Shift Probability", delta["regime_shift_probability"])

    st.markdown("**Dominant Change Drivers**")
    for driver in delta["change_drivers"]:
        st.markdown(f"- {driver}")

    delta_state = delta["delta_state"]
    if delta_state == "REGIME_SHIFT_EMERGING":
        st.error(delta["change_memo"])
    elif delta_state == "MATERIAL_DETERIORATION":
        st.warning(delta["change_memo"])
    elif delta_state == "MATERIAL_IMPROVEMENT":
        st.success(delta["change_memo"])
    else:
        st.info(delta["change_memo"])

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Forward Governance Outlook", forward["outlook_display"])
    f2.metric("Regime Transition Probability", forward["regime_transition_probability"])
    f3.metric("Stability Risk Forecast", forward["stability_risk_forecast"])
    f4.metric("Early Warning Sensitivity", forward["early_warning_sensitivity"])

    st.markdown("**Early Warning Drivers**")
    for driver in forward["early_warning_drivers"]:
        st.markdown(f"- {driver}")

    forward_outlook = forward["forward_outlook"]
    if forward_outlook == "STRUCTURAL_DETERIORATION_RISK":
        st.error(forward["forecast_memo"])
    elif forward_outlook == "MIXED_FORWARD_PATH":
        st.warning(forward["forecast_memo"])
    elif forward_outlook == "STABILIZATION_PATH":
        st.success(forward["forecast_memo"])
    else:
        st.info(forward["forecast_memo"])

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Governance Scenario State", scenario_matrix["scenario_display"])
    sc2.metric("Most Likely Next Regime", scenario_matrix["next_regime_display"])
    sc3.metric("Deterioration Trigger Risk", scenario_matrix["deterioration_trigger_risk"])
    sc4.metric(
        "Stabilization Trigger Probability",
        scenario_matrix["stabilization_trigger_probability"],
    )

    trig_left, trig_right = st.columns(2)
    with trig_left:
        st.markdown("**Deterioration Triggers**")
        for trigger in scenario_matrix["deterioration_triggers"]:
            st.markdown(f"- {trigger}")
    with trig_right:
        st.markdown("**Stabilization Triggers**")
        for trigger in scenario_matrix["stabilization_triggers"]:
            st.markdown(f"- {trigger}")

    gov_scenario = scenario_matrix["governance_scenario"]
    if gov_scenario == "FRAGMENTATION_STRESS_SCENARIO":
        st.warning(scenario_matrix["scenario_memo"])
    elif gov_scenario == "REGIME_TRANSITION_SCENARIO":
        st.warning(scenario_matrix["scenario_memo"])
    elif gov_scenario == "STABILIZATION_SCENARIO":
        st.success(scenario_matrix["scenario_memo"])
    else:
        st.info(scenario_matrix["scenario_memo"])

    state = executive["verdict_state"]
    if state in ("CONSTITUTIONALLY_LOCKED_GOVERNANCE", "SYSTEMIC_GOVERNANCE_INSTABILITY"):
        st.error(executive["executive_memo"])
    elif state == "GOVERNANCE_UNDER_INSTITUTIONAL_STRESS":
        st.warning(executive["executive_memo"])
    elif state == "GOVERNANCE_OPERATIONALLY_STABLE":
        st.success(executive["executive_memo"])
    else:
        st.info(executive["executive_memo"])

    st.markdown("**Top Governance Risks**")
    for risk in executive["top_risks"]:
        if state in ("CONSTITUTIONALLY_LOCKED_GOVERNANCE", "SYSTEMIC_GOVERNANCE_INSTABILITY"):
            st.warning(f"• {risk}")
        else:
            st.markdown(f'<div class="gcc-hist-metric">• {risk}</div>', unsafe_allow_html=True)

    st.metric("Recommended Operator Action", executive["operator_action"])

    with st.expander("Why governance reached this verdict", expanded=False):
        for driver in executive["why_drivers"]:
            st.markdown(f"- {driver}")
        st.markdown(
            f"- **Runtime mutation allowed:** `{bool(_gcc_get(dossier_summary, 'runtime_mutation_allowed', False))}`"
        )


def page_governance_command_center() -> None:
    """Governance Command Center — institutional operator view of runtime governance (read-only)."""
    st.title("🏛 Governance Command Center")
    st.caption(
        "Phase 2 institutional operator dashboard. Surfaces runtime governance status, "
        "constitutional posture, institutional verdict, and human escalation dossier. "
        "**Read-only observability — no execution, no broker calls, no runtime mutation.**"
    )

    dossier_summary_early = (
        _ad_load_json(GCC_DOSSIER_SUMMARY_PATH, "Human escalation dossier summary") or {}
    )
    guard_snap = _ad_load_json(GUARD_SNAPSHOT_PATH, "Guard snapshot") or {}
    render_gcc_operations_overview(
        dossier_summary=dossier_summary_early,
        guard_snapshot=guard_snap,
    )

    readiness = _ad_load_json(GCC_READINESS_SUMMARY_PATH, "Runtime readiness summary") or {}
    admission = _ad_load_json(GCC_ADMISSION_SUMMARY_PATH, "Runtime admission summary") or {}
    eligibility = (
        _ad_load_json(GCC_ELIGIBILITY_SUMMARY_PATH, "Constitutional eligibility summary") or {}
    )
    recommendation = (
        _ad_load_json(GCC_RECOMMENDATION_SUMMARY_PATH, "Enablement recommendation summary") or {}
    )
    review = _ad_load_json(GCC_REVIEW_SUMMARY_PATH, "Enablement review summary") or {}
    verdict = _ad_load_json(GCC_VERDICT_SUMMARY_PATH, "Institutional verdict summary") or {}
    dossier_summary = (
        _ad_load_json(GCC_DOSSIER_SUMMARY_PATH, "Human escalation dossier summary") or {}
    )
    dossier_record = _ad_load_json(GCC_DOSSIER_JSON_PATH, "Human escalation dossier") or {}

    any_data = any(
        isinstance(d, dict) and d
        for d in (
            readiness,
            admission,
            eligibility,
            recommendation,
            review,
            verdict,
            dossier_summary,
        )
    )
    if not any_data:
        st.info(
            "No runtime governance artifacts found yet. Run the ARM runtime governance pipeline "
            "(Steps 53–60) to populate `data/results/arm_runtime_governance_*` outputs."
        )

    st.markdown(_GCC_UX_CSS, unsafe_allow_html=True)

    if any_data:
        gcc_stack = _gcc_build_governance_intelligence_stack(
            readiness=readiness,
            admission=admission,
            eligibility=eligibility,
            recommendation=recommendation,
            review=review,
            verdict=verdict,
            dossier_summary=dossier_summary,
            dossier_record=dossier_record,
        )
        _gcc_render_executive_summary_intelligence(
            stack=gcc_stack,
            dossier_summary=dossier_summary,
        )
        st.markdown("---")

    # ── SECTION 1 — RUNTIME GOVERNANCE STATUS ────────────────────────
    st.markdown("### Runtime Governance Status")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        _gcc_render_card(
            "Runtime Readiness",
            _gcc_get(readiness, "runtime_readiness_classification"),
            _gcc_get(readiness, "readiness_confidence"),
            _gcc_get(readiness, "readiness_state"),
        )
    with r1c2:
        _gcc_render_card(
            "Runtime Admission",
            _gcc_get(admission, "runtime_admission_classification"),
            _gcc_get(admission, "admission_confidence"),
            _gcc_get(admission, "admission_state"),
        )
    with r1c3:
        _gcc_render_card(
            "Constitutional Eligibility",
            _gcc_get(eligibility, "runtime_constitutional_eligibility_classification"),
            _gcc_get(eligibility, "constitutional_eligibility_confidence"),
            _gcc_get(eligibility, "constitutional_eligibility_state"),
        )
    with r1c4:
        _gcc_render_card(
            "Recommendation",
            _gcc_get(recommendation, "runtime_enablement_recommendation_classification"),
            _gcc_get(recommendation, "recommendation_confidence"),
            _gcc_get(recommendation, "recommendation_state"),
        )

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        _gcc_render_card(
            "Formal Review",
            _gcc_get(review, "runtime_enablement_review_classification"),
            _gcc_get(review, "review_confidence"),
            _gcc_get(review, "review_state"),
        )
    with r2c2:
        _gcc_render_card(
            "Institutional Verdict",
            _gcc_get(verdict, "runtime_verdict_classification"),
            _gcc_get(verdict, "verdict_confidence"),
            _gcc_get(verdict, "verdict_state"),
        )
    with r2c3:
        _gcc_render_card(
            "Human Escalation",
            _gcc_get(dossier_summary, "human_escalation_classification"),
            _gcc_get(dossier_summary, "escalation_confidence"),
            _gcc_get(dossier_summary, "dossier_state"),
        )

    # ── SECTION 2 — GOVERNANCE LADDER ────────────────────────────────
    st.markdown("### Governance Ladder")
    st.caption("Institutional progression across the runtime governance pipeline.")

    stage_sources: List[Tuple[str, Dict[str, Any]]] = [
        ("Readiness", readiness),
        ("Admission", admission),
        ("Eligibility", eligibility),
        ("Recommendation", recommendation),
        ("Review", review),
        ("Verdict", verdict),
        ("Human Escalation", dossier_summary),
    ]
    ladder_rows: List[Tuple[str, Any, Any, Any]] = []
    for idx, (label, src) in enumerate(stage_sources):
        state_key, cls_key, conf_key = _GCC_LADDER_STAGES[idx][1:]
        ladder_rows.append(
            (
                label,
                _gcc_get(src, cls_key),
                _gcc_get(src, conf_key),
                _gcc_get(src, state_key),
            )
        )
    _gcc_render_ladder(ladder_rows)

    # ── SECTION 3 — WHY RUNTIME IS BLOCKED ───────────────────────────
    st.markdown("### Why Runtime Is Blocked")
    blocked_groups = _gcc_collect_blocked_reason_groups(
        readiness,
        admission,
        eligibility,
        recommendation,
        review,
        verdict,
        dossier_summary,
        dossier_record,
    )
    st.warning("Runtime governance is blocked because:")
    for group_name, items in blocked_groups:
        st.markdown(f'<div class="gcc-block-group">{group_name}</div>', unsafe_allow_html=True)
        for reason in items:
            st.markdown(f'<div class="gcc-block-item">• {reason}</div>', unsafe_allow_html=True)

    # ── SECTION 4 — INSTITUTIONAL RUNTIME POSITION ───────────────────
    st.markdown("### Institutional Runtime Position")
    dossier = _gcc_get(dossier_record, "human_escalation_dossier") or {}
    position = (
        _gcc_get(dossier, "institutional_runtime_position")
        or _gcc_get(dossier_summary, "institutional_runtime_position")
        or _gcc_get(verdict, "institutional_runtime_position")
        or "—"
    )
    future_candidate = _gcc_get(
        dossier, "future_runtime_candidate", _gcc_get(dossier_summary, "future_runtime_candidate")
    )
    constitutional_safe = _gcc_get(dossier, "constitutional_safe")
    mutation_allowed = _gcc_get(dossier_summary, "runtime_mutation_allowed", False)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Institutional position", str(position))
    p2.metric("Future runtime candidate", "Yes" if future_candidate is True else "No")
    p3.metric("Constitutional safe", "Yes" if constitutional_safe is True else "No")
    p4.metric("Runtime mutation allowed", "Yes" if mutation_allowed is True else "No")

    st.error(
        "**Runtime is locked.** `runtime_mutation_allowed = false` — this dashboard does not "
        "enable, authorize, or mutate live runtime policy."
    )

    # ── SECTION 5 — EXECUTIVE BRIEFING ───────────────────────────────
    st.markdown("### Executive Briefing")
    executive_summary = _gcc_get(dossier, "executive_summary")
    if executive_summary:
        st.info(executive_summary)
    else:
        st.info(
            "Executive briefing unavailable. Run the human escalation dossier engine "
            "(`python -m services.arm_runtime_governance_human_escalation_dossier_engine`) "
            "to populate the institutional briefing."
        )

    recs = _gcc_get(dossier_record, "recommendations") or []
    if recs:
        with st.expander("Governance recommendations", expanded=False):
            for rec in recs:
                st.markdown(f"- {rec}")

    # ── SECTION 6 — GOVERNANCE CONFIDENCE TIMELINE ───────────────────
    st.markdown("### Governance Confidence Timeline")
    fig, max_conf = _gcc_build_confidence_timeline()
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        if max_conf is not None and max_conf <= 0.01:
            st.info(
                "Governance confidence is currently dormant.\n\n"
                "Timeline will become informative as governance maturity evolves."
            )
    else:
        st.info(
            "Governance confidence timeline unavailable. Memory CSVs will appear after the "
            "runtime governance engines accumulate observation cycles."
        )

    # ── SECTION 6B — GOVERNANCE HISTORICAL INTELLIGENCE ──────────────
    hist = _gcc_analyze_governance_history(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
    )
    _gcc_render_historical_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
    )

    # ── SECTION 6C — GOVERNANCE REGIME DETECTION ─────────────────────
    regime = _gcc_render_regime_detection(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
    )

    # ── SECTION 6D — GOVERNANCE FORECASTING & TRAJECTORY ─────────────
    forecast = _gcc_render_forecast_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
    )

    # ── SECTION 6E — GOVERNANCE CONTRADICTION & TENSION ──────────────
    tension = _gcc_render_tension_detection(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
    )

    # ── SECTION 6F — GOVERNANCE CONVERGENCE / CONSENSUS ──────────────
    consensus = _gcc_render_consensus_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
    )

    # ── SECTION 6G — GOVERNANCE CONFIDENCE INTEGRITY ─────────────────
    integrity = _gcc_render_confidence_integrity(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
    )

    # ── SECTION 6H — GOVERNANCE DECISION READINESS ───────────────────
    decision = _gcc_render_decision_readiness(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
    )

    # ── SECTION 6I — GOVERNANCE FAILURE MODES & INSTITUTIONAL RISK ───
    failure = _gcc_render_failure_mode_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
    )

    # ── SECTION 6J — GOVERNANCE AUDITABILITY & EVIDENCE INTEGRITY ──
    audit = _gcc_render_auditability_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
    )

    # ── SECTION 6K — GOVERNANCE COHERENCE & LOGIC INTEGRITY ────────
    coherence = _gcc_render_coherence_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
    )

    # ── SECTION 6L — GOVERNANCE STABILITY & DRIFT INTELLIGENCE ─────
    stability = _gcc_render_stability_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
    )

    # ── SECTION 6M — GOVERNANCE RESILIENCE & RECOVERY INTELLIGENCE ─
    resilience = _gcc_render_resilience_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
    )

    # ── SECTION 6N — GOVERNANCE LEARNING & IMPROVEMENT INTELLIGENCE ─
    improvement = _gcc_render_improvement_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
    )

    # ── SECTION 6O — GOVERNANCE FAILURE SCENARIO INTELLIGENCE ──────
    scenario = _gcc_render_failure_scenario_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
        improvement=improvement,
    )

    # ── SECTION 6P — GOVERNANCE INTERVENTION & CONTAINMENT ─────────
    intervention = _gcc_render_intervention_intelligence(
        readiness=readiness,
        admission=admission,
        eligibility=eligibility,
        recommendation=recommendation,
        review=review,
        verdict=verdict,
        dossier_summary=dossier_summary,
        dossier_record=dossier_record,
        hist=hist,
        regime=regime,
        forecast=forecast,
        tension=tension,
        consensus=consensus,
        integrity=integrity,
        decision=decision,
        failure=failure,
        audit=audit,
        coherence=coherence,
        stability=stability,
        resilience=resilience,
        improvement=improvement,
        scenario=scenario,
    )

    # ── SECTION 6Q — CONFIDENCE & EVIDENCE INTEGRITY INTELLIGENCE ──
    evidence = _gcc_render_institutional_evidence_intelligence(
        integrity=integrity,
        audit=audit,
        coherence=coherence,
        tension=tension,
        decision=decision,
        stability=stability,
        resilience=resilience,
        intervention=intervention,
    )

    # ── SECTION 6R — CONSENSUS & ALIGNMENT INTELLIGENCE ────────────
    alignment = _gcc_render_institutional_alignment_intelligence(
        consensus=consensus,
        coherence=coherence,
        tension=tension,
        integrity=integrity,
        decision=decision,
        stability=stability,
        resilience=resilience,
        evidence=evidence,
        intervention=intervention,
        hist=hist,
    )

    # ── SECTION 6S — DECISION QUALITY & REASONING INTEGRITY ────────
    _gcc_render_decision_quality_intelligence(
        coherence=coherence,
        tension=tension,
        integrity=integrity,
        decision=decision,
        stability=stability,
        resilience=resilience,
        evidence=evidence,
        alignment=alignment,
        intervention=intervention,
        improvement=improvement,
        hist=hist,
    )

    # ── SECTION 7 — HUMAN ESCALATION DOSSIER ─────────────────────────
    st.markdown("### Human Escalation Dossier")
    if not dossier:
        st.info("Human escalation dossier record not found yet.")
    else:
        posture = _gcc_get(dossier, "recommended_operator_posture", "—")
        st.success(f"**Recommended operator posture:** `{posture}`")

        with st.expander("Case for runtime governance", expanded=True):
            for item in _gcc_get(dossier, "case_for_runtime") or ["No supporting case recorded."]:
                st.markdown(f"- {item}")

        with st.expander("Case against runtime governance", expanded=True):
            for item in _gcc_get(dossier, "case_against_runtime") or ["No opposing case recorded."]:
                st.warning(item)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Key risks**")
            for item in _gcc_get(dossier, "key_risks") or ["—"]:
                st.markdown(f"- {item}")
        with c2:
            st.markdown("**Key safeguards**")
            for item in _gcc_get(dossier, "key_safeguards") or ["—"]:
                st.markdown(f"- {item}")

        chain = _gcc_get(dossier, "governance_chain_summary")
        if isinstance(chain, dict) and chain:
            with st.expander("Governance chain snapshot", expanded=False):
                chain_df = pd.DataFrame([{"stage": k, "state": v} for k, v in chain.items()])
                _ei_render_table(chain_df, height=220)

    gen_bits: List[str] = []
    for label, obj in (
        ("dossier", dossier_summary),
        ("verdict", verdict),
        ("readiness", readiness),
    ):
        ts = _gcc_get(obj, "generated_at_utc")
        if ts:
            gen_bits.append(f"{label}=`{ts}`")
    if gen_bits:
        st.caption(" • ".join(gen_bits))


# ──────────────────────────────
# GOVERNANCE LIBRARY CENTER (Step 138 — read-only)
# ──────────────────────────────

_GLC_CATEGORIES: Tuple[str, ...] = (
    "Constitution & Foundations",
    "Operations",
    "Committee & Decision Records",
    "Audit & Evidence",
    "Certification & Release",
    "Intelligence & Health",
    "Continuity & Resilience",
    "Ethics, Trust & Legitimacy",
    "Scalability & Evolution",
    "Glossary, Dependencies & Traceability",
    "Other",
)

_GLC_IMPORTANT_SPECS: Tuple[Tuple[int, str, str], ...] = (
    (
        100,
        "Governance Constitution",
        "Supreme principles, operating charter, and institutional rules.",
    ),
    (
        101,
        "Governance README / Navigation",
        "Canonical index and fast routing across the governance library.",
    ),
    (
        106,
        "Governance Committee Charter",
        "Committee authority, quorum decisions, and constitutional adjudication.",
    ),
    (
        113,
        "Governance Codex",
        "Unified system map, priority order, and framework interoperability.",
    ),
    (130, "Governance Operating System", "GOS capstone architecture integrating Steps 90–129."),
    (132, "Master Glossary", "Terminology authority and canonical definitions."),
    (133, "Dependency Matrix", "Framework dependencies, change impact, and relationship map."),
    (134, "Traceability Framework", "Evidence chains and audit/decision/change lineage."),
    (135, "Final Certification", "GOS release certification and recertification requirements."),
    (
        136,
        "Committee Operating Pack",
        "Committee session templates and decision/escalation registers.",
    ),
    (137, "Audit Pack", "Operational audit toolkit and evidence collection procedures."),
)

_GLC_WHERE_TO_GO: Tuple[Tuple[str, str], ...] = (
    (
        "Understand Triton's constitution",
        "Step 100 Constitution · Step 113 Codex · Step 130 GOS · Step 101 README",
    ),
    (
        "Run a committee meeting",
        "Step 106 Committee Charter · Step 136 Committee Operating Pack · Step 93 Roles & Authority",
    ),
    (
        "Find audit/evidence rules",
        "Step 107 Audit Readiness · Step 134 Traceability · Step 137 Audit Pack · Step 131 Library Audit",
    ),
    (
        "Check certification status",
        "Step 135 Final Certification · Step 110 Readiness Certification · Step 97 Training Certification",
    ),
    (
        "Look up a governance term",
        "Step 132 Master Glossary · Step 101 README index",
    ),
    (
        "Trace a decision back to authority",
        "Step 127 Delegation · Step 134 Traceability · Step 93 Roles & Authority · Step 121 Precedent",
    ),
    (
        "Review governance dependencies",
        "Step 133 Dependency Matrix · Step 130 GOS · Step 113 Codex",
    ),
    (
        "Understand governance command center outputs",
        "Step 91 Operator Playbook · Step 102 Operator Handbook · GCC (System → Governance Command Center)",
    ),
)


def _glc_parse_readme_index(readme_text: str) -> Dict[str, Tuple[int, str, str]]:
    """filename -> (step, link_label, purpose column)."""
    index: Dict[str, Tuple[int, str, str]] = {}
    row_re = re.compile(r"\|\s*\*\*(\d+)\*\*\s*\|\s*\[([^\]]+)\]\(\./([^)]+)\)\s*\|\s*([^|]+)\|")
    for step_s, label, fname, purpose in row_re.findall(readme_text):
        index[fname.strip()] = (int(step_s), label.strip(), purpose.strip())
    index["README.md"] = (
        101,
        "Governance README",
        "Canonical navigation index for the governance library.",
    )
    return index


def _glc_extract_title(head: str, filename: str) -> str:
    for line in head.splitlines()[:30]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    stem = Path(filename).stem.replace("_", " ")
    return stem


def _glc_extract_purpose_snippet(content: str) -> str:
    lines = content.splitlines()
    for i, line in enumerate(lines[:120]):
        low = line.strip().lower()
        if low in ("## purpose", "## purpose of the operator handbook") or low.startswith(
            "## purpose "
        ):
            parts: List[str] = []
            for j in range(i + 1, min(i + 12, len(lines))):
                nxt = lines[j].strip()
                if nxt.startswith("#") or nxt == "---":
                    break
                if nxt:
                    parts.append(nxt.lstrip("> ").strip())
            if parts:
                return " ".join(parts)[:400]
    for line in lines[1:25]:
        t = line.strip()
        if t and not t.startswith("#") and not t.startswith("|") and not t.startswith("**"):
            if len(t) > 40:
                return t[:400]
    return ""


def _glc_infer_category(step: Optional[int], filename: str, title: str) -> str:
    blob = f"{filename} {title}".lower()
    if step in (100, 101, 113, 128, 130) or any(
        k in blob
        for k in ("constitution", "codex", "operating_system", "meta_governance", "readme")
    ):
        return "Constitution & Foundations"
    if step in (106, 136, 127, 121, 120, 93) or any(
        k in blob
        for k in ("committee", "delegation", "precedent", "decision_quality", "roles_authority")
    ):
        return "Committee & Decision Records"
    if step in (107, 131, 137, 96) or any(
        k in blob
        for k in ("audit", "regulatory", "reporting_audit", "consolidation", "traceability")
    ):
        if step == 134:
            return "Glossary, Dependencies & Traceability"
        return "Audit & Evidence"
    if step in (97, 110, 135) or any(
        k in blob for k in ("training", "readiness_certification", "final_certification")
    ):
        return "Certification & Release"
    if step in (92, 99, 122) or any(
        k in blob for k in ("metrics_kpi", "observability", "health_intelligence")
    ):
        return "Intelligence & Health"
    if step in (108, 109, 111, 123, 119) or any(
        k in blob
        for k in ("crisis", "wargaming", "resilience", "succession", "postmortem", "survivability")
    ):
        return "Continuity & Resilience"
    if step in (116, 117, 129, 118) or any(
        k in blob for k in ("ethics", "trust", "legitimacy", "stakeholder", "capital_stewardship")
    ):
        return "Ethics, Trust & Legitimacy"
    if step in (114, 125, 126, 94) or any(
        k in blob for k in ("maturity_roadmap", "scalability", "complexity", "lifecycle_maturity")
    ):
        return "Scalability & Evolution"
    if step in (132, 133, 134) or any(
        k in blob for k in ("glossary", "dependency_matrix", "traceability_framework")
    ):
        return "Glossary, Dependencies & Traceability"
    if step in (90, 91, 98, 102, 103, 104, 115, 124) or any(
        k in blob
        for k in (
            "incident",
            "operator",
            "developer",
            "executive",
            "change_management",
            "foresight",
            "mission_alignment",
            "playbook",
            "handbook",
        )
    ):
        return "Operations"
    return "Other"


def _glc_rule_based_meta(
    *, category: str, title: str, step: Optional[int], summary: str
) -> Tuple[str, str, str]:
    purpose = summary or f"Governance manual supporting {category.lower()}."
    when = "Open when this topic applies to your role, audit, or operational situation."
    if step == 100:
        when = "Orientation, conflict resolution, constitutional attestation, and supreme-rule questions."
    elif step == 101:
        when = "Finding the right manual quickly; onboarding; library navigation."
    elif step == 91:
        when = "Every operator session after reading the GCC Operator Decision Brief."
    elif step in (106, 136):
        when = "Committee sessions, Hard Halt lift, constitutional votes, and formal governance records."
    elif step in (107, 137):
        when = "Audits, diligence, investigations, and evidence preservation."
    elif step == 132:
        when = "Terminology disputes, authoring, and audit language consistency."
    elif step == 134:
        when = "Tracing decisions, evidence chains, and audit lineage."
    elif step in (92, 122):
        when = "Weekly or quarterly governance health review and deterioration signals."
    related = "See Step 101 README index and Step 113 Codex for routing."
    if category == "Committee & Decision Records":
        related = "Steps 106, 136, 93, 127; GCC for live posture."
    elif category == "Audit & Evidence":
        related = "Steps 107, 131, 134, 137; Step 96 reporting packs."
    elif category == "Certification & Release":
        related = "Steps 135, 110, 97, 95 validation drills."
    return purpose, when, related


@st.cache_data(show_spinner=False)
def _glc_build_catalog(docs_dir_str: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    docs_dir = Path(docs_dir_str)
    if not docs_dir.is_dir():
        return [], f"Governance docs folder not found: {docs_dir}"

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        return [], "No markdown files found in docs/governance."

    readme_path = docs_dir / "README.md"
    index: Dict[str, Tuple[int, str, str]] = {}
    if readme_path.is_file():
        try:
            index = _glc_parse_readme_index(
                readme_path.read_text(encoding="utf-8", errors="replace")
            )
        except Exception:
            index = {"README.md": (101, "Governance README", "Navigation index.")}

    catalog: List[Dict[str, Any]] = []
    for path in md_files:
        rel = path.name
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            catalog.append(
                {
                    "rel_path": rel,
                    "filename": rel,
                    "step": index.get(rel, (None, "", ""))[0],
                    "title": rel,
                    "category": "Other",
                    "summary": f"Could not read file: {exc}",
                    "purpose": "",
                    "when_to_use": "",
                    "related": "",
                    "modified_utc": "",
                    "search_text": rel.lower(),
                    "read_error": str(exc),
                }
            )
            continue

        head = content[:8000]
        step: Optional[int] = None
        link_label = ""
        readme_purpose = ""
        if rel in index:
            step, link_label, readme_purpose = index[rel]
        else:
            m_end = re.search(r"\(Step\s+(\d{2,3})\s+completion\)", content[-800:], re.I)
            if m_end:
                step = int(m_end.group(1))
            m_head = re.search(r"Step\s+(\d{2,3})", head[:500], re.I)
            if m_head:
                step = int(m_head.group(1))

        title = _glc_extract_title(head, rel)
        if link_label and link_label not in title:
            short_title = link_label
        else:
            short_title = title

        snippet = _glc_extract_purpose_snippet(content)
        summary = readme_purpose or snippet or f"Governance manual: {short_title}."
        category = _glc_infer_category(step, rel, title)
        purpose, when_to_use, related = _glc_rule_based_meta(
            category=category, title=title, step=step, summary=summary
        )
        if snippet and not readme_purpose:
            purpose = snippet[:400]

        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        except Exception:
            mtime = ""

        search_text = " ".join(
            [
                rel,
                title,
                short_title,
                summary,
                category,
                str(step or ""),
                content[:16000],
            ]
        ).lower()

        catalog.append(
            {
                "rel_path": rel,
                "filename": rel,
                "step": step,
                "title": short_title,
                "category": category,
                "summary": summary,
                "purpose": purpose,
                "when_to_use": when_to_use,
                "related": related,
                "modified_utc": mtime,
                "search_text": search_text,
                "read_error": None,
            }
        )

    catalog.sort(
        key=lambda d: (
            d["step"] is None,
            d["step"] if d["step"] is not None else 9999,
            d["title"].lower(),
        )
    )
    return catalog, None


@st.cache_data(show_spinner=False)
def _glc_read_file(docs_dir_str: str, rel_path: str) -> Tuple[str, Optional[str]]:
    path = Path(docs_dir_str) / rel_path
    if not path.is_file() or path.suffix.lower() != ".md":
        return "", "Invalid document path."
    try:
        resolved = path.resolve()
        base = Path(docs_dir_str).resolve()
        if base not in resolved.parents and resolved != base:
            return "", "Path outside governance library."
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as exc:
        return "", str(exc)


def _glc_doc_by_step(catalog: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    for doc in catalog:
        if doc.get("step") == step:
            return doc
    return None


def _glc_filter_catalog(
    catalog: List[Dict[str, Any]],
    *,
    query: str,
    category: str,
    step_min: Optional[int],
    step_max: Optional[int],
) -> List[Dict[str, Any]]:
    q = query.strip().lower()
    out: List[Dict[str, Any]] = []
    for doc in catalog:
        if category != "All" and doc.get("category") != category:
            continue
        stp = doc.get("step")
        if step_min is not None and (stp is None or stp < step_min):
            continue
        if step_max is not None and (stp is None or stp > step_max):
            continue
        if q and q not in doc.get("search_text", ""):
            continue
        out.append(doc)
    return out


def page_governance_library_center() -> None:
    """Governance Library Center — read-only browse/search for docs/governance manuals."""
    st.title("📚 Governance Library Center")
    st.caption(
        "Central library for Triton governance manuals, operating packs, audit frameworks, "
        "and constitutional references."
    )
    st.info(
        "**Read-only documentation center.** This page does not change governance state, "
        "runtime policy, broker behavior, or execution logic."
    )

    st.markdown(
        """
        <style>
        .glc-banner{border-radius:10px;border:1px solid rgba(148,163,184,.35);
        padding:.5rem .75rem;margin:.25rem 0 1rem 0;background:rgba(15,23,42,.45);}
        .glc-imp{border-radius:10px;border:1px solid rgba(59,130,246,.35);
        padding:.65rem .75rem;margin:.35rem 0;background:rgba(15,23,42,.35);min-height:7.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    catalog, catalog_err = _glc_build_catalog(str(GOVERNANCE_DOCS_DIR.resolve()))
    if catalog_err:
        st.warning(catalog_err)
        return

    st.caption(f"**{len(catalog)}** governance documents indexed from `docs/governance/`.")

    # ── Important documents ──────────────────────────────────────────
    st.markdown("### Important documents")
    imp_cols = st.columns(2)
    for idx, (step, label, purpose) in enumerate(_GLC_IMPORTANT_SPECS):
        doc = _glc_doc_by_step(catalog, step)
        col = imp_cols[idx % 2]
        with col:
            with st.container(border=True):
                st.markdown(f"**Step {step} — {label}**")
                st.caption(purpose)
                if doc and not doc.get("read_error"):
                    if st.button("Open / Read", key=f"glc_imp_open_{step}"):
                        st.session_state["glc_selected_path"] = doc["rel_path"]
                        st.rerun()
                else:
                    st.caption("_Document not found in library scan._")

    st.markdown("---")

    # ── Where do I go? ───────────────────────────────────────────────
    with st.expander("What are you trying to do?", expanded=False):
        for question, pointers in _GLC_WHERE_TO_GO:
            st.markdown(f"**{question}**")
            st.caption(f"→ {pointers}")

    st.markdown("---")

    # ── Search & filters ─────────────────────────────────────────────
    st.markdown("### Browse & search")
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        search_q = st.text_input(
            "Search",
            placeholder="Title, step, keyword, filename, or content…",
            key="glc_search_q",
        )
    with f2:
        cat_filter = st.selectbox(
            "Category",
            ["All", *_GLC_CATEGORIES],
            key="glc_cat_filter",
        )
    with f3:
        use_step_min = st.checkbox("Min step", value=False, key="glc_use_min")
        step_min = st.number_input("Min", min_value=90, max_value=200, value=90, key="glc_step_min")
    with f4:
        use_step_max = st.checkbox("Max step", value=False, key="glc_use_max")
        step_max = st.number_input(
            "Max", min_value=90, max_value=200, value=137, key="glc_step_max"
        )

    filtered = _glc_filter_catalog(
        catalog,
        query=search_q,
        category=cat_filter,
        step_min=int(step_min) if use_step_min else None,
        step_max=int(step_max) if use_step_max else None,
    )
    st.caption(f"Showing **{len(filtered)}** of **{len(catalog)}** documents.")

    if not filtered:
        st.info("No documents match your filters. Clear search or widen the step range.")
    else:
        list_df = pd.DataFrame(
            [
                {
                    "Step": d["step"] if d["step"] is not None else "—",
                    "Title": d["title"],
                    "Category": d["category"],
                    "Summary": (d["summary"] or "")[:120]
                    + ("…" if len(d["summary"] or "") > 120 else ""),
                    "File": d["filename"],
                    "Modified": d.get("modified_utc") or "—",
                }
                for d in filtered
            ]
        )
        st.dataframe(list_df, use_container_width=True, hide_index=True)

        pick_labels = [
            f"Step {d['step']} — {d['title']}" if d["step"] is not None else d["title"]
            for d in filtered
        ]
        pick_map = {lbl: d["rel_path"] for lbl, d in zip(pick_labels, filtered)}
        pick_choice = st.selectbox(
            "Select document to read",
            ["—"] + pick_labels,
            key="glc_pick_doc",
        )
        if st.button("Read selected document", key="glc_read_selected"):
            if pick_choice and pick_choice != "—":
                st.session_state["glc_selected_path"] = pick_map[pick_choice]
                st.rerun()

    st.markdown("---")

    # ── Document reader ────────────────────────────────────────────────
    selected_rel = st.session_state.get("glc_selected_path")
    if not selected_rel:
        return

    selected = next((d for d in catalog if d["rel_path"] == selected_rel), None)
    if selected is None:
        st.warning(f"Selected document not in catalog: `{selected_rel}`")
        return

    st.markdown("### Document reader")
    if st.button("← Back to library list", key="glc_clear_reader"):
        st.session_state.pop("glc_selected_path", None)
        st.rerun()

    if selected.get("read_error"):
        st.error(f"Cannot read `{selected_rel}`: {selected['read_error']}")
        return

    content, read_err = _glc_read_file(str(GOVERNANCE_DOCS_DIR.resolve()), selected_rel)
    if read_err:
        st.error(f"Could not load `{selected_rel}`: {read_err}")
        return

    step_disp = selected["step"] if selected["step"] is not None else "—"
    st.markdown(f"## {selected['title']}")
    st.caption(
        f"**File:** `docs/governance/{selected_rel}` · **Category:** {selected['category']} · "
        f"**Step:** {step_disp} · **Modified:** {selected.get('modified_utc') or '—'}"
    )

    st.markdown("#### What this document is for")
    st.write(selected.get("purpose") or selected.get("summary") or "—")
    st.markdown("#### When to use it")
    st.write(selected.get("when_to_use") or "—")
    st.markdown("#### Related documents")
    st.write(selected.get("related") or "—")

    st.markdown("---")
    with st.expander("Raw markdown source", expanded=False):
        st.code(content, language="markdown")

    st.markdown("---")
    st.markdown(content)


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
    ("Risk & Guardrails", "🛡️ Risk Office"): page_risk_office,
    ("Risk & Guardrails", "🧪 Defensive Simulation Lab"): page_defensive_simulation_lab,
    ("Risk & Guardrails", "🏛 Executive Risk Command Center"): page_executive_risk_command_center,
    ("Risk & Guardrails", "🧱 Defensive Automation Sandbox"): page_defensive_automation_sandbox,
    ("Risk & Guardrails", "👤 Human Approval Center"): page_human_approval_center,
    (
        "Risk & Guardrails",
        "🛡️ Protective Action Policy Center",
    ): page_protective_action_policy_center,
    (
        "Risk & Guardrails",
        "🏛 Governance Authorization Center",
    ): page_governance_authorization_center,
    ("Risk & Guardrails", "⚙️ Execution Readiness Center"): page_execution_readiness_center,
    ("Risk & Guardrails", "🧪 Protective Action Trials"): page_protective_action_trials,
    ("Risk & Guardrails", "📊 Protective Action Evaluation"): page_protective_action_evaluation,
    ("Risk & Guardrails", "🧠 Adaptive Capital Preservation"): page_adaptive_capital_preservation,
    ("Risk & Guardrails", "👑 Capital Preservation Governor"): page_capital_preservation_governor,
    (
        "Risk & Guardrails",
        "📋 Capital Preservation Audit Center",
    ): page_capital_preservation_audit_center,
    ("Risk & Guardrails", "🧪 Preservation Stress Lab"): page_preservation_stress_lab,
    (
        "Risk & Guardrails",
        "🏅 Preservation Certification Center",
    ): page_preservation_certification_center,
    ("Risk & Guardrails", "🏛 Risk Committee Oversight"): page_risk_committee_oversight,
    ("Risk & Guardrails", "📑 Accountability Registry"): page_accountability_registry,
    ("Risk & Guardrails", "👑 Preservation Governance Board"): page_preservation_governance_board,
    ("Risk & Guardrails", "🏛 Investment Committee Review"): page_investment_committee_review,
    ("Risk & Guardrails", "📈 Triton Maturity Assessment"): page_triton_maturity_assessment,
    ("Risk & Guardrails", "🎯 Strategic Oversight Center"): page_strategic_oversight_center,
    ("Risk & Guardrails", "🧩 Decision Quality Center"): page_decision_quality_center,
    ("Risk & Guardrails", "🏛 Institutional Intelligence"): page_institutional_intelligence,
    ("Risk & Guardrails", "🚀 Strategic Self-Improvement"): page_strategic_self_improvement,
    ("Risk & Guardrails", "🧠 Institutional Memory"): page_institutional_memory,
    ("Risk & Guardrails", "🕸 Institutional Knowledge Graph"): page_institutional_knowledge_graph,
    ("Risk & Guardrails", "📚 Organizational Learning Center"): page_organizational_learning_center,
    ("Risk & Guardrails", "🔍 Causal Reasoning Center"): page_causal_reasoning_center,
    ("Risk & Guardrails", "📖 Explainability Center"): page_explainability_center,
    ("Risk & Guardrails", "💡 Institutional Insights"): page_institutional_insights_center,
    ("Risk & Guardrails", "♟ Strategic Reasoning Center"): page_strategic_reasoning_center,
    ("Risk & Guardrails", "🔮 Consequence Forecast Center"): page_consequence_forecast_center,
    ("Risk & Guardrails", "📜 Institutional Wisdom Center"): page_institutional_wisdom_center,
    ("Risk & Guardrails", "🗺 Scenario Planning Center"): page_scenario_planning_center,
    ("Risk & Guardrails", "🛣 Future Path Analysis"): page_future_path_analysis_center,
    ("Risk & Guardrails", "🎯 Strategic Priorities Center"): page_strategic_priorities_center,
    ("Risk & Guardrails", "🎯 Attention Allocation Center"): page_attention_allocation_center,
    ("Risk & Guardrails", "🔗 System Coordination Center"): page_system_coordination_center,
    (
        "Risk & Guardrails",
        "⚡ Institutional Optimization Center",
    ): page_institutional_optimization_center,
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
    ("System", "🏛 Governance Command Center"): page_governance_command_center,
    ("System", "📚 Governance Library Center"): page_governance_library_center,
    ("System", "🗂 Governance Evidence Registry"): page_governance_evidence_registry,
    ("System", "🧾 Governance Audit Center"): page_governance_audit_center,
    ("System", "⚖️ Governance Decision Registry"): page_governance_decision_registry,
    ("System", "🚨 Governance Escalation Registry"): page_governance_escalation_registry,
    ("System", "🔗 Governance Traceability Explorer"): page_governance_traceability_explorer,
    ("System", "🕵️ Governance Investigation Center"): page_governance_investigation_center,
    ("System", "🧠 Governance Intelligence Lab"): page_governance_intelligence_lab,
    ("System", "🕰 Governance Timeline Center"): page_governance_timeline_center,
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
    "Risk & Guardrails": [
        "Risk Report",
        "🛡️ Risk Office",
        "🧪 Defensive Simulation Lab",
        "🏛 Executive Risk Command Center",
        "🧱 Defensive Automation Sandbox",
        "👤 Human Approval Center",
        "🛡️ Protective Action Policy Center",
        "🏛 Governance Authorization Center",
        "⚙️ Execution Readiness Center",
        "🧪 Protective Action Trials",
        "📊 Protective Action Evaluation",
        "🧠 Adaptive Capital Preservation",
        "👑 Capital Preservation Governor",
        "📋 Capital Preservation Audit Center",
        "🧪 Preservation Stress Lab",
        "🏅 Preservation Certification Center",
        "🏛 Risk Committee Oversight",
        "📑 Accountability Registry",
        "👑 Preservation Governance Board",
        "🏛 Investment Committee Review",
        "📈 Triton Maturity Assessment",
        "🎯 Strategic Oversight Center",
        "🧩 Decision Quality Center",
        "🏛 Institutional Intelligence",
        "🚀 Strategic Self-Improvement",
        "🧠 Institutional Memory",
        "🕸 Institutional Knowledge Graph",
        "📚 Organizational Learning Center",
        "🔍 Causal Reasoning Center",
        "📖 Explainability Center",
        "💡 Institutional Insights",
        "♟ Strategic Reasoning Center",
        "🔮 Consequence Forecast Center",
        "📜 Institutional Wisdom Center",
        "🗺 Scenario Planning Center",
        "🛣 Future Path Analysis",
        "🎯 Strategic Priorities Center",
        "🎯 Attention Allocation Center",
        "🔗 System Coordination Center",
        "⚡ Institutional Optimization Center",
        "Strategy Diagnostics",
    ],
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
        "🏛 Governance Command Center",
        "📚 Governance Library Center",
        "🗂 Governance Evidence Registry",
        "🧾 Governance Audit Center",
        "⚖️ Governance Decision Registry",
        "🚨 Governance Escalation Registry",
        "🔗 Governance Traceability Explorer",
        "🕵️ Governance Investigation Center",
        "🧠 Governance Intelligence Lab",
        "🕰 Governance Timeline Center",
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
