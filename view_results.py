# view_results.py — TRITON Command Center (Phase 1 Unified Dashboard)
# Includes:
#   - project-root detection
#   - CSV/parquet loaders
#   - portfolio / guard / market-status helpers
#   - global header w/ regime + drawdown + BP + timestamp
#   - sidebar navigation (Sections -> Pages)
#   - per-page render stubs (to be filled with real tab logic)

import os
import re
import json
import math
import time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import datetime as _dt  # for isinstance checks vs datetime

# Optional plotting libs (we'll hook them up in tab bodies later)
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_OK = True
except Exception:
    px = go = None
    PLOTLY_OK = False

try:
    import matplotlib.pyplot as plt

    MPL_OK = True
except Exception:
    plt = None
    MPL_OK = False

# --- Triton Alpaca env bootstrap (no python-dotenv needed) ---
import os, re


def _load_dotenv_if_present(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            # keep existing process env precedence
            if k not in os.environ:
                os.environ[k] = v


def _wire_alpaca_env():
    # precedence: existing process env; otherwise pull from .env we just loaded
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    base = os.environ.get("APCA_API_BASE_URL")
    # respect ALPACA_ENV if base missing
    if not base:
        env = (os.environ.get("ALPACA_ENV") or "paper").strip().lower()
        if env == "paper":
            base = "https://paper-api.alpaca.markets"
        else:
            base = "https://api.alpaca.markets"
    # write back canonical APCA_* so downstream code can rely on them
    os.environ.setdefault("APCA_API_KEY_ID", key or "")
    os.environ.setdefault("APCA_API_SECRET_KEY", sec or "")
    os.environ["APCA_API_BASE_URL"] = base  # always set base


def _probe_buying_power():
    key = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    base = (os.environ.get("APCA_API_BASE_URL") or "").rstrip("/")
    if not (key and sec and base):
        return 0.0
    # try alpaca-py first
    try:
        from alpaca.trading.client import TradingClient

        client = TradingClient(key, sec, paper=("paper-api" in base))
        acct = client.get_account()
        return float(getattr(acct, "buying_power", 0) or 0)
    except Exception:
        pass
    # fallback: raw HTTP
    try:
        import requests

        r = requests.get(
            f"{base}/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
            timeout=10,
        )
        r.raise_for_status()
        return float((r.json() or {}).get("buying_power") or 0)
    except Exception:
        return 0.0


# --- DataFrame column sanitizer ---
def sanitize_df_cols(df):
    """Make column names flat, trimmed, and unique (avoids Arrow duplicate-name error)."""
    import pandas as pd

    # 1) Flatten MultiIndex columns (if any)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in tup if x is not None and str(x) != "") for tup in df.columns
        ]

    # 2) Trim + dedupe
    seen = {}
    new_cols = []
    for c in [str(c).strip() for c in df.columns]:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")  # e.g., action -> action_1, action_2, ...
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df


# initialize
_load_dotenv_if_present(".env")
_wire_alpaca_env()

# ──────────────────────────────
# BRAND / THEME CONSTANTS
# ──────────────────────────────
APP_VERSION = "r25-29_2025-09-27a"

BRAND_BG = "#0f172a"  # deep navy / slate
CARD_BG = "#1e293b"  # slightly lighter card bg
TEXT_COL = "#f8fafc"  # near-white
ACCENT = "#38bdf8"  # cyan accent


# ──────────────────────────────
# PAGE CONFIG (set once)
# ──────────────────────────────
st.set_page_config(
    page_title="TRITON • Command Center",
    page_icon="🧠",
    layout="wide",
)


# ──────────────────────────────
# GLOBAL CSS INJECTION (dark Triton skin)
# ──────────────────────────────
st.markdown(
    f"""
<style>
/* page background + default text */
body {{
    background-color: {BRAND_BG};
    color: {TEXT_COL};
}}
section.main > div {{
    padding-top: 0rem;
}}

.triton-card {{
    background-color: {CARD_BG};
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.2);
    padding: 1rem 1.25rem;
    color: {TEXT_COL};
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Inter","Roboto","Segoe UI",sans-serif;
    margin-bottom: 1rem;
}}
.triton-card h3 {{
    margin-top: 0;
    color: {TEXT_COL};
    font-size: 1rem;
    font-weight: 600;
}}
.data-label {{
    font-size: .75rem;
    color: #94a3b8;
}}
.data-value {{
    font-size: 1rem;
    font-weight: 600;
    color: {TEXT_COL};
}}

/* market pill base */
.market-pill {{
    border-radius:8px;
    padding:0.6rem 0.8rem;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
    font-size:0.8rem;
    line-height:1.3;
    min-width:220px;
    max-width:360px;
    box-shadow:0 1px 2px rgba(0,0,0,0.2);
}}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────
# PATHS / PROJECT ROOT SETUP
# ──────────────────────────────
def _safe_this_file() -> Path:
    """
    Streamlit runs via `streamlit run view_results.py`, which can mess with __file__.
    Graceful fallback to CWD if needed.
    """
    try:
        return Path(__file__).resolve()
    except Exception:
        return Path.cwd() / "view_results.py"


THIS_FILE = _safe_this_file()

ENV_ROOT = os.environ.get("TRITON_PROJECT_ROOT", "").strip()
DEFAULT_PROJECT_ROOT = Path(ENV_ROOT).expanduser().resolve() if ENV_ROOT else THIS_FILE.parent

# (these will be overridden if user sets Manual path in sidebar)
PROJECT_ROOT: Path = DEFAULT_PROJECT_ROOT
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_ROOT / "results"
ORDERS_DIR = DATA_ROOT / "orders"
PRED_DIR = DATA_ROOT / "predictions"
STRESS_DIR = DATA_ROOT / "stress_test_results"

for p in (RESULTS_DIR, ORDERS_DIR, PRED_DIR, STRESS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# key artifact paths
PORTFOLIO_HISTORY_PATH = RESULTS_DIR / "portfolio_history.csv"
LIVE_ORDERS_LOG_PATH = RESULTS_DIR / "live_orders.csv"
GUARD_SNAPSHOT_PATH = RESULTS_DIR / "guard_snapshot.json"


# ──────────────────────────────
# BASIC LOADERS
# ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(filename: str, folder: Path = RESULTS_DIR) -> pd.DataFrame:
    path = folder / filename
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Could not load {path}: {e}")
        return pd.DataFrame()


def load_csv_from(folder: Path, filename: str) -> pd.DataFrame:
    return load_csv(filename, folder)


@st.cache_data(show_spinner=False)
def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"⚠️ Could not read {path.name}: {e}")
        return pd.DataFrame()


def parse_dates_inplace(df: pd.DataFrame, cols=("date",), normalize: bool = False) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.tz_localize(None)
            if normalize:
                df[c] = df[c].dt.normalize()
    return df


def ensure_date(
    df: pd.DataFrame,
    candidates: Optional[List[str]] = None,
    normalize: bool = False,
) -> pd.DataFrame:
    if candidates is None:
        candidates = [
            "date",
            "as_of",
            "timestamp",
            "time",
            "datetime",
            "Date",
            "created_at",
            "updated_at",
        ]
    chosen = next((c for c in df.columns if c in candidates), None)
    if chosen is not None:
        s = pd.to_datetime(df[chosen], errors="coerce", utc=True)
        df["date"] = s.dt.tz_localize(None)
        if normalize:
            df["date"] = df["date"].dt.normalize()
    else:
        df["date"] = pd.NaT
    return df


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_score_col(df: pd.DataFrame) -> Optional[str]:
    if "total_score" in df.columns:
        return "total_score"
    if "score" in df.columns:
        return "score"
    return None


# ──────────────────────────────
# MODEL METRICS (used later in Model Comparison tab)
# ──────────────────────────────
def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ──────────────────────────────
# PORTFOLIO / PERFORMANCE HELPERS
# ──────────────────────────────
def normalize_to_one(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    first = s.dropna().iloc[0]
    if not np.isfinite(first) or first == 0:
        return s
    return s / first


def perf_stats_from_levels(levels: pd.Series, freq_per_year: int = 252) -> dict:
    s = pd.to_numeric(levels, errors="coerce").dropna()
    if s.size < 3:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
        }

    rets = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = float(s.iloc[-1] / s.iloc[0] - 1)

    if hasattr(s.index, "dtype"):
        days = max((s.index[-1] - s.index[0]).days, 1)
        years = days / 365.25
    else:
        years = len(s) / freq_per_year

    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(rets.std() * np.sqrt(freq_per_year)) if rets.size else np.nan
    sharpe = (
        float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(freq_per_year)) if rets.size else np.nan
    )

    peak = s.cummax()
    dd = (s / peak - 1).min()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": float(dd),
    }


# ──────────────────────────────
# SENTIMENT / NEWS HELPERS
# ──────────────────────────────
def strip_html(s):
    if pd.isna(s):
        return s
    return re.sub(r"<[^>]*>", "", str(s))


def extract_href(s):
    if pd.isna(s):
        return None
    m = re.search(r'href="([^"]+)"', str(s))
    return m.group(1) if m else None


def make_clickable(title, url):
    if pd.isna(url) or not str(url).strip():
        return str(title) if not pd.isna(title) else ""
    if str(url).strip().startswith("<a "):
        return str(url)
    safe_title = str(title) if (not pd.isna(title) and str(title).strip()) else "Link"
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{safe_title}</a>'


def sum_safe(s, default=0.0):
    try:
        s = pd.to_numeric(pd.Series(s), errors="coerce")
        return float(s.fillna(0).sum())
    except Exception:
        return float(default)


@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def derive_total_value(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "total_value" in out.columns:
        out["total_value"] = pd.to_numeric(out["total_value"], errors="coerce")
        return out
    if "portfolio_value" in out.columns:
        out["total_value"] = pd.to_numeric(out["portfolio_value"], errors="coerce")
        return out
    if {"cash", "market_value"}.issubset(out.columns):
        out["total_value"] = pd.to_numeric(out["cash"], errors="coerce").fillna(
            0.0
        ) + pd.to_numeric(out["market_value"], errors="coerce").fillna(0.0)
    return out


def backfill_close_from_parquet(sig_df: pd.DataFrame) -> pd.DataFrame:
    out = sig_df.copy()
    if "close" not in out.columns:
        out["close"] = np.nan

    need_close = ~out["close"].notna()
    if not need_close.any():
        return out

    tickers = out.loc[need_close, "ticker"].dropna().astype(str).unique().tolist()

    rows = []
    for t in tickers:
        pq = RESULTS_DIR / f"{t}.parquet"
        ohlc = load_parquet(pq)
        last_close = np.nan
        if not ohlc.empty and "close" in ohlc.columns:
            parse_dates_inplace(ohlc, ("date",))
            ohlc = ohlc.dropna(subset=["date"]).sort_values("date")
            if not ohlc.empty:
                last_close = float(ohlc["close"].iloc[-1])
        rows.append((t, last_close))

    if rows:
        fill = pd.DataFrame(rows, columns=["ticker", "_last_close"])
        out = out.merge(fill, on="ticker", how="left")
        out["close"] = pd.to_numeric(out["close"], errors="coerce").fillna(out["_last_close"])
        out.drop(columns=["_last_close"], inplace=True, errors="ignore")

    return out


def stabilize_weights(raw_wide: pd.DataFrame, cap: float) -> pd.DataFrame:
    """
    Row-wise cap & normalize portfolio weights.
    """
    if raw_wide is None or raw_wide.empty:
        return pd.DataFrame(
            index=getattr(raw_wide, "index", None),
            columns=getattr(raw_wide, "columns", None),
            dtype=float,
        )

    avail = raw_wide.notna()
    W = raw_wide.copy().astype(float).fillna(0.0).clip(lower=0.0)

    if cap is not None and np.isfinite(cap) and cap > 0:
        W = W.clip(upper=float(cap))

    row_sum = W.sum(axis=1)
    has_mass = row_sum > 0

    W_norm = pd.DataFrame(0.0, index=W.index, columns=W.columns)

    if has_mass.any():
        W_norm.loc[has_mass] = W.loc[has_mass].div(row_sum.loc[has_mass], axis=0)

    no_mass = ~has_mass
    if no_mass.any():
        counts = avail.loc[no_mass].sum(axis=1)
        valid = counts > 0
        if valid.any():
            eq_idx = counts[valid].index
            W_eq = avail.loc[eq_idx].div(counts.loc[eq_idx], axis=0).astype(float)
            W_norm.loc[eq_idx] = W_eq

    return W_norm.fillna(0.0)


def _ensure_columns(df: pd.DataFrame, needed: List[str], defaults: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for col in needed:
        if col not in df.columns:
            df[col] = defaults.get(col, "")
    return df


def _safe_pick(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.info(
            f"Some optional columns are missing and were skipped: {', '.join(missing)}",
            icon="ℹ️",
        )
    return df[present]


# ──────────────────────────────
# CAPITAL GUARD / STATUS SNAPSHOT HELPERS
# ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_guard_snapshot(path: Path = GUARD_SNAPSHOT_PATH) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        for k, v in data.items():
            out[k] = v
        return out
    except Exception as e:
        st.warning(f"⚠️ Could not read {path.name}: {e}")
        return {}


@st.cache_data(show_spinner=False)
def load_portfolio_history(path: Path = PORTFOLIO_HISTORY_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "equity"])

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Could not read {path}: {e}")
        return pd.DataFrame(columns=["timestamp", "equity"])

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = ts.dt.tz_localize(None)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    df["equity"] = pd.to_numeric(df.get("equity", np.nan), errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_open_orders(log_path: Path = LIVE_ORDERS_LOG_PATH) -> pd.DataFrame:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        st.warning(f"⚠️ Could not read {log_path.name}: {e}")
        return pd.DataFrame()

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = ts.dt.tz_localize(None)

    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    if "limit_price" in df.columns:
        df["limit_price"] = pd.to_numeric(df["limit_price"], errors="coerce")

    for c in ("side", "status"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    return df


def latest_portfolio_status() -> Dict[str, Any]:
    out = {
        "mode": "UNKNOWN",
        "reason": "",
        "drawdown_pct": np.nan,
        "latest_equity": np.nan,
        "buying_power": np.nan,
        "reserve_pct": np.nan,
        "timestamp": "",
    }

    # equity + drawdown
    ph = load_portfolio_history()
    if not ph.empty:
        latest_equity = float(ph["equity"].iloc[-1])
        peak_equity = float(ph["equity"].max())
        out["latest_equity"] = latest_equity
        if np.isfinite(latest_equity) and np.isfinite(peak_equity) and peak_equity > 0:
            dd = (latest_equity / peak_equity) - 1.0
            out["drawdown_pct"] = float(dd)

    # guard_snapshot.json overrides
    guard = load_guard_snapshot()
    if guard:
        if "mode" in guard:
            out["mode"] = str(guard.get("mode", out["mode"])).upper() or out["mode"]
        if "reason" in guard:
            out["reason"] = guard.get("reason", out["reason"])
        if "buying_power" in guard and np.isfinite(guard["buying_power"]):
            out["buying_power"] = float(guard["buying_power"])
        if "reserve_pct" in guard and np.isfinite(guard["reserve_pct"]):
            out["reserve_pct"] = float(guard["reserve_pct"])
        if "latest_equity" in guard and np.isfinite(guard["latest_equity"]):
            out["latest_equity"] = float(guard["latest_equity"])
        if "drawdown_pct" in guard and np.isfinite(guard["drawdown_pct"]):
            out["drawdown_pct"] = float(guard["drawdown_pct"])
        if "timestamp" in guard:
            out["timestamp"] = str(guard["timestamp"])

    # fallback mode using last open_orders row if still UNKNOWN
    if out["mode"] == "UNKNOWN":
        audit = load_open_orders()
        if not audit.empty:
            last_row = audit.iloc[0].to_dict()
            note_txt = str(last_row.get("note", "")).upper()
            status_txt = str(last_row.get("status", "")).upper()

            if "LOCKDOWN" in note_txt:
                out["mode"] = "LOCKDOWN"
                out["reason"] = "Capital Preservation / LOCKDOWN referenced in last order note."
            elif "DEFENSIVE" in note_txt:
                out["mode"] = "DEFENSIVE"
                out["reason"] = "Scaled down due to DEFENSIVE posture."
            elif "OK" in status_txt:
                out["mode"] = "NORMAL"
                if not out["reason"]:
                    out["reason"] = "Orders accepted; looks normal."

    return out


# ──────────────────────────────
# STATUS ACCESSORS (used by header chips)
# ──────────────────────────────
def get_current_regime() -> str:
    snap = latest_portfolio_status()
    mode = snap.get("mode", "UNKNOWN").upper()
    if mode == "LOCKDOWN":
        return "LOCKDOWN"
    if mode == "DEFENSIVE":
        return "DEFENSIVE"
    if mode == "NORMAL":
        return "NORMAL"
    return "UNKNOWN"


def get_drawdown_pct() -> float:
    snap = latest_portfolio_status()
    dd = snap.get("drawdown_pct", np.nan)
    if dd is None or not np.isfinite(dd):
        return float("nan")
    return float(dd * 100.0)  # -0.033 → -3.3


def get_buying_power() -> float:
    snap = latest_portfolio_status()
    bp = snap.get("buying_power", np.nan)
    if bp is None or not np.isfinite(bp):
        return float("nan")
    return float(bp)


def get_guard_timestamp() -> str:
    snap = latest_portfolio_status()
    ts = snap.get("timestamp", "")
    if ts:
        return ts
    # fallback: current ET
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    offset = timedelta(hours=-4)
    now_et = now_utc.astimezone(timezone(offset))
    return now_et.strftime("%Y-%m-%d %H:%M ET")


# ──────────────────────────────
# MARKET STATUS HELPERS (for the red/green pill)
# ──────────────────────────────
def _humanize_any_timedelta(td_obj) -> str:
    if td_obj is None:
        return "--"
    try:
        total_seconds = td_obj.total_seconds()
    except Exception:
        return "--"

    secs = int(max(total_seconds, 0))
    days = secs // 86400
    secs -= days * 86400
    hours = secs // 3600
    secs -= hours * 3600
    minutes = secs // 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def get_market_status() -> Dict[str, Any]:
    """
    Approximate US equities session (ET ~ UTC-4), Mon–Fri 09:30–16:00.
    """
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    offset = timedelta(hours=-4)
    now_et = now_utc.astimezone(timezone(offset))

    weekday = now_et.weekday()  # 0=Mon ... 6=Sun
    open_today = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_today = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    is_trading_day = weekday < 5

    if is_trading_day and (now_et >= open_today) and (now_et < close_today):
        is_open = True
        next_label = "Next close:"
        next_ts = close_today
    else:
        is_open = False
        if is_trading_day and now_et < open_today:
            next_open = open_today
        else:
            tmp = now_et + timedelta(days=1)
            while tmp.weekday() >= 5:
                tmp += timedelta(days=1)
            next_open = tmp.replace(hour=9, minute=30, second=0, microsecond=0)
        next_label = "Next open:"
        next_ts = next_open

    delta = next_ts - now_et

    return {
        "is_open": is_open,
        "now_et": now_et,
        "next_label": next_label,
        "next_ts": next_ts,
        "delta": delta,
    }


def render_market_chip_html() -> str:
    try:
        status = get_market_status()
    except Exception:
        status = {}

    is_open = bool(status.get("is_open", False))
    now_et = status.get("now_et", None)
    next_label = status.get("next_label", "Next:")
    next_ts = status.get("next_ts", None)
    delta = status.get("delta", None)

    icon = "🟢" if is_open else "🔴"
    state_txt = "OPEN" if is_open else "CLOSED"
    countdown_txt = _humanize_any_timedelta(delta)

    if isinstance(next_ts, (_dt.datetime, pd.Timestamp)):
        nxt_str = pd.Timestamp(next_ts).strftime("%Y-%m-%d %H:%M ET")
    else:
        nxt_str = "--"

    if hasattr(now_et, "strftime"):
        last_refresh_str = now_et.strftime("%Y-%m-%d %H:%M ET")
    else:
        last_refresh_str = "--"

    bg = "#e6ffed" if is_open else "#ffecec"
    border = "#2ecc71" if is_open else "#e74c3c"
    text_color = "#0a3622" if is_open else "#511010"

    pill_html = f"""
    <div class="market-pill" style="
        display:flex;
        flex-wrap:wrap;
        align-items:flex-start;
        gap:0.5rem;
        border:1px solid {border};
        background:{bg};
        color:{text_color};
    ">
        <div style="font-weight:600; font-size:0.8rem; min-width:5rem;">
            <span style="font-size:0.8rem; margin-right:0.4rem;">{icon}</span>
            <span style="letter-spacing:0.02em;">Market {state_txt}</span>
        </div>
        <div style="display:flex; flex-direction:column; font-size:0.75rem; line-height:1.2; opacity:0.9;">
            <div>
                {("closes" if is_open else "opens")} in
                <strong>{countdown_txt}</strong>
            </div>
            <div style="opacity:0.8;">
                {next_label} {nxt_str}
            </div>
            <div style="opacity:0.6; margin-top:0.4rem;">
                Last update: {last_refresh_str}
            </div>
        </div>
    </div>
    """
    return pill_html


# ──────────────────────────────
# GLOBAL HEADER RENDER (dark Triton top block, now via components.html)
# ──────────────────────────────
def render_global_header():
    dd_pct_raw = get_drawdown_pct()  # may be NaN
    bp_raw = get_buying_power()
    regime_raw = get_current_regime()
    ts_raw = get_guard_timestamp()
    market_chip_html = render_market_chip_html()

    def _fmt_pct_local(val):
        if val is None or not np.isfinite(val):
            return "--"
        return f"{val:.2f}%"

    def _fmt_dollar_local(val):
        if val is None or not np.isfinite(val):
            return "--"
        return f"${val:,.2f}"

    def _fmt_text_local(s):
        s = (s or "").strip()
        return s if s else "--"

    dd_pct = _fmt_pct_local(dd_pct_raw)
    bp = _fmt_dollar_local(bp_raw)
    regime = _fmt_text_local(regime_raw)
    ts = _fmt_text_local(ts_raw)

    # NOTE: no HTML comments, all inline styles,
    # and we render with components.html so Streamlit
    # doesn't try to "help" or escape it.
    header_html = f"""
    <div style="
        background: radial-gradient(circle at 10% 10%, rgba(56,189,248,0.18) 0%, rgba(15,23,42,0) 60%);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        color: {TEXT_COL};
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Inter', 'Roboto', 'Segoe UI', sans-serif;
        max-width: 1100px;
    ">

      <div style="
          display:flex;
          flex-wrap:wrap;
          align-items:flex-start;
          justify-content:space-between;
          gap:0.75rem;
      ">

        <div style="display:flex;flex-direction:column;">
          <div style="
              font-size:0.8rem;
              font-weight:500;
              color:{ACCENT};
              letter-spacing:.05em;
              text-transform:uppercase;
          ">
            TRITON • COMMAND CENTER
          </div>

          <div style="
              font-size:1.25rem;
              font-weight:600;
              color:{TEXT_COL};
              line-height:1.4;
          ">
            Blue Atlantic Asset Intelligence
          </div>

          <div style="
              font-size:0.8rem;
              color:#94a3b8;
              line-height:1.3;
              white-space:nowrap;
          ">
            Capital Preservation First • Adaptive AI Execution • {APP_VERSION}
          </div>
        </div>

        <div style="
            display:flex;
            flex-wrap:wrap;
            gap:.5rem .75rem;
            align-items:flex-start;
        ">

          <div style="
              background-color:{CARD_BG};
              border-radius:999px;
              border:1px solid rgba(148,163,184,0.35);
              padding:.4rem .75rem;
              font-size:.8rem;
              font-weight:500;
              line-height:1.2;
              color:{TEXT_COL};
              display:flex;
              align-items:center;
              gap:.4rem;
              white-space:nowrap;
          ">
            <span style="color:#94a3b8;font-weight:400;">Regime</span>
            <span>{regime}</span>
          </div>

          <div style="
              background-color:{CARD_BG};
              border-radius:999px;
              border:1px solid rgba(148,163,184,0.35);
              padding:.4rem .75rem;
              font-size:.8rem;
              font-weight:500;
              line-height:1.2;
              color:{TEXT_COL};
              display:flex;
              align-items:center;
              gap:.4rem;
              white-space:nowrap;
          ">
            <span style="color:#94a3b8;font-weight:400;">Max Drawdown</span>
            <span>{dd_pct}</span>
          </div>

          <div style="
              background-color:{CARD_BG};
              border-radius:999px;
              border:1px solid rgba(148,163,184,0.35);
              padding:.4rem .75rem;
              font-size:.8rem;
              font-weight:500;
              line-height:1.2;
              color:{TEXT_COL};
              display:flex;
              align-items:center;
              gap:.4rem;
              white-space:nowrap;
          ">
            <span style="color:#94a3b8;font-weight:400;">Buying Power</span>
            <span>{bp}</span>
          </div>

          <div style="
              background-color:{CARD_BG};
              border-radius:999px;
              border:1px solid rgba(148,163,184,0.35);
              padding:.4rem .75rem;
              font-size:.8rem;
              font-weight:500;
              line-height:1.2;
              color:{TEXT_COL};
              display:flex;
              align-items:center;
              gap:.4rem;
              white-space:nowrap;
          ">
            <span style="color:#94a3b8;font-weight:400;">Updated</span>
            <span>{ts}</span>
          </div>

        </div>
      </div>

      <div style="margin-top:0.75rem;">
        {market_chip_html}
      </div>

    </div>
    """

    components.html(header_html, height=240, scrolling=False)

    # refresh button below header iframe
    if st.button("⟳ Refresh data", key="force_rerun"):
        st.rerun()


# ──────────────────────────────
# SIMPLE PAGE RENDER STUBS
# ──────────────────────────────
def page_portfolio_history():
    st.markdown(
        '<div class="triton-card"><h3>📈 Portfolio History</h3>'
        '<p class="data-label">Cumulative PnL / equity curve / drawdown timeline.</p></div>',
        unsafe_allow_html=True,
    )


def page_trade_log():
    st.markdown(
        '<div class="triton-card"><h3>📝 Trade Log</h3>'
        '<p class="data-label">Full trade table with ticker, side, qty, PnL, SL/TP hits.</p></div>',
        unsafe_allow_html=True,
    )


def page_allocations():
    st.markdown(
        '<div class="triton-card"><h3>🏗 Portfolio Allocations</h3>'
        '<p class="data-label">Current weights by ticker/sector, plus exposure caps.</p></div>',
        unsafe_allow_html=True,
    )


def page_sltp():
    st.markdown(
        '<div class="triton-card"><h3>🎯 SL/TP Performance</h3>'
        '<p class="data-label">Stop loss vs take profit effectiveness, win rate, avg hold time.</p></div>',
        unsafe_allow_html=True,
    )


def page_ai_signals():
    st.markdown(
        '<div class="triton-card"><h3>🤖 AI Signals</h3>'
        '<p class="data-label">Latest BUY / SELL / WAIT per ticker, with confidence.</p></div>',
        unsafe_allow_html=True,
    )


def page_top_picks():
    st.markdown(
        '<div class="triton-card"><h3>⭐ Top Picks</h3>'
        '<p class="data-label">Ranked candidates from fundamentals + momentum + sentiment fusion.</p></div>',
        unsafe_allow_html=True,
    )


def page_feature_importance():
    st.markdown(
        '<div class="triton-card"><h3>🔬 Feature Importance</h3>'
        '<p class="data-label">Which features the model says are driving predictions.</p></div>',
        unsafe_allow_html=True,
    )


def page_model_comparison():
    st.markdown(
        '<div class="triton-card"><h3>📊 Model Comparison</h3>'
        '<p class="data-label">RF vs XGBoost vs LSTM vs Baseline, per ticker.</p></div>',
        unsafe_allow_html=True,
    )


def page_risk_report():
    st.markdown(
        '<div class="triton-card"><h3>🛡 Risk Report</h3>'
        '<p class="data-label">VaR, CVaR, concentration, capital protection status.</p></div>',
        unsafe_allow_html=True,
    )


def page_strategy_diagnostics():
    st.markdown(
        '<div class="triton-card"><h3>🧪 Strategy Diagnostics</h3>'
        '<p class="data-label">Signal accuracy, win/loss distribution, profit per trade.</p></div>',
        unsafe_allow_html=True,
    )


def page_adaptive_risk():
    st.markdown(
        '<div class="triton-card"><h3>📉 Adaptive Risk / Regime Monitor</h3>'
        '<p class="data-label">Regime (bull/bear/volatile), exposure targets, defensive mode.</p></div>',
        unsafe_allow_html=True,
    )


def page_stress_test():
    st.markdown(
        '<div class="triton-card"><h3>🔥 Stress Test Snapshot</h3>'
        '<p class="data-label">Tail-risk shocks, gap-down scenarios, worst-case drawdown.</p></div>',
        unsafe_allow_html=True,
    )


def page_news_sentiment():
    st.markdown(
        '<div class="triton-card"><h3>📰 News Sentiment</h3>'
        '<p class="data-label">Ticker-level sentiment feed & market mood.</p></div>',
        unsafe_allow_html=True,
    )


def page_smart_alerts():
    st.markdown(
        '<div class="triton-card"><h3>🚨 Smart Alerts</h3>'
        '<p class="data-label">Alerts Triton is watching (liquidity shocks, unusual flows).</p></div>',
        unsafe_allow_html=True,
    )


def page_econ_calendar():
    st.markdown(
        '<div class="triton-card"><h3>📅 Economic Calendar</h3>'
        '<p class="data-label">Upcoming macro events and impact level.</p></div>',
        unsafe_allow_html=True,
    )


def page_baseline_weights():
    st.markdown(
        '<div class="triton-card"><h3>📦 Baseline Weights</h3>'
        '<p class="data-label">Reference portfolio, drift vs target, safety rails.</p></div>',
        unsafe_allow_html=True,
    )


def page_advisor_introspection():
    st.markdown(
        '<div class="triton-card"><h3>🧠 Advisor Introspection</h3>'
        '<p class="data-label">How each advisor persona is positioned and performing.</p></div>',
        unsafe_allow_html=True,
    )


def page_execution_health():
    st.markdown(
        '<div class="triton-card"><h3>⚙ Execution / Health</h3>'
        '<p class="data-label">Broker link, buying power, scheduler heartbeat, open orders health.</p></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────
# BODY ROUTER
# ──────────────────────────────
def render_body(section: str, page: str):
    # Portfolio
    if section == "Portfolio":
        if page == "Portfolio History":
            page_portfolio_history()
        elif page == "Trade Log":
            page_trade_log()
        elif page == "Allocations":
            page_allocations()
        elif page == "SL/TP Performance":
            page_sltp()

    # Signals
    elif section == "Signals":
        if page == "AI Signals":
            page_ai_signals()
        elif page == "Top Picks":
            page_top_picks()
        elif page == "Feature Importance":
            page_feature_importance()
        elif page == "Model Comparison":
            page_model_comparison()

    # Risk & Guardrails
    elif section == "Risk & Guardrails":
        if page == "Risk Report":
            page_risk_report()
        elif page == "Strategy Diagnostics":
            page_strategy_diagnostics()
        elif page == "Adaptive Risk / Regime Monitor":
            page_adaptive_risk()
        elif page == "Stress Test Snapshot":
            page_stress_test()

    # Research / Intel
    elif section == "Research / Intel":
        if page == "News Sentiment":
            page_news_sentiment()
        elif page == "Smart Alerts":
            page_smart_alerts()
        elif page == "Economic Calendar":
            page_econ_calendar()

    # System
    elif section == "System":
        if page == "Baseline Weights":
            page_baseline_weights()
        elif page == "Advisor Introspection":
            page_advisor_introspection()
        elif page == "Execution / Health":
            page_execution_health()


# ──────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────
SECTIONS = {
    "Portfolio": [
        "Portfolio History",
        "Trade Log",
        "Allocations",
        "SL/TP Performance",
    ],
    "Signals": [
        "AI Signals",
        "Top Picks",
        "Feature Importance",
        "Model Comparison",
    ],
    "Risk & Guardrails": [
        "Risk Report",
        "Strategy Diagnostics",
        "Adaptive Risk / Regime Monitor",
        "Stress Test Snapshot",
    ],
    "Research / Intel": [
        "News Sentiment",
        "Smart Alerts",
        "Economic Calendar",
    ],
    "System": [
        "Baseline Weights",
        "Advisor Introspection",
        "Execution / Health",
    ],
}


def _sidebar_controls():
    global PROJECT_ROOT, DATA_ROOT, RESULTS_DIR, ORDERS_DIR, PRED_DIR, STRESS_DIR

    with st.sidebar:
        st.title("TRITON Nav")

        # Advanced / project root selector
        st.subheader("⚙️ Advanced")
        root_mode = st.radio(
            "Project root",
            ["Auto (this file’s folder)", "Manual path"],
            index=0,
            key="root_mode_choice",
        )

        if root_mode == "Manual path":
            default_text = str(st.session_state.get("custom_root", PROJECT_ROOT))
            custom_root = st.text_input(
                "Enter absolute path to repo root",
                value=default_text,
                key="root_manual_input",
            )
            try:
                PROJECT_ROOT = Path(custom_root).expanduser().resolve()
                st.session_state["custom_root"] = str(PROJECT_ROOT)
                st.caption(f"Using custom root: {PROJECT_ROOT}")
            except Exception as e:
                st.error(f"Invalid custom root: {e}")
        else:
            st.caption(f"Using auto root: {PROJECT_ROOT}")

        st.caption(f"Build: {APP_VERSION}")

        # Re-derive dirs now that PROJECT_ROOT may have changed
        DATA_ROOT = PROJECT_ROOT / "data"
        RESULTS_DIR = DATA_ROOT / "results"
        ORDERS_DIR = DATA_ROOT / "orders"
        PRED_DIR = DATA_ROOT / "predictions"
        STRESS_DIR = DATA_ROOT / "stress_test_results"
        for p in (RESULTS_DIR, ORDERS_DIR, PRED_DIR, STRESS_DIR):
            p.mkdir(parents=True, exist_ok=True)

        st.markdown("---")

        # Main nav
        section_choice = st.selectbox(
            "Section",
            list(SECTIONS.keys()),
            index=0,
            help="High-level area of Triton",
            key="section_choice",
        )

        subpages = SECTIONS[section_choice]
        sub_choice = st.radio(
            "View",
            subpages,
            index=0,
            help="Which view inside that section",
            key="sub_choice",
        )

    return section_choice, sub_choice


# ──────────────────────────────
# APP ENTRYPOINT
# ──────────────────────────────
def _render_main():
    section_choice, sub_choice = _sidebar_controls()
    render_global_header()  # dark header + chips + market pill in iframe
    render_body(section_choice, sub_choice)


# Run it
_render_main()

tab_labels = [
    "🔍 Portfolio Drilldown",  # 0
    "📈 Portfolio History",  # 1
    "📋 Trade Log",  # 2
    "📊 Strategy vs Market",  # 3
    "🧠 AI Signals + Rationale",  # 4
    "📁 Browse Any CSV",  # 5
    "📋 Backtest Summary",  # 6
    "📉 Risk: Portfolio Drawdown",  # 7
    "📊 Strategy Diagnostics",  # 8
    "🏦 Portfolio Allocations",  # 9
    "📽️ Trade Replay",  # 10
    "📘 Fundamental Data",  # 11
    "📈 Stock Scores",  # 12
    "🎯 Top Fundamental Picks",  # 13
    "📰 News Sentiment",  # 14
    "🚨 Smart Alerts",  # 15
    "📆 Economic Calendar",  # 16
    "🔬 Feature Importance",  # 17
    "🎯 SL/TP Performance Analysis",  # 18
    "💬 Sentiment + Signal Fusion",  # 19
    "📊 Model Comparison",  # 20
    "🧠 AI Learning Lab",  # 21
    "🧾 Buffett Orders (current)",  # 22
    "🗂️ Consolidated Orders (ML × Buffett blend)",  # 23
    "🤖 AI Feedback (allocator runs)",  # 24
    "📚 Equal-Weight Portfolio vs Benchmark",  # 25
    "🧮 Smart-Weight Portfolio vs Benchmark",  # 26
    "🧪 Confidence Calibration",  # 27
    "🧪 Confidence-Filtered Portfolio vs Benchmark",  # 28
    "📊 Confidence × Sharpe Portfolio vs Benchmark",  # 29
    "🧪 Stress Test Reports & Runner",  # 30  # NEW
    "🩺 Market Sentinels",  # 31  # NEW
]

tabs = st.tabs(tab_labels)

# ─────────────────────────────────────────────
# Market status helper (US equities / NYSE hours)
# ─────────────────────────────────────────────
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


def _next_weekday(dt, weekday_target):
    """Return dt moved forward to the next weekday == weekday_target (0=Mon,...4=Fri).
    If dt is already that weekday and before that day's market open, we return same-day open."""
    days_ahead = (weekday_target - dt.weekday()) % 7
    return dt + timedelta(days=days_ahead)


def _next_market_open(now_et: datetime) -> datetime:
    """
    Figure out the next regular session open time (9:30 ET Mon-Fri).
    Does not yet handle US market holidays (future enhancement).
    """
    market_open_t = time(9, 30)
    market_close_t = time(16, 0)

    # If it's a weekday Mon-Fri
    if now_et.weekday() < 5:
        # Before open today -> opens today
        if now_et.time() < market_open_t:
            return now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        # During session -> we're already open, so "next open" is actually next session open (next biz day 9:30)
        if market_open_t <= now_et.time() < market_close_t:
            # next business day 9:30
            d = now_et + timedelta(days=1)
        else:
            # After 4pm -> next business day 9:30
            d = now_et + timedelta(days=1)
    else:
        # Weekend: jump to Monday
        d = now_et + timedelta(days=1)

    # move forward until Mon-Fri
    while d.weekday() >= 5:
        d += timedelta(days=1)

    return d.replace(hour=9, minute=30, second=0, microsecond=0)


def _market_close_today(now_et: datetime) -> datetime | None:
    """
    If we are currently inside the regular session, return today's 4:00 PM ET close.
    Otherwise None.
    """
    market_open_t = time(9, 30)
    market_close_t = time(16, 0)

    if now_et.weekday() < 5 and market_open_t <= now_et.time() < market_close_t:
        return now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return None


def _fmt_timedelta(td: timedelta) -> str:
    """Format a timedelta cleanly like '1d 3h 12m' or '3h 07m'."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    mins = (total_seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    parts.append(f"{hours}h")
    parts.append(f"{mins:02d}m")
    return " ".join(parts)


def get_market_status():
    """
    Returns dict:
    {
        "is_open": bool,
        "label": "OPEN" | "CLOSED",
        "detail": "closes in 2h 11m" or "opens in 1d 3h 02m",
        "next_event_ts": datetime (ET),
        "now_et": datetime (ET),
    }
    """
    et = datetime.now(ZoneInfo("America/New_York"))
    open_t = time(9, 30)
    close_t = time(16, 0)

    # Are we in a regular session?
    in_session = et.weekday() < 5 and open_t <= et.time() < close_t

    if in_session:
        close_ts = _market_close_today(et)
        if close_ts is None:
            # fallback, shouldn't normally happen
            close_ts = et.replace(hour=16, minute=0, second=0, microsecond=0)
        remaining = close_ts - et
        return {
            "is_open": True,
            "label": "OPEN",
            "detail": f"closes in {_fmt_timedelta(remaining)}",
            "next_event_ts": close_ts,
            "now_et": et,
        }
    else:
        # Market closed. When is next open?
        nxt = _next_market_open(et)
        remaining = nxt - et
        return {
            "is_open": False,
            "label": "CLOSED",
            "detail": f"opens in {_fmt_timedelta(remaining)}",
            "next_event_ts": nxt,
            "now_et": et,
        }


# ─────────────────────────────────────────────
# Market status banner
# ─────────────────────────────────────────────
ms = get_market_status()

# choose color vibe
if ms["is_open"]:
    bg = "#e8fff1"  # light green
    border = "#2e7d32"
    emoji = "🟢"
else:
    bg = "#fff5e6"  # light orange / amber
    border = "#b26a00"
    emoji = "🔴"

banner_html = f"""
<div style="
    background:{bg};
    border:1px solid {border};
    border-radius:8px;
    padding:0.75rem 1rem;
    margin-bottom:1rem;
    font-family:system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display:flex;
    flex-wrap:wrap;
    align-items:flex-start;
    gap:1.5rem;
">
  <div style="font-size:0.9rem; line-height:1.4;">
    <div style="font-weight:600; font-size:1rem;">
      {emoji} Market {ms['label']}
    </div>
    <div style="opacity:0.8;">
      {ms['detail']}
    </div>
  </div>

  <div style="font-size:0.8rem; line-height:1.4; opacity:0.8;">
    <div>Now (ET): {ms['now_et'].strftime('%Y-%m-%d %H:%M')}</div>
    <div>Next event: {ms['next_event_ts'].strftime('%Y-%m-%d %H:%M')} ET</div>
  </div>
</div>
"""

st.markdown(banner_html, unsafe_allow_html=True)

# ──────────────────────────────
# Tab 0 — Portfolio Drilldown (live guard + orders + per-ticker deep dive)
# ──────────────────────────────
with tabs[0]:
    st.subheader("🔍 Portfolio Drilldown")

    # --- Top status / capital preservation snapshot ---
    status = latest_portfolio_status()
    colA, colB, colC, colD = st.columns(4)

    with colA:
        latest_eq = status.get("latest_equity", np.nan)
        st.metric(
            "Latest Equity",
            f"${latest_eq:,.2f}" if np.isfinite(latest_eq) else "—",
        )

    with colB:
        draw = status.get("drawdown_pct", np.nan)
        st.metric(
            "Drawdown",
            f"{draw:.1%}" if np.isfinite(draw) else "—",
            help="How far below peak we are right now.",
        )

    with colC:
        bp = status.get("buying_power", np.nan)
        st.metric(
            "Buying Power",
            f"${bp:,.0f}" if np.isfinite(bp) else "—",
            help="From broker /v2/account buying_power.",
        )

    with colD:
        reserve = status.get("reserve_pct", np.nan)
        mode = status.get("mode", "UNKNOWN")
        mode_label = f"{mode} ({reserve:.0%} reserve)" if np.isfinite(reserve) else mode
        st.metric(
            "Guard Mode",
            mode_label,
            help=status.get("reason", ""),
        )

    # --- Most recent order intents (what Triton is trying to execute) ---
    open_orders_df = load_open_orders()
    with st.expander("📝 Recent Order Intents (latest first)"):
        if open_orders_df.empty:
            st.caption("No recent order intents logged yet.")
        else:
            # show top ~30 most recent orders. Cleaner subset of columns.
            cols = [
                c
                for c in [
                    "timestamp",
                    "symbol",
                    "side",
                    "qty",
                    "order_type",
                    "limit_price",
                    "status",
                    "note",
                ]
                if c in open_orders_df.columns
            ]
            st.dataframe(open_orders_df[cols].head(30), use_container_width=True)

    st.markdown("---")

    # --- Load data sources we might drill into for a specific ticker ---
    tl = load_csv("trade_log.csv", RESULTS_DIR)
    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig.empty:
        sig = load_csv("signals.csv", RESULTS_DIR)

    ns = load_csv("news_sentiment.csv", RESULTS_DIR)
    cur_orders = load_csv("orders_today.csv", ORDERS_DIR)
    bo = load_csv("buffett_orders.csv", ORDERS_DIR)

    # Universe of tickers from whatever we have
    tickers = set()
    for df_ in (tl, sig, ns, cur_orders, bo):
        if not df_.empty and "ticker" in df_.columns:
            tickers.update(df_["ticker"].dropna().astype(str).unique())
    tickers = sorted(tickers)

    if not tickers:
        st.info("No tickers found across signals / trades / orders / news yet.")
    else:
        # Controls
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            sel = st.selectbox("Ticker", tickers, index=0, key="t0_ticker")
        with c2:
            lookback = st.slider("Lookback (days)", 30, 365, 180, 15, key="t0_lb")
        with c3:
            view = st.selectbox("Price View", ["Line", "Candlestick"], index=0, key="t0_view")

        # We build cutoff timestamp in naive UTC
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lookback)

        # ── Trades (filtered for this ticker, recent only)
        tl_t = pd.DataFrame()
        if not tl.empty:
            tl_t = ensure_date(tl, normalize=True).copy()
            if "ticker" in tl_t.columns:
                tl_t["ticker"] = tl_t["ticker"].astype(str)
                tl_t = tl_t[(tl_t["ticker"] == sel) & (tl_t["date"] >= cutoff)].copy()
            else:
                tl_t = pd.DataFrame()

            if "profit" in tl_t.columns:
                tl_t["profit"] = pd.to_numeric(tl_t["profit"], errors="coerce")

        # ── Signals (filtered)
        sig_t = pd.DataFrame()
        if not sig.empty:
            sig_t = ensure_date(sig, normalize=True).copy()
            if "ticker" in sig_t.columns:
                sig_t["ticker"] = sig_t["ticker"].astype(str)
                sig_t = sig_t[(sig_t["ticker"] == sel) & (sig_t["date"] >= cutoff)].copy()
            else:
                sig_t = pd.DataFrame()

            # safe numerics
            for c in ["close", "predicted_close", "confidence"]:
                if c in sig_t.columns:
                    sig_t[c] = pd.to_numeric(sig_t[c], errors="coerce")

            # backfill 'close' from {ticker}.parquet if missing
            if not sig_t.empty and ("close" not in sig_t.columns or sig_t["close"].isna().any()):
                sig_t = backfill_close_from_parquet(sig_t)

            # expected edge %
            with np.errstate(divide="ignore", invalid="ignore"):
                base = sig_t.get("close")
                pred = sig_t.get("predicted_close")
                sig_t["edge_pct"] = np.where(
                    base.notna() & pred.notna() & (base != 0),
                    (pred - base) / base,
                    np.nan,
                )

        # ── News (filtered)
        ns_t = pd.DataFrame()
        if not ns.empty:
            ns_t = ensure_date(ns, normalize=True).copy()
            if "ticker" in ns_t.columns:
                ns_t["ticker"] = ns_t["ticker"].astype(str)
                ns_t = ns_t[(ns_t["ticker"] == sel) & (ns_t["date"] >= cutoff)].copy()
            else:
                ns_t = pd.DataFrame()

            # Ensure URL column exists (fallbacks) and clean description HTML
            if "url" not in ns_t.columns or ns_t["url"].isna().all():
                for alt in ["link", "source_url"]:
                    if alt in ns_t.columns and ns_t[alt].notna().any():
                        ns_t["url"] = ns_t[alt]
                        break
            if "description" in ns_t.columns:
                # If url still missing, try extracting from HTML; always strip HTML
                if "url" not in ns_t.columns or ns_t["url"].isna().all():
                    ns_t["url"] = ns_t["description"].apply(extract_href)
                ns_t["description"] = ns_t["description"].apply(strip_html)

            # clickable title if present
            if {"title", "url"}.issubset(ns_t.columns):
                ns_t["news"] = ns_t.apply(
                    lambda r: make_clickable(r.get("title", ""), r.get("url", "")),
                    axis=1,
                )

        # ── KPIs for this ticker (trades & PnL)
        trades = int(len(tl_t))
        wins = int((tl_t["profit"] > 0).sum()) if "profit" in tl_t.columns else 0
        win_rate = f"{(wins / trades):.0%}" if trades else "0%"
        cum_pnl = f"{tl_t['profit'].sum():,.2f}" if "profit" in tl_t.columns and trades else "—"

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Trades", trades)
        with k2:
            st.metric("Win Rate", win_rate)
        with k3:
            st.metric("Cum P&L", cum_pnl)

        # ── Price / Signal chart for this ticker
        if not PLOTLY_OK:
            st.warning("Plotly not installed — `pip install plotly` for charts.")
        else:
            fig = go.Figure()
            added_price = False

            # Candlestick from OHLC parquet {ticker}.parquet
            if view == "Candlestick":
                ohlc_path = RESULTS_DIR / f"{sel}.parquet"
                ohlc = load_parquet(ohlc_path)
                if not ohlc.empty and {"date", "open", "high", "low", "close"}.issubset(
                    ohlc.columns
                ):
                    parse_dates_inplace(ohlc, ("date",))
                    ohlc = ohlc.dropna(subset=["date"]).sort_values("date").query("date >= @cutoff")
                    if not ohlc.empty:
                        fig.add_trace(
                            go.Candlestick(
                                x=ohlc["date"],
                                open=pd.to_numeric(ohlc["open"], errors="coerce"),
                                high=pd.to_numeric(ohlc["high"], errors="coerce"),
                                low=pd.to_numeric(ohlc["low"], errors="coerce"),
                                close=pd.to_numeric(ohlc["close"], errors="coerce"),
                                name="Price",
                            )
                        )
                        added_price = True

            # Line fallback from signals.close
            if not added_price and not sig_t.empty and "close" in sig_t.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sig_t["date"],
                        y=sig_t["close"],
                        mode="lines",
                        name="Close",
                        opacity=0.85,
                    )
                )

            # Predicted_close overlay if present
            if not sig_t.empty and "predicted_close" in sig_t.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sig_t["date"],
                        y=sig_t["predicted_close"],
                        mode="lines",
                        name="Predicted",
                        opacity=0.55,
                    )
                )

            fig.update_layout(
                title=f"{sel} — Price & Signals",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=(view != "Line"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Detailed tables
        with st.expander("Signals Table"):
            st.dataframe(sig_t.tail(200), use_container_width=True)

        with st.expander("Related News"):
            if not ns_t.empty:
                cols = [
                    c
                    for c in [
                        "date",
                        "ticker",
                        "sentiment",
                        "news",
                        "description",
                        "source",
                    ]
                    if c in ns_t.columns or c == "news"
                ]
                disp = ns_t[cols] if cols else ns_t
                if "date" in disp.columns:
                    disp = disp.sort_values("date", ascending=False)
                st.markdown(
                    disp.to_html(escape=False, index=False),
                    unsafe_allow_html=True,
                )
            else:
                st.info("No news for this ticker in the selected window.")

# ──────────────────────────────
# Tab 1 — Portfolio History (equity curve + metrics + drawdown inline)
# ──────────────────────────────
with tabs[1]:
    st.subheader("📈 Portfolio Value Over Time")

    hist = load_portfolio_history()  # timestamp,equity from live run snapshots
    guard = latest_portfolio_status()  # in case hist is thin, we still get latest stats

    if hist.empty and (not np.isfinite(guard.get("latest_equity", np.nan))):
        st.info("No portfolio history yet.")
    else:
        # clean curve for plotting / stats
        df = hist.copy()

        if not df.empty:
            # ensure timestamp is datetime and monotonic
            if "timestamp" in df.columns:
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                df = df.rename(columns={"timestamp": "date"})

            df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
            df = df.dropna(subset=["date", "equity"])

        # compute perf stats from the equity series
        eq_series = (
            df.set_index("date")["equity"]
            if (not df.empty and "date" in df.columns)
            else pd.Series(
                [guard.get("latest_equity", np.nan)],
                index=[pd.Timestamp.utcnow().tz_localize(None)],
                name="equity",
            )
        )

        stats = perf_stats_from_levels(eq_series)

        # drawdown calc inline for the curve
        dd_pct = np.nan
        if eq_series.size > 0:
            peak_curve = eq_series.cummax()
            dd_curve = eq_series / peak_curve - 1.0
            if dd_curve.size > 0:
                dd_pct = float(dd_curve.min())

        # KPIs row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Total Return",
                (
                    f"{stats.get('total_return', np.nan):.1%}"
                    if np.isfinite(stats.get("total_return", np.nan))
                    else "—"
                ),
            )
        with c2:
            st.metric(
                "CAGR",
                (
                    f"{stats.get('cagr', np.nan):.1%}"
                    if np.isfinite(stats.get("cagr", np.nan))
                    else "—"
                ),
            )
        with c3:
            st.metric(
                "Sharpe",
                (
                    f"{stats.get('sharpe', np.nan):.2f}"
                    if np.isfinite(stats.get("sharpe", np.nan))
                    else "—"
                ),
            )
        with c4:
            st.metric(
                "Max Drawdown",
                f"{dd_pct:.1%}" if np.isfinite(dd_pct) else "—",
            )

        # Plot
        if df.empty:
            st.caption(
                "Not enough time series to chart yet — only showing latest snapshot from guard info."
            )
            st.write(
                f"Latest Equity: ${guard.get('latest_equity', np.nan):,.2f}  "
                f"(Mode: {guard.get('mode','UNKNOWN')})"
            )
        else:
            if not PLOTLY_OK:
                st.warning("Plotly not installed — `pip install plotly` for charts.")
                st.line_chart(df.set_index("date")["equity"].rename("Equity ($)"))
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df["equity"],
                        mode="lines",
                        name="Equity ($)",
                    )
                )
                fig.update_layout(
                    title="Portfolio Equity Curve",
                    xaxis_title="Time",
                    yaxis_title="Equity ($)",
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Download
        if not df.empty:
            st.download_button(
                "⬇️ Download portfolio history (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="portfolio_history.csv",
                mime="text/csv",
                key="t1_dl",
            )

# ──────────────────────────────
# Tab 2 — Trade Log
# ──────────────────────────────
with tabs[2]:
    st.subheader("📋 Trade Log")
    df = load_csv("trade_log.csv", RESULTS_DIR)
    st.dataframe(df if not df.empty else pd.DataFrame([{"info": "No trade_log.csv yet."}]))
# ──────────────────────────────
# Tab 3 — Strategy vs Market (robust)
# ──────────────────────────────
with tabs[3]:
    st.subheader("📊 Strategy vs Market")

    df = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if df.empty:
        st.info("No strategy_vs_market.csv yet.")
    else:
        # Dates & sanity
        parse_dates_inplace(df, ("date",))
        if "ticker" not in df.columns:
            st.warning("Missing 'ticker' column.")
            st.stop()

        # What do we have?
        has_ret = {"strategy_return", "market_return"}.issubset(df.columns)
        has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(df.columns)
        if not (has_ret or has_cum):
            st.warning("Expected daily returns or cumulative levels (columns missing).")
            st.stop()

        # UI
        tickers = sorted(df["ticker"].dropna().astype(str).unique())
        if not tickers:
            st.info("No tickers available.")
            st.stop()
        c1, c2 = st.columns([1.2, 1])
        with c1:
            sel = st.selectbox("Select ticker", tickers, key="t3_ticker")
        with c2:
            normalize = st.checkbox("Normalize curves to 1.0 at start", True, key="t3_norm")

        # Subset + compute cumulative if needed
        sub = (
            df[df["ticker"].astype(str) == sel]
            .dropna(subset=["date"])
            .sort_values("date")
            .set_index("date")
            .copy()
        )

        if sub.empty:
            st.info("No rows for the selected ticker.")
            st.stop()

        if has_ret and not has_cum:
            sub["strategy_return"] = pd.to_numeric(sub["strategy_return"], errors="coerce").fillna(
                0
            )
            sub["market_return"] = pd.to_numeric(sub["market_return"], errors="coerce").fillna(0)
            sub["cumulative_strategy"] = (1 + sub["strategy_return"]).cumprod()
            sub["cumulative_market"] = (1 + sub["market_return"]).cumprod()
        else:
            # Ensure numerics if cum columns already present
            for c in ["cumulative_strategy", "cumulative_market"]:
                if c in sub.columns:
                    sub[c] = pd.to_numeric(sub[c], errors="coerce")

        # Extract series
        cs_raw = sub.get("cumulative_strategy", pd.Series(dtype=float)).dropna()
        cm_raw = sub.get("cumulative_market", pd.Series(dtype=float)).dropna()

        if cs_raw.empty or cm_raw.empty:
            st.info("Not enough data to plot for this ticker.")
            st.stop()

        # Stats should be computed on the raw curves (not normalized)
        s_stats = perf_stats_from_levels(cs_raw)
        b_stats = perf_stats_from_levels(cm_raw)

        # Normalize for plotting if user wants
        cs_plot = normalize_to_one(cs_raw) if normalize else cs_raw
        cm_plot = normalize_to_one(cm_raw) if normalize else cm_raw

        # Chart
        if not PLOTLY_OK:
            st.warning("Plotly not installed — `pip install plotly` for charts.")
            st.line_chart(pd.concat({"Strategy": cs_plot, "Market": cm_plot}, axis=1))
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=cs_plot.index, y=cs_plot.values, name="Strategy", mode="lines")
            )
            fig.add_trace(
                go.Scatter(x=cm_plot.index, y=cm_plot.values, name="Market", mode="lines")
            )
            ttl = f"{sel} — Strategy vs Market" + (" (normalized)" if normalize else "")
            fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
            st.plotly_chart(fig, use_container_width=True)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                "Strategy Total",
                (
                    f"{s_stats.get('total_return', np.nan):.1%}"
                    if np.isfinite(s_stats.get("total_return", np.nan))
                    else "—"
                ),
            )
        with k2:
            st.metric(
                "Strategy CAGR",
                (
                    f"{s_stats.get('cagr', np.nan):.1%}"
                    if np.isfinite(s_stats.get("cagr", np.nan))
                    else "—"
                ),
            )
        with k3:
            st.metric(
                "Strategy Sharpe",
                (
                    f"{s_stats.get('sharpe', np.nan):.2f}"
                    if np.isfinite(s_stats.get("sharpe", np.nan))
                    else "—"
                ),
            )
        with k4:
            st.metric(
                "Strategy MaxDD",
                (
                    f"{s_stats.get('max_dd', np.nan):.1%}"
                    if np.isfinite(s_stats.get("max_dd", np.nan))
                    else "—"
                ),
            )

        k5, k6, _, _ = st.columns(4)
        with k5:
            st.metric(
                "Market Total",
                (
                    f"{b_stats.get('total_return', np.nan):.1%}"
                    if np.isfinite(b_stats.get("total_return", np.nan))
                    else "—"
                ),
            )
        with k6:
            st.metric(
                "Market CAGR",
                (
                    f"{b_stats.get('cagr', np.nan):.1%}"
                    if np.isfinite(b_stats.get("cagr", np.nan))
                    else "—"
                ),
            )

        # Download filtered
        out = sub.reset_index()[["date", "cumulative_strategy", "cumulative_market"]].rename(
            columns={"cumulative_strategy": "strategy", "cumulative_market": "market"}
        )
        st.download_button(
            "⬇️ Download curves (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"{sel}_strategy_vs_market.csv",
            mime="text/csv",
            key="t3_dl",
        )
# ──────────────────────────────
# Tab 4 — AI Signals (clean + robust)
# ──────────────────────────────
with tabs[4]:
    st.subheader("🧠 AI Signals + Rationale")

    # Load signals (prefer rationale version)
    df = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if df.empty:
        df = load_csv("signals.csv", RESULTS_DIR)

    if df.empty:
        st.info("No signals CSV yet.")
    else:
        # Ensure a proper 'date' column and a string ticker
        df = ensure_date(
            df,
            candidates=["date", "as_of", "timestamp", "time", "datetime", "Date"],
            normalize=False,
        )
        if "ticker" not in df.columns:
            df["ticker"] = np.nan
        df["ticker"] = df["ticker"].astype(str)

        # Ensure expected numeric columns exist
        numeric_cols = [
            "close",
            "predicted_close",
            "confidence",
            "rsi14",
            "sma20",
            "sma50",
            "atr14",
            "sentiment",
            "total_score",
            "pe_ratio",
            "dividend_yield",
        ]
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = np.nan
        to_numeric(df, numeric_cols)

        # Try to backfill missing close from per-ticker parquet files
        df = backfill_close_from_parquet(df)

        # Derived edge (%), safe against divide by zero
        if {"close", "predicted_close"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["edge_pct"] = ((df["predicted_close"] - df["close"]) / df["close"]).replace(
                    [np.inf, -np.inf], np.nan
                )

        # Order rows and build controls
        df = df.dropna(subset=["ticker"]).sort_values(["ticker", "date"])
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        with c1:
            tickers = sorted(df["ticker"].dropna().unique().tolist())
            selected_ticker = st.selectbox("Ticker", tickers, key="t4_ticker")
        with c2:
            sel_signals = st.multiselect(
                "Signals",
                ["BUY", "SELL", "HOLD"],
                default=["BUY", "SELL", "HOLD"],
                key="t4_signals",
            )
        with c3:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.00, 0.01, key="t4_minconf")
        with c4:
            chart_type = st.selectbox("Chart type", ["Line", "Candlestick"], key="t4_charttype")
        with c5:
            size_min, size_max = st.slider("Marker size range", 4, 32, (6, 22), key="t4_sizes")

        show_sma = st.checkbox("Overlay SMA(20)", value=False, key="t4_sma")

        # Filter for selected ticker & criteria
        f = df[df["ticker"] == selected_ticker].copy()
        if "signal" in f.columns:
            f = f[f["signal"].astype(str).isin(sel_signals)]
        if "confidence" in f.columns:
            f = f[f["confidence"].fillna(0) >= min_conf]

        if f.empty:
            st.info("No rows after filtering. Try different filters.")
        else:
            if not PLOTLY_OK:
                st.warning("Plotly not installed — `pip install plotly` for charts.")
            else:
                base = df[df["ticker"] == selected_ticker].copy().sort_values("date")
                has_base_dates = base["date"].notna().any()

                # Optional SMA(20) from close (numeric-safe)
                if "close" in base.columns:
                    base["close"] = pd.to_numeric(base["close"], errors="coerce")
                    base["sma20_calc"] = base["close"].rolling(20).mean()
                else:
                    base["sma20_calc"] = np.nan

                # Marker sizes from normalized confidence
                conf = (
                    f["confidence"].fillna(0.0)
                    if "confidence" in f.columns
                    else pd.Series([0] * len(f), index=f.index)
                )
                cmin, cmax = float(conf.min()), float(conf.max())
                conf_norm = (conf - cmin) / (cmax - cmin + 1e-9)
                f["conf_size"] = conf_norm * (size_max - size_min) + size_min

                fig = go.Figure()
                added_price = False

                # Candlestick if we have parquet OHLC
                if chart_type == "Candlestick" and has_base_dates:
                    ohlc_path = RESULTS_DIR / f"{selected_ticker}.parquet"
                    ohlc = load_parquet(ohlc_path)
                    if not ohlc.empty and {"date", "open", "high", "low", "close"}.issubset(
                        ohlc.columns
                    ):
                        parse_dates_inplace(ohlc, ("date",))
                        ohlc = ohlc.dropna(subset=["date"]).sort_values("date")
                        fig.add_trace(
                            go.Candlestick(
                                x=ohlc["date"],
                                open=ohlc["open"],
                                high=ohlc["high"],
                                low=ohlc["low"],
                                close=ohlc["close"],
                                name="Price",
                            )
                        )
                        added_price = True

                # Fallback to line close
                if not added_price and "close" in base.columns:
                    x_base = base["date"] if has_base_dates else np.arange(len(base))
                    fig.add_trace(
                        go.Scatter(
                            x=x_base,
                            y=base["close"],
                            mode="lines",
                            name="Price",
                            opacity=0.55,
                        )
                    )

                # Optional SMA overlay
                if show_sma and "sma20_calc" in base.columns:
                    x_base = base["date"] if has_base_dates else np.arange(len(base))
                    fig.add_trace(
                        go.Scatter(
                            x=x_base,
                            y=base["sma20_calc"],
                            mode="lines",
                            name="SMA(20)",
                            opacity=0.85,
                        )
                    )

                # Plot signal markers, with rationale in hover
                if "signal" in f.columns and "close" in f.columns:
                    # Ensure numeric for plotting
                    f["close"] = pd.to_numeric(f["close"], errors="coerce")
                    f["predicted_close"] = pd.to_numeric(
                        f.get("predicted_close", np.nan), errors="coerce"
                    )
                    f["edge_pct"] = pd.to_numeric(f.get("edge_pct", np.nan), errors="coerce")
                    has_f_dates = f["date"].notna().any()

                    for sig_name, dfg in f.groupby("signal"):
                        x_vals = dfg["date"] if has_f_dates else np.arange(len(dfg))
                        hover_x = "%{x|%Y-%m-%d}" if has_f_dates else "%{x}"
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=dfg["close"],
                                mode="markers",
                                name=str(sig_name),
                                marker=dict(size=dfg["conf_size"]),
                                hovertemplate=(
                                    f"<b>{hover_x}</b><br>"
                                    "Close: %{y:.2f}<br>"
                                    "Predicted: %{customdata[2]:.2f}<br>"
                                    f"Signal: {sig_name}<br>"
                                    "Confidence: %{customdata[0]:.2f}<br>"
                                    "Edge: %{customdata[1]:.2%}<br>"
                                    "<br><i>%{customdata[3]}</i><extra></extra>"
                                ),
                                customdata=np.stack(
                                    [
                                        dfg.get("confidence", pd.Series(0, index=dfg.index))
                                        .fillna(0)
                                        .values,
                                        dfg.get("edge_pct", pd.Series(0, index=dfg.index))
                                        .fillna(0)
                                        .values,
                                        dfg.get("predicted_close", pd.Series(0, index=dfg.index))
                                        .fillna(0)
                                        .values,
                                        dfg.get("rationale", pd.Series("", index=dfg.index))
                                        .fillna("")
                                        .values,
                                    ],
                                    axis=-1,
                                ),
                            )
                        )

                fig.update_layout(
                    title=f"{selected_ticker} — Signals over time (hover for rationale)",
                    xaxis_title="date" if has_base_dates else "index",
                    yaxis_title="close",
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Table + Download
            with st.expander("Show table"):
                cols = [
                    "date",
                    "ticker",
                    "close",
                    "predicted_close",
                    "edge_pct",
                    "signal",
                    "confidence",
                    "rsi14",
                    "sma20",
                    "sma50",
                    "atr14",
                    "sentiment",
                    "total_score",
                    "pe_ratio",
                    "dividend_yield",
                    "rationale",
                ]
                cols = [c for c in cols if c in f.columns]
                st.dataframe(f[cols], use_container_width=True)

            st.download_button(
                "⬇️ Download filtered signals (CSV)",
                data=f.to_csv(index=False).encode("utf-8"),
                file_name=f"signals_filtered_{selected_ticker}.csv",
                mime="text/csv",
                key="t4_dl",
            )
# ──────────────────────────────
# Tab 5 — Raw CSV Browser (multi-folder)
# ──────────────────────────────
with tabs[5]:
    st.subheader("📁 Browse Any CSV")

    def _list_csvs(root: Path) -> list[Path]:
        try:
            return sorted([p for p in root.glob("*.csv")], key=lambda p: p.name.lower())
        except Exception:
            return []

    c1, c2, c3, _ = st.columns([2, 1.2, 2, 1])
    with c1:
        which_dir: Path = st.selectbox(
            "Folder",
            [RESULTS_DIR, ORDERS_DIR, PRED_DIR],
            format_func=lambda p: str(p),
            key="t5_folder",
        )
    with c2:
        if st.button("↻ Refresh list", key="t5_refresh"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()
    with c3:
        name_filter = st.text_input("Filename filter (contains)", value="", key="t5_filter")

    files = _list_csvs(which_dir)
    if name_filter.strip():
        files = [p for p in files if name_filter.lower() in p.name.lower()]

    st.caption(f"Found {len(files)} CSVs in {which_dir}")

    if not files:
        st.info(f"No CSV files found in {which_dir}.")
    else:
        names = [p.name for p in files]
        sel_key = f"t5_file_{hash(str(which_dir))}"
        selected = st.selectbox("Select a file", names, key=sel_key)

        # Load via the shared helper (no separate load_csv_from needed)
        df = load_csv(selected, which_dir)

        # Context header
        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Rows", len(df))
        with cB:
            st.metric("Columns", df.shape[1] if not df.empty else 0)
        with cC:
            st.metric("Filename", selected)

        if df.empty:
            st.warning("This CSV is empty or failed to parse.")
        else:
            st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download this CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=selected,
            mime="text/csv",
            key="t5_dl",
        )
# ──────────────────────────────
# Tab 6 — Backtest Summary (robust)
# ──────────────────────────────
with tabs[6]:
    st.subheader("📋 Backtest Summary")

    df = load_csv("backtest_summary.csv", RESULTS_DIR)
    if df.empty:
        st.info("No backtest_summary.csv yet.")
    else:
        # Light cleanup
        parse_dates_inplace(df, ("date",), normalize=True)
        for col in ["sharpe", "cagr", "total_return", "win_rate", "rmse", "mae"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # KPIs row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Rows", len(df))
        with k2:
            st.metric("Unique tickers", df["ticker"].nunique() if "ticker" in df.columns else 0)
        with k3:
            if "date" in df.columns and df["date"].notna().any():
                dmin = df["date"].min()
                dmax = df["date"].max()
                st.metric("Date range", f"{dmin.date()} → {dmax.date()}")
            else:
                st.metric("Date range", "—")
        with k4:
            if "sharpe" in df.columns:
                st.metric("Avg Sharpe", f"{df['sharpe'].mean():.2f}")
            elif "cagr" in df.columns:
                st.metric("Avg CAGR", f"{df['cagr'].mean():.2%}")
            else:
                st.metric("Summary", "—")

        # Sorting controls (only show valid numeric sort columns)
        numeric_cols = [
            c
            for c in ["sharpe", "cagr", "total_return", "win_rate", "rmse", "mae"]
            if c in df.columns
        ]
        left, mid = st.columns([1.2, 1])
        with left:
            sort_col = st.selectbox("Sort by", options=numeric_cols or ["(none)"], index=0)
        with mid:
            sort_dir = st.radio("Direction", ["Desc", "Asc"], horizontal=True)

        df_disp = df.copy()
        if sort_col in df_disp.columns:
            df_disp = df_disp.sort_values(
                sort_col, ascending=(sort_dir == "Asc"), na_position="last"
            )

        # Optional bar chart if we have a “key” column + a metric
        key_col = None
        for candidate in ["strategy", "model", "ticker"]:
            if candidate in df.columns:
                key_col = candidate
                break

        metric_col = (
            sort_col if sort_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        )

        if PLOTLY_OK and key_col and metric_col:
            topN = st.slider("Show top N in chart", 5, 50, 15, 1, key="t6_topn")
            chart_df = (
                df_disp[[key_col, metric_col]]
                .dropna(subset=[metric_col])
                .groupby(key_col, as_index=False)[metric_col]
                .mean()
                .sort_values(metric_col, ascending=False)
                .head(topN)
            )
            fig = px.bar(chart_df, x=key_col, y=metric_col, title=f"Top {topN} by {metric_col}")
            st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(df_disp, use_container_width=True)

        # Download
        st.download_button(
            "⬇️ Download summary (CSV)",
            data=df_disp.to_csv(index=False).encode("utf-8"),
            file_name="backtest_summary_clean.csv",
            mime="text/csv",
            key="t6_dl",
        )
# ──────────────────────────────
# Tab 7 — Risk Report (Drawdown + Guard Mode)
# ──────────────────────────────
with tabs[7]:
    st.subheader("📉 Risk: Portfolio Drawdown & Capital Guard")

    # Pull live risk posture (guard mode, drawdown) plus equity snapshots
    guard_status = latest_portfolio_status()  # dict from our helper
    hist_df = load_portfolio_history()  # timestamp,equity snapshots

    # --- Top row: live protection status ---
    # These are the "are we in DEFENSIVE or LOCKDOWN right now?" talking points.
    g_mode = guard_status.get("mode", "UNKNOWN")
    g_reason = guard_status.get("reason", "")
    g_draw = guard_status.get("drawdown_pct", np.nan)
    g_bp = guard_status.get("buying_power", np.nan)
    g_equity = guard_status.get("latest_equity", np.nan)
    g_reserve_pct = guard_status.get("reserve_pct", np.nan)

    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.metric(
            "Guard Mode",
            f"{g_mode} ({g_reserve_pct:.0%} reserve)" if np.isfinite(g_reserve_pct) else g_mode,
            help=g_reason or "Capital Preservation Doctrine state.",
        )
    with cB:
        st.metric(
            "Current Drawdown",
            f"{g_draw:.1%}" if np.isfinite(g_draw) else "—",
            help="Live drawdown vs prior peak from portfolio_health.",
        )
    with cC:
        st.metric(
            "Latest Equity",
            f"${g_equity:,.2f}" if np.isfinite(g_equity) else "—",
            help="Most recent known portfolio equity.",
        )
    with cD:
        st.metric(
            "Buying Power",
            f"${g_bp:,.0f}" if np.isfinite(g_bp) else "—",
            help="Broker-reported available capital.",
        )

    st.markdown("---")

    # --- Historical drawdown curve from portfolio_history.csv snapshots ---
    # portfolio_history.csv currently: timestamp,equity (appended every run in place_orders_from_csv.py)
    if hist_df.empty:
        st.info("No historical equity snapshots yet. We'll start logging as trades execute.")
    else:
        df = hist_df.copy()

        # Normalize schema -> 'date' + numeric equity
        # We want monotonic time, 1 row per timestamp
        if "timestamp" in df.columns:
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            df = df.rename(columns={"timestamp": "date"})
        # fallback if somehow already called 'date'
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)

        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
        df = df.dropna(subset=["date", "equity"]).copy()

        # If multiple samples land on same day (or same minute), we just keep
        # them all; but for drawdown math we want strictly forward cumulative peak
        df = df.sort_values("date")
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = df["equity"] / df["peak"] - 1.0

        # Metrics from curve
        max_dd = float(df["drawdown"].min()) if not df.empty else np.nan
        curr_dd = float(df["drawdown"].iloc[-1]) if not df.empty else np.nan
        obs_cnt = len(df)

        # Peak→trough window of worst drawdown
        # We'll locate the row with worst drawdown, then find the preceding peak.
        peak_date_str = "—"
        trough_date_str = "—"
        dd_days_str = "—"
        try:
            trough_idx = int(df["drawdown"].idxmin())
            trough_row = df.loc[trough_idx]
            trough_date = trough_row["date"]

            # up to trough_idx, find highest equity (the "peak")
            sub = df.loc[:trough_idx]
            peak_idx = int(sub["equity"].idxmax())
            peak_row = df.loc[peak_idx]
            peak_date = peak_row["date"]

            peak_date_str = peak_date.strftime("%Y-%m-%d %H:%M:%S")
            trough_date_str = trough_date.strftime("%Y-%m-%d %H:%M:%S")

            if pd.notna(peak_date) and pd.notna(trough_date):
                dd_days_val = (trough_date - peak_date).days
                dd_days_str = str(int(dd_days_val))
        except Exception:
            pass

        # KPIs for the historical curve
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                "Max Drawdown (hist)",
                f"{max_dd:.1%}" if np.isfinite(max_dd) else "—",
            )
        with k2:
            st.metric(
                "Current Drawdown (hist)",
                f"{curr_dd:.1%}" if np.isfinite(curr_dd) else "—",
            )
        with k3:
            st.metric(
                "Peak → Trough (days)",
                dd_days_str,
                help=f"Peak: {peak_date_str}\nTrough: {trough_date_str}",
            )
        with k4:
            st.metric("Snapshots Logged", obs_cnt)

        # Drawdown area chart
        if not PLOTLY_OK:
            st.warning("Plotly not installed — `pip install plotly` for charts.")
            # fallback: line of drawdown
            st.line_chart(
                df.set_index("date")[["drawdown"]].rename(columns={"drawdown": "drawdown (neg)"})
            )
        else:
            fig = px.area(
                df,
                x="date",
                y="drawdown",
                title="Historical Drawdown (vs running peak)",
            )
            fig.update_traces(
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Drawdown: %{y:.1%}<extra></extra>"
            )

            # clamp top at 0%, bottom at at least -90% (or actual min if worse)
            y_min = (
                float(min(-0.9, df["drawdown"].min()))
                if np.isfinite(df["drawdown"].min())
                else -1.0
            )
            fig.update_yaxes(tickformat=".0%", range=[y_min, 0])
            fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.5)

            st.plotly_chart(fig, use_container_width=True)

        # Export
        out_cols = ["date", "equity", "peak", "drawdown"]
        st.download_button(
            "⬇️ Download drawdown series (CSV)",
            data=df[out_cols].to_csv(index=False).encode("utf-8"),
            file_name="portfolio_drawdown.csv",
            mime="text/csv",
            key="t7_dl",
        )

# ──────────────────────────────
# Tab 8 — Strategy Diagnostics (fixed date slider)
# ──────────────────────────────
with tabs[8]:
    st.subheader("📊 Strategy Diagnostics")

    df = load_csv("trade_log.csv", RESULTS_DIR)
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        # Light cleanup
        parse_dates_inplace(df, ("date",), normalize=False)
        for col in ["profit", "price", "entry_price", "exit_price", "quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Choose label column
        label_col = next((c for c in ["signal", "action", "side"] if c in df.columns), None)

        # Filters
        c1, c2 = st.columns([1.4, 1.2])
        with c1:
            if label_col:
                labels = sorted(df[label_col].dropna().astype(str).unique().tolist())
                sel_labels = st.multiselect(
                    f"Filter {label_col}",
                    options=labels,
                    default=labels,
                    key="t8_labels",
                )
            else:
                sel_labels = None

        # Build a safe datetime slider (Streamlit expects datetime.datetime, not pandas.Timestamp)
        with c2:
            if "date" in df.columns and df["date"].notna().any():
                dcol = pd.to_datetime(df["date"], errors="coerce")
                dvalid = dcol.dropna()
                if not dvalid.empty:
                    dmin = dvalid.min().to_pydatetime()
                    dmax = dvalid.max().to_pydatetime()
                    # If min == max, expand by 1 day for a usable range
                    if dmin == dmax:
                        dmax = (pd.Timestamp(dmax) + pd.Timedelta(days=1)).to_pydatetime()
                    date_range = st.slider(
                        "Date range",
                        value=(dmin, dmax),
                        min_value=dmin,
                        max_value=dmax,
                        format="YYYY-MM-DD",
                        key="t8_daterange",
                    )
                else:
                    date_range = None
            else:
                date_range = None

        # Apply filters
f = df.copy()
f = sanitize_df_cols(f)  # <- make columns unique first

if label_col and sel_labels and label_col in f.columns:
    f = f[f[label_col].astype(str).isin(sel_labels)]

if date_range and "date" in f.columns:
    d0 = pd.Timestamp(date_range[0])
    d1 = pd.Timestamp(date_range[1])
    # ensure datetime for robust filtering
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f[(f["date"] >= d0) & (f["date"] <= d1)]

if f.empty:
    st.info("No rows after filtering.")
else:
    # KPIs
    trades = len(f)
    prof_series = f["profit"] if "profit" in f.columns else pd.Series(dtype=float)
    wins = int((prof_series > 0).sum()) if "profit" in f.columns else 0
    total_pnl = float(prof_series.sum()) if "profit" in f.columns else np.nan
    avg_pnl = float(prof_series.mean()) if "profit" in f.columns else np.nan
    med_pnl = float(prof_series.median()) if "profit" in f.columns else np.nan
    win_rate = (wins / trades) if trades and "profit" in f.columns else np.nan

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Trades", trades)
    with k2:
        st.metric("Win rate", f"{win_rate:.0%}" if np.isfinite(win_rate) else "—")
    with k3:
        st.metric("Total P&L", f"{total_pnl:,.2f}" if np.isfinite(total_pnl) else "—")
    with k4:
        st.metric(
            "Avg / Med P&L",
            (
                f"{avg_pnl:,.2f} / {med_pnl:,.2f}"
                if np.isfinite(avg_pnl) and np.isfinite(med_pnl)
                else "—"
            ),
        )

    # Charts
    if PLOTLY_OK:
        if label_col and label_col in f.columns:
            counts = f[label_col].astype(str).value_counts()
            fig1 = px.bar(
                x=counts.index,
                y=counts.values,
                labels={"x": label_col.capitalize(), "y": "Count"},
                title=f"{label_col.capitalize()} Distribution",
            )
            st.plotly_chart(fig1, use_container_width=True)

        if "profit" in f.columns and f["profit"].notna().any():
            fig2 = px.histogram(f, x="profit", nbins=40, title="P&L per Trade — Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        if "profit" in f.columns and "date" in f.columns and f["date"].notna().any():
            tmp = f.dropna(subset=["date", "profit"]).sort_values("date").copy()
            fig3 = px.line(
                tmp.assign(cum_pnl=lambda d: d["profit"].cumsum()),
                x="date",
                y="cum_pnl",
                title="Cumulative P&L",
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.caption("Plotly not installed — install with `pip install plotly` to see charts.")

    # Table (last 200 for readability)
    candidate_cols = [
        "date",
        "ticker",
        label_col if label_col else None,
        "action",
        "side",
        "price",
        "quantity",
        "profit",
    ]
    # De-dup and keep only existing columns (preserves order)
    show_cols = [c for c in dict.fromkeys(candidate_cols) if c and c in f.columns]

    view = f.loc[:, show_cols].tail(200) if show_cols else f.tail(200)
    st.dataframe(view, use_container_width=True)

    # Download filtered
    st.download_button(
        "⬇️ Download filtered trades (CSV)",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="trade_log_filtered.csv",
        mime="text/csv",
        key="t8_dl",
    )

# ──────────────────────────────
# Tab 9 — Portfolio Allocations (robust, no st.stop)
# ──────────────────────────────
with tabs[9]:
    st.subheader("🏦 Portfolio Allocations")

    df = load_csv("trade_log.csv", RESULTS_DIR)
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        # Required columns
        needed = {"action", "quantity", "ticker"}
        missing = sorted(needed - set(df.columns))
        if missing:
            st.warning(f"Missing columns in trade_log.csv: {missing}")
        else:
            df = df.copy()
            df["ticker"] = df["ticker"].astype(str)
            df["action_up"] = df["action"].astype(str).str.upper()
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)

            # Optional date filter
            parse_dates_inplace(df, ("date",))
            if "date" in df.columns and df["date"].notna().any():
                dmin = pd.to_datetime(df["date"], errors="coerce").min()
                dmax = pd.to_datetime(df["date"], errors="coerce").max()
                c1, c2 = st.columns([1.2, 2])
                with c1:
                    use_range = st.checkbox("Filter by date range", value=False, key="t9_use_range")
                with c2:
                    if use_range and pd.notna(dmin) and pd.notna(dmax):
                        date_range = st.slider(
                            "Date range",
                            min_value=dmin.to_pydatetime(),
                            max_value=dmax.to_pydatetime(),
                            value=(dmin.to_pydatetime(), dmax.to_pydatetime()),
                            format="YYYY-MM-DD",
                            key="t9_date_range",
                        )
                        start_dt, end_dt = pd.to_datetime(list(date_range))
                        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            # Signed shares
            def _signed_q(row):
                a = row["action_up"]
                q = row["quantity"]
                if "BUY" in a or "COVER" in a:
                    return q
                if "SELL" in a or "SHORT" in a:
                    return -q
                return 0.0

            df["signed_qty"] = df.apply(_signed_q, axis=1)

            pos = df.groupby("ticker", dropna=True)["signed_qty"].sum().sort_values(ascending=False)

            if pos.empty or pos.abs().sum() == 0:
                st.info("No net positions to visualize yet.")
            else:
                # Controls
                cL, cR = st.columns([1.4, 1])
                with cL:
                    include_shorts = st.checkbox(
                        "Include short positions", value=True, key="t9_inc_shorts"
                    )
                with cR:
                    top_n = st.slider(
                        "Show top N in chart (rest → Other)", 3, 25, 10, 1, key="t9_topn"
                    )

                longs = pos[pos > 0]
                shorts = -pos[pos < 0]  # positive for display

                def _prep_series(s: pd.Series, N: int) -> pd.Series:
                    s = s.sort_values(ascending=False)
                    if len(s) <= N:
                        return s
                    head = s.iloc[:N].copy()
                    tail_sum = s.iloc[N:].sum()
                    if tail_sum > 0:
                        head.loc["Other"] = head.get("Other", 0.0) + tail_sum
                    return head

                if not PLOTLY_OK:
                    st.warning("Plotly not installed — `pip install plotly` for charts.")
                    st.subheader("Long Positions")
                    if not longs.empty:
                        st.dataframe(longs.rename("shares").reset_index(), use_container_width=True)
                    else:
                        st.caption("No longs.")
                    if include_shorts:
                        st.subheader("Short Positions (absolute shares)")
                        if not shorts.empty:
                            st.dataframe(
                                shorts.rename("shares").reset_index(), use_container_width=True
                            )
                        else:
                            st.caption("No shorts.")
                else:
                    if not longs.empty:
                        st.subheader("📈 Longs Allocation")
                        longs_top = _prep_series(longs, top_n)
                        fig_long = px.pie(
                            values=longs_top.values,
                            names=longs_top.index.astype(str),
                            title="Holdings Allocation — Longs",
                        )
                        st.plotly_chart(fig_long, use_container_width=True)
                    else:
                        st.caption("No long positions.")

                    if include_shorts:
                        if not shorts.empty:
                            st.subheader("📉 Shorts Allocation")
                            shorts_top = _prep_series(shorts, top_n)
                            fig_short = px.pie(
                                values=shorts_top.values,
                                names=shorts_top.index.astype(str),
                                title="Holdings Allocation — Shorts (absolute shares)",
                            )
                            st.plotly_chart(fig_short, use_container_width=True)
                        else:
                            st.caption("No short positions (or shorts excluded).")

                # Tables + KPIs
                with st.expander("🔎 Position tables"):
                    c1, c2, c3 = st.columns([1.2, 1.2, 1])
                    with c1:
                        st.write("**Longs (shares)**")
                        st.dataframe(
                            longs.rename("shares")
                            .reset_index()
                            .rename(columns={"index": "ticker"}),
                            use_container_width=True,
                        )
                    with c2:
                        st.write("**Shorts (absolute shares)**")
                        st.dataframe(
                            shorts.rename("shares")
                            .reset_index()
                            .rename(columns={"index": "ticker"}),
                            use_container_width=True,
                        )
                    with c3:
                        st.metric("Net tickers", int((pos != 0).sum()))
                        st.metric("Total long shares", int(longs.sum()) if not longs.empty else 0)
                        st.metric(
                            "Total short shares",
                            int(shorts.sum()) if include_shorts and not shorts.empty else 0,
                        )

                # Download
                with st.expander("⬇️ Export"):
                    out = pd.DataFrame(
                        {
                            "ticker": pos.index.astype(str),
                            "net_shares": pos.values,
                            "long_shares": pos.clip(lower=0).values,
                            "short_shares_abs": (-pos.clip(upper=0)).values,
                        }
                    )
                    st.download_button(
                        "Download positions (CSV)",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="portfolio_positions.csv",
                        mime="text/csv",
                        key="t9_dl_positions",
                    )

# ──────────────────────────────
# Tab 10 — Trade Replay (clean & robust)
# ──────────────────────────────
with tabs[10]:
    st.subheader("📽️ Trade Replay")
    df = load_csv("trade_log.csv", RESULTS_DIR)

    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        # Normalize schema
        df = ensure_date(df, normalize=False)
        for need in ["ticker", "action", "quantity", "price"]:
            if need not in df.columns:
                df[need] = np.nan
        # Some logs use 'side' or 'fill_price' names
        if df["action"].isna().all() and "side" in df.columns:
            df["action"] = df["side"]
        if df["price"].isna().all():
            for alt in ("fill_price", "avg_fill_price", "entry_price"):
                if alt in df.columns and df[alt].notna().any():
                    df["price"] = df[alt]
                    break

        # Basic cleaning
        df["ticker"] = df["ticker"].astype(str)
        df["action"] = df["action"].astype(str).str.upper()
        to_numeric(df, ["quantity", "price"])
        df = df.dropna(subset=["ticker"]).copy()

        if df.empty or "ticker" not in df.columns:
            st.warning("Missing or empty 'ticker' column.")
        else:
            tickers = sorted(df["ticker"].dropna().unique())
            ticker = st.selectbox("Select ticker", tickers, key="t10_ticker")

            # Subset & sort
            tdf = df[df["ticker"] == ticker].copy()
            tdf = tdf.dropna(subset=["date"]).sort_values("date")
            # Metrics
            buys = tdf[tdf["action"] == "BUY"]
            sells = tdf[tdf["action"] == "SELL"]
            net_shares = buys["quantity"].fillna(0).sum() - sells["quantity"].fillna(0).sum()
            total_trades = len(tdf)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Trades", total_trades)
            with c2:
                st.metric("Net shares", int(net_shares))
            with c3:
                notional = (tdf["price"].fillna(0) * tdf["quantity"].fillna(0)).abs().sum()
                st.metric("Gross notional traded", f"${notional:,.0f}")

            # Date range filter (if we have dates)
            if not tdf.empty:
                dmin, dmax = tdf["date"].min(), tdf["date"].max()
                if pd.notna(dmin) and pd.notna(dmax) and dmin < dmax:
                    date_range = st.slider(
                        "Date range",
                        min_value=dmin.to_pydatetime(),
                        max_value=dmax.to_pydatetime(),
                        value=(dmin.to_pydatetime(), dmax.to_pydatetime()),
                        key="t10_daterange",
                    )
                    if date_range:
                        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                        tdf = tdf[(tdf["date"] >= start) & (tdf["date"] <= end)]

            # Chart (try to overlay on price if available)
            if PLOTLY_OK and not tdf.empty:
                fig = go.Figure()

                # Try OHLC from parquet for richer context
                added_price = False
                try:
                    pq = RESULTS_DIR / f"{ticker}.parquet"
                    ohlc = load_parquet(pq)
                    if not ohlc.empty and {"date", "close"}.issubset(ohlc.columns):
                        parse_dates_inplace(ohlc, ("date",))
                        ohlc = ohlc.dropna(subset=["date", "close"]).sort_values("date")
                        # Respect selected date range
                        if "start" in locals() and "end" in locals():
                            ohlc = ohlc[(ohlc["date"] >= start) & (ohlc["date"] <= end)]
                        if not ohlc.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=ohlc["date"],
                                    y=ohlc["close"],
                                    mode="lines",
                                    name="Close",
                                    opacity=0.55,
                                )
                            )
                            added_price = True
                except Exception:
                    pass

                # Fallback: use trade prices as a line to give scale
                if not added_price:
                    if tdf["price"].notna().any():
                        fig.add_trace(
                            go.Scatter(
                                x=tdf["date"],
                                y=tdf["price"],
                                mode="lines",
                                name="Trade price (trace)",
                                opacity=0.25,
                            )
                        )

                # Trade markers
                for side, g in tdf.groupby("action"):
                    fig.add_trace(
                        go.Scatter(
                            x=g["date"],
                            y=g["price"],
                            mode="markers",
                            name=side,
                            marker=dict(
                                size=np.clip((g["quantity"].fillna(0).abs() ** 0.5) * 2 + 6, 6, 24)
                            ),
                            hovertemplate=(
                                "<b>%{x|%Y-%m-%d}</b><br>"
                                "Action: %{meta[0]}<br>"
                                "Qty: %{meta[1]}<br>"
                                "Price: %{y:.2f}<br>"
                                "Notional: %{meta[2]:,.0f}<extra></extra>"
                            ),
                            meta=np.stack(
                                [
                                    g["action"].astype(str).values,
                                    g["quantity"].fillna(0).values,
                                    (g["price"].fillna(0) * g["quantity"].fillna(0)).values,
                                ],
                                axis=-1,
                            ),
                        )
                    )

                fig.update_layout(
                    title=f"{ticker} — Trade Replay",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            elif not PLOTLY_OK:
                st.warning("Plotly not installed — `pip install plotly` for charts.")

            # Table (nice defaults)
            cols = [
                c
                for c in ["date", "action", "price", "quantity", "order_id", "note"]
                if c in tdf.columns
            ]
            st.dataframe(tdf[cols] if cols else tdf, use_container_width=True)

            # Download filtered
            st.download_button(
                "⬇️ Download filtered trades (CSV)",
                data=tdf.to_csv(index=False).encode("utf-8"),
                file_name=f"trade_replay_{ticker}.csv",
                mime="text/csv",
                key="t10_dl",
            )
# ──────────────────────────────
# Tab 11 — Fundamentals
# ──────────────────────────────
with tabs[11]:
    st.subheader("📘 Fundamental Data")

    df = load_csv("fundamentals.csv", RESULTS_DIR)
    if df.empty:
        st.info("No fundamentals.csv yet.")
    else:
        # Best-effort date parsing if present
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(
                    None
                )
            except Exception:
                pass

        # Standardize some common column names (case-insensitive)
        def _col(df_cols, *cands):
            s = {c.lower(): c for c in df_cols}
            for c in cands:
                if c.lower() in s:
                    return s[c.lower()]
            return None

        col_ticker = _col(df.columns, "ticker", "symbol")
        col_sector = _col(df.columns, "sector")
        col_industry = _col(df.columns, "industry", "subindustry")
        col_mktcap = _col(df.columns, "market_cap", "marketcap", "market_capitalization")
        col_pe = _col(df.columns, "pe_ratio", "pe", "price_to_earnings")
        col_divy = _col(df.columns, "dividend_yield", "div_yield", "dividendYield")

        # Auto-detect numeric columns
        numeric_cols = []
        for c in df.columns:
            if df[c].dtype.kind in "biufc":
                numeric_cols.append(c)
            else:
                # try coercion to see if it's numeric-like
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().sum() > 0 and coerced.notna().sum() >= max(
                    5, int(0.2 * len(coerced))
                ):
                    df[c] = coerced
                    numeric_cols.append(c)

        # Filters
        c1, c2, c3 = st.columns([1.4, 1, 1.2])
        with c1:
            tickers = sorted(df[col_ticker].dropna().astype(str).unique()) if col_ticker else []
            sel_tickers = st.multiselect("Tickers", tickers, default=[], key="t11_tickers")
        with c2:
            sectors = sorted(df[col_sector].dropna().astype(str).unique()) if col_sector else []
            sel_sector = st.selectbox("Sector", ["(All)"] + sectors, index=0, key="t11_sector")
        with c3:
            kw = st.text_input("Search (ticker/name)", value="", key="t11_kw")

        f = df.copy()

        if sel_tickers and col_ticker:
            f = f[f[col_ticker].astype(str).isin(sel_tickers)]

        if col_sector and sel_sector != "(All)":
            f = f[f[col_sector].astype(str) == sel_sector]

        if kw.strip():
            kw_l = kw.strip().lower()
            hay = []
            if col_ticker:
                hay.append(f[col_ticker].astype(str).str.lower())
            name_col = _col(f.columns, "name", "company", "company_name")
            if name_col:
                hay.append(f[name_col].astype(str).str.lower())
            if hay:
                mask = False
                for h in hay:
                    mask = mask | h.str.contains(kw_l, na=False)
                f = f[mask]

        # KPIs
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Rows", len(f))
        with k2:
            if col_mktcap and col_ticker:
                mc = pd.to_numeric(f[col_mktcap], errors="coerce")
                st.metric("Total Market Cap", f"{mc.sum():,.0f}")
        with k3:
            if col_pe and col_ticker:
                pe = pd.to_numeric(f[col_pe], errors="coerce")
                st.metric("Median P/E", f"{pe.median():.1f}" if pe.notna().any() else "—")

        # Optional quick view toggles
        show_only = st.multiselect(
            "Columns to show (optional)",
            options=list(f.columns),
            default=[],
            key="t11_cols",
            help="Leave empty to show all columns.",
        )
        view = f[show_only] if show_only else f

        # Friendly sorting: by Market Cap desc if present, else by ticker
        if col_mktcap and col_mktcap in view.columns:
            view = view.sort_values(col_mktcap, ascending=False)
        elif col_ticker and col_ticker in view.columns:
            view = view.sort_values(col_ticker)

        st.dataframe(view, use_container_width=True)

        st.download_button(
            "⬇️ Download filtered fundamentals (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="fundamentals_filtered.csv",
            mime="text/csv",
            key="t11_dl",
        )
# ──────────────────────────────
# Tab 12 — Stock Scores (robust)
# ──────────────────────────────
with tabs[12]:
    st.subheader("📈 Stock Scores")
    df = load_csv("stock_scores.csv", RESULTS_DIR)

    if df.empty:
        st.info("No stock_scores.csv yet.")
    else:
        # Make sure a few expected columns exist so sorting/printing won't error
        for need in ["ticker"]:
            if need not in df.columns:
                df[need] = np.nan

        score_col = get_score_col(df)  # prefers 'total_score', falls back to 'score'
        if score_col:
            to_numeric(df, [score_col])
            df = df.sort_values(score_col, ascending=False)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download scores (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="stock_scores.csv",
            mime="text/csv",
            key="t12_dl",
        )
# ──────────────────────────────
# Tab 13 — Top Picks (configurable)
# ──────────────────────────────
with tabs[13]:
    st.subheader("🎯 Top Fundamental Picks")
    df = load_csv("stock_scores.csv", RESULTS_DIR)

    if df.empty:
        st.info("No stock_scores.csv yet.")
    else:
        score_col = get_score_col(df)
        top_n = st.slider("How many picks?", 5, 50, 10, 1, key="t13_topn")

        if score_col:
            to_numeric(df, [score_col])
            top = df.sort_values(score_col, ascending=False).head(top_n)
            st.dataframe(top, use_container_width=True)
            st.download_button(
                f"⬇️ Download top {top_n} (CSV)",
                data=top.to_csv(index=False).encode("utf-8"),
                file_name=f"top_picks_{top_n}.csv",
                mime="text/csv",
                key="t13_dl",
            )
        else:
            st.warning(
                "No score column found (expected 'total_score' or 'score'). Showing first rows."
            )
            head = df.head(top_n)
            st.dataframe(head, use_container_width=True)
            st.download_button(
                f"⬇️ Download first {top_n} (CSV)",
                data=head.to_csv(index=False).encode("utf-8"),
                file_name=f"top_first_{top_n}.csv",
                mime="text/csv",
                key="t13_dl_fallback",
            )
# ──────────────────────────────
# Tab 14 — News Sentiment (anchored window + trusted filter, fixed)
# ──────────────────────────────
with tabs[14]:
    st.subheader("📰 News Sentiment")

    df = load_csv("news_sentiment.csv", RESULTS_DIR)
    if df.empty:
        st.info("No news_sentiment.csv yet.")
    else:
        # 1) Parse/normalize date robustly
        if "publishedAt" in df.columns and "date" not in df.columns:
            df["date"] = (
                pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )
        else:
            parse_dates_inplace(df, ("date",), normalize=True)

        # 2) Ensure ticker column exists
        if "ticker" not in df.columns:
            df["ticker"] = ""

        # 3) Sentiment to numeric (keep original)
        if "sentiment" in df.columns:
            df["sentiment_num"] = pd.to_numeric(df["sentiment"], errors="coerce")
        else:
            df["sentiment_num"] = np.nan

        # 4) Ensure a URL column; clean description HTML; clickable headline
        if "url" not in df.columns or df["url"].isna().all():
            for alt in ("link", "source_url"):
                if alt in df.columns and df[alt].notna().any():
                    df["url"] = df[alt]
                    break

        if "description" in df.columns:
            if "url" not in df.columns or df["url"].isna().all():
                df["url"] = df["description"].apply(extract_href)
            df["description"] = df["description"].apply(strip_html)

        title_col = "title" if "title" in df.columns else None
        url_col = "url" if "url" in df.columns else None
        if title_col or url_col:
            df["news"] = df.apply(
                lambda r: make_clickable(r.get(title_col, ""), r.get(url_col, "")),
                axis=1,
            )
        else:
            df["news"] = ""

        # 5) Canonicalize source from URL domain
        def _domain(u: str) -> str:
            try:
                from urllib.parse import urlparse

                n = urlparse(str(u)).netloc.lower()
                parts = [q for q in n.split(".") if q not in ("", "www", "m")]
                return ".".join(parts[-2:]) if len(parts) >= 2 else n
            except Exception:
                return ""

        # If domain missing, derive from url (or empty)
        if "domain" not in df.columns:
            base = df["url"] if "url" in df.columns else pd.Series([""] * len(df), index=df.index)
            df["domain"] = base.apply(_domain)

        CANON = {
            "bloomberg.com": "Bloomberg",
            "finance.yahoo.com": "Yahoo Finance",
            "yahoo.com": "Yahoo Finance",
            "reuters.com": "Reuters",
            "wsj.com": "WSJ",
            "ft.com": "Financial Times",
            "marketwatch.com": "MarketWatch",
            "barrons.com": "Barron's",
            "cnbc.com": "CNBC",
            "apnews.com": "AP News",
            "washingtonpost.com": "Washington Post",
            "businessinsider.com": "Business Insider",
            "forbes.com": "Forbes",
            "thestreet.com": "TheStreet",
            "seekingalpha.com": "Seeking Alpha",
            "investing.com": "Investing.com",
            "coindesk.com": "CoinDesk",
        }

        # Build source_display without .replace(scalar, Series)
        src_col = (
            df["source"] if "source" in df.columns else pd.Series([""] * len(df), index=df.index)
        )
        sd = df["domain"].map(CANON)  # Canonical if known
        sd = sd.fillna(src_col)  # else original 'source'
        sd = sd.astype(str)
        sd = sd.mask(sd.str.strip() == "", df["domain"])  # if still empty, use domain
        df["source_display"] = sd

        # Light de-duplication (same ticker+url+title → keep newest date)
        sort_cols = [c for c in ("date",) if c in df.columns]
        df = df.sort_values(sort_cols, ascending=[False] if sort_cols else True)
        dedupe_keys = [c for c in ("ticker", "url", "title") if c in df.columns]
        if dedupe_keys:
            df = df.drop_duplicates(subset=dedupe_keys, keep="first")

        # 6) Controls
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.4])
        with c1:
            tickers = (
                sorted(df["ticker"].dropna().astype(str).unique()) if "ticker" in df.columns else []
            )
            sel_tickers = st.multiselect("Tickers", tickers, default=[], key="t14_tickers")

        with c2:
            use_latest_anchor = st.toggle(
                "Anchor to latest in file",
                value=True,
                help="When on, the lookback is relative to the newest article in this CSV (not today's date).",
            )
            days = st.slider("Last N days", 1, 180, 30, 1, key="t14_days")

        with c3:
            s_vals = df["sentiment_num"].replace([np.inf, -np.inf], np.nan)
            finite = s_vals[np.isfinite(s_vals)]
            smin = float(finite.min()) if not finite.empty else -1.0
            smax = float(finite.max()) if not finite.empty else 1.0
            if smin == smax:
                smin, smax = smin - 1.0, smax + 1.0
            sel_sent = st.slider(
                "Sentiment range", smin, smax, (smin, smax), 0.01, key="t14_srange"
            )

        with c4:
            kw = st.text_input("Keyword (title/desc)", "", key="t14_kw")

        # Source filters
        cS1, cS2 = st.columns([1.2, 1])
        TRUSTED = {
            "Bloomberg",
            "Yahoo Finance",
            "Reuters",
            "WSJ",
            "Financial Times",
            "MarketWatch",
            "Barron's",
            "CNBC",
            "AP News",
            "Washington Post",
            "Business Insider",
            "Forbes",
            "TheStreet",
            "Seeking Alpha",
            "Investing.com",
            "CoinDesk",
        }
        with cS1:
            trusted_only = st.checkbox("Trusted outlets only", value=False, key="t14_trusted")
        with cS2:
            all_sources = sorted(
                x for x in df["source_display"].fillna("").unique() if str(x).strip()
            )
            pick_sources = st.multiselect("Sources", all_sources, default=[], key="t14_sources")

        # 7) Build cutoff date
        anchor = (
            df["date"].max()
            if (use_latest_anchor and "date" in df.columns and df["date"].notna().any())
            else pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        )
        cutoff = anchor - pd.Timedelta(days=days)

        # 8) Filtering (safe order)
        f = df.copy()
        if "date" in f.columns:
            f = f[f["date"].notna() & (f["date"] >= cutoff)]

        if sel_tickers:
            f = f[f["ticker"].astype(str).isin(sel_tickers)]

        if "sentiment_num" in f.columns:
            s = f["sentiment_num"]
            f = f[(s >= sel_sent[0]) & (s <= sel_sent[1])]

        if trusted_only and "source_display" in f.columns:
            f = f[f["source_display"].isin(TRUSTED)]

        if pick_sources:
            f = f[f["source_display"].isin(pick_sources)]

        if kw.strip():
            kw_l = kw.strip().lower()
            hay = []
            if "title" in f.columns:
                hay.append(f["title"].astype(str).str.lower())
            if "description" in f.columns:
                hay.append(f["description"].astype(str).str.lower())
            if hay:
                mask = False
                for h in hay:
                    mask = mask | h.str.contains(re.escape(kw_l), na=False)
                f = f[mask]

        # 9) KPIs (+ empty-state hint)
        if f.empty:
            st.warning(
                "No rows match the current filters. Tip: turn off 'Trusted outlets only', clear Sources, or widen the date window."
            )
            st.stop()

        cL, cM, cR, cA = st.columns(4)
        with cL:
            st.metric("Rows", int(len(f)))
        with cM:
            st.metric("Unique tickers", int(f["ticker"].nunique()) if "ticker" in f.columns else 0)
        with cR:
            if "sentiment_num" in f.columns and f["sentiment_num"].notna().any():
                st.metric("Avg Sentiment", f"{f['sentiment_num'].mean():.2f}")
            else:
                st.metric("Avg Sentiment", "—")
        with cA:
            if "date" in df.columns and df["date"].notna().any():
                st.caption(f"Anchor date: {anchor:%Y-%m-%d}")

        # 10) Display (newest first; keep links clickable)
        show_cols_pref = [
            "date",
            "ticker",
            "sentiment",
            "source_display",
            "news",
            "description",
            "domain",
            "author",
        ]
        show_cols = [c for c in show_cols_pref if (c in f.columns or c == "news")]
        disp = f[show_cols] if show_cols else f
        if "date" in disp.columns:
            disp = disp.sort_values("date", ascending=False)
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        # 11) Download filtered
        st.download_button(
            "⬇️ Download filtered (CSV)",
            data=(disp.to_csv(index=False).encode("utf-8")),
            file_name="news_sentiment_filtered.csv",
            mime="text/csv",
            key="t14_dl",
        )

# ──────────────────────────────
# Tab 15 — Smart Alerts (robust, fixed categorical fillna)
# ──────────────────────────────
with tabs[15]:
    st.subheader("🚨 Smart Alerts")

    # Load both possible filenames
    df = load_csv("alerts.csv", RESULTS_DIR)
    if df.empty:
        df = load_csv("smart_alerts.csv", RESULTS_DIR)

    if df.empty:
        st.info("No alerts CSV found.")
    else:
        # Ensure expected columns exist to prevent KeyErrors downstream
        for col in [
            "date",
            "timestamp",
            "priority",
            "ticker",
            "title",
            "url",
            "message",
            "type",
            "score",
        ]:
            if col not in df.columns:
                df[col] = np.nan

        # Dates
        parse_dates_inplace(df, ("date", "timestamp"))
        if "date" in df.columns:
            df["date"] = (
                pd.to_datetime(df["date"], errors="coerce", utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )

        # Priority as ordered category (LOW < MEDIUM < HIGH)
        pri_order = ["LOW", "MEDIUM", "HIGH"]
        # Clean priority strings then cast to ordered categorical
        df["priority"] = (
            df["priority"].astype(str).str.upper().where(df["priority"].notna(), np.nan)
        )
        df["priority"] = pd.Categorical(df["priority"], categories=pri_order, ordered=True)

        # UI controls
        col_l, col_r = st.columns([3, 2])
        with col_l:
            min_pri = st.selectbox(
                "Minimum priority",
                options=pri_order,
                index=1,  # default MEDIUM
                key="t15_minpri",
            )
            tickers = (
                sorted(df["ticker"].dropna().astype(str).unique()) if "ticker" in df.columns else []
            )
            sel_tickers = st.multiselect("Tickers", tickers, default=[], key="t15_tickers")
        with col_r:
            days_back = st.slider("Show last N days", 3, 60, 30, 1, key="t15_days")

        # Filtering
        f = df.copy()

        # Priority filter — map AFTER converting off categorical to avoid fillna-on-categorical error
        pri_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        pri_vals = f["priority"].astype(object).map(pri_rank)  # -> numeric Series, not categorical
        pri_vals = pd.to_numeric(pri_vals, errors="coerce").fillna(-1)
        f = f[pri_vals >= pri_rank[min_pri]]

        # Ticker filter
        if sel_tickers:
            f = f[f["ticker"].astype(str).isin(sel_tickers)]

        # Date filter
        if "date" in f.columns:
            cutoff = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(
                days=days_back
            )
            f = f[pd.to_datetime(f["date"], errors="coerce") >= cutoff]

        # Sort: priority desc, score desc, date desc (only those that exist)
        # Use a numeric priority key for stable sorting
        f["_pri_key"] = f["priority"].astype(object).map(pri_rank)
        to_numeric(f, ["_pri_key", "score"])
        sort_cols, ascending = [], []
        if "_pri_key" in f.columns:
            sort_cols.append("_pri_key")
            ascending.append(False)
        if "score" in f.columns:
            sort_cols.append("score")
            ascending.append(False)
        if "date" in f.columns:
            sort_cols.append("date")
            ascending.append(False)
        if sort_cols:
            f = f.sort_values(sort_cols, ascending=ascending)

        # Clickable title
        f["news"] = f.apply(lambda r: make_clickable(r.get("title", ""), r.get("url", "")), axis=1)

        # Display table
        show_cols = [
            c
            for c in ["date", "ticker", "type", "priority", "score", "news", "message"]
            if c in f.columns or c == "news"
        ]
        disp = f[show_cols] if show_cols else f
        if "date" in disp.columns:
            disp = disp.sort_values("date", ascending=False)
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        # KPIs
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Alerts shown", int(len(f)))
        with c2:
            st.metric(
                "HIGH priority",
                int((f["priority"] == "HIGH").sum()) if "priority" in f.columns else 0,
            )
        with c3:
            st.metric("Unique tickers", int(f["ticker"].nunique()) if "ticker" in f.columns else 0)

        # Download
        dl_df = f.drop(columns=["_pri_key"], errors="ignore").copy()
        if "date" in dl_df.columns:
            dl_df = dl_df.sort_values("date", ascending=False)
        st.download_button(
            "⬇️ Download filtered alerts (CSV)",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name="alerts_filtered.csv",
            mime="text/csv",
            key="t15_dl",
        )
# ──────────────────────────────
# Tab 16 — Economic Calendar (filters + download)
# ──────────────────────────────
with tabs[16]:
    st.subheader("📆 Economic Calendar")
    df = load_csv("economic_calendar.csv", RESULTS_DIR)

    if df.empty:
        st.info("No economic_calendar.csv yet.")
    else:
        # Make sure expected columns exist to avoid KeyErrors
        for c in ["date", "country", "event", "actual", "forecast", "previous", "impact", "source"]:
            if c not in df.columns:
                df[c] = np.nan

        # Parse dates (tz-naive) and sort
        parse_dates_inplace(df, ("date",))
        df = df.dropna(subset=["date"]).sort_values("date")

        # Sidebar-ish filters on the row
        c1, c2, c3 = st.columns([1.4, 1, 1.2])
        with c1:
            # Date range
            dmin = df["date"].min()
            dmax = df["date"].max()
            date_range = st.date_input(
                "Date range",
                value=(dmin.date(), dmax.date()) if pd.notna(dmin) and pd.notna(dmax) else None,
                key="t16_daterange",
            )
        with c2:
            countries = sorted(df["country"].dropna().astype(str).unique())
            sel_countries = st.multiselect("Country", countries, key="t16_countries")
        with c3:
            # Common impact scales: LOW/MEDIUM/HIGH or 1/2/3, handle both
            impact_vals = sorted(df["impact"].dropna().astype(str).unique())
            sel_impacts = st.multiselect("Impact", impact_vals, key="t16_impacts")

        # Text filter row
        c4, c5 = st.columns([1.6, 1])
        with c4:
            kw = st.text_input("Keyword (event/source)", "", key="t16_kw")
        with c5:
            show_cols_default = [
                "date",
                "country",
                "event",
                "impact",
                "actual",
                "forecast",
                "previous",
                "source",
            ]
            show_cols = st.multiselect(
                "Columns",
                options=list(df.columns),
                default=[c for c in show_cols_default if c in df.columns],
                key="t16_cols",
            )

        # Apply filters
        f = df.copy()

        # Date filter
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            d0 = pd.to_datetime(date_range[0])
            d1 = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # inclusive end
            f = f[(f["date"] >= d0) & (f["date"] < d1)]

        # Countries
        if sel_countries:
            f = f[f["country"].astype(str).isin(sel_countries)]

        # Impact
        if sel_impacts:
            f = f[f["impact"].astype(str).isin(sel_impacts)]

        # Keyword
        if kw.strip():
            needle = kw.strip().lower()
            hay = []
            if "event" in f.columns:
                hay.append(f["event"].astype(str).str.lower())
            if "source" in f.columns:
                hay.append(f["source"].astype(str).str.lower())
            if hay:
                mask = False
                for h in hay:
                    mask = mask | h.str.contains(re.escape(needle), na=False)
                f = f[mask]

        # KPIs
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Rows", len(f))
        with k2:
            st.metric("Unique events", f["event"].nunique() if "event" in f.columns else 0)

        if f.empty:
            st.info("No rows match the current filters.")
        else:
            st.dataframe(f[show_cols] if show_cols else f, use_container_width=True)

        st.download_button(
            "⬇️ Download filtered (CSV)",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="economic_calendar_filtered.csv",
            mime="text/csv",
            key="t16_dl",
        )
# ──────────────────────────────
# Tab 17 — Feature Importance (top-N + chart + download)
# ──────────────────────────────
with tabs[17]:
    st.subheader("🔬 Feature Importance")
    df = load_csv("feature_importance.csv", RESULTS_DIR)

    if df.empty:
        st.info("No feature_importance.csv yet.")
    else:
        # Ensure columns exist
        for c in ["ticker", "feature", "importance"]:
            if c not in df.columns:
                df[c] = np.nan

        to_numeric(df, ["importance"])
        df["ticker"] = df["ticker"].astype(str)

        # Optional model dimension (if present)
        has_model = "model" in df.columns
        if has_model:
            df["model"] = df["model"].astype(str)

        # UI controls
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            tickers = sorted(df["ticker"].dropna().unique())
            sel_ticker = st.selectbox("Select a ticker", tickers, key="t17_ticker")
        with c2:
            sel_model = None
            if has_model:
                models = sorted(df.loc[df["ticker"] == sel_ticker, "model"].dropna().unique())
                sel_model = st.selectbox(
                    "Model (if available)", ["(all)"] + models, key="t17_model"
                )
        with c3:
            top_n = st.slider("Top N features", 5, 100, 20, 1, key="t17_topn")

        # Filter
        sub = df[df["ticker"] == sel_ticker]
        if has_model and sel_model and sel_model != "(all)":
            sub = sub[sub["model"] == sel_model]

        sub = sub.dropna(subset=["feature"]).sort_values("importance", ascending=False)
        top = sub.head(top_n)

        # Plot
        if PLOTLY_OK and not top.empty:
            fig = px.bar(
                top,
                x="feature",
                y="importance",
                title=f"Feature Importance: {sel_ticker}"
                + (f" — {sel_model}" if has_model and sel_model and sel_model != "(all)" else ""),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(top, use_container_width=True)

        # Download filtered
        fname_suffix = f"{sel_ticker}" + (
            f"_{sel_model}" if has_model and sel_model and sel_model != "(all)" else ""
        )
        st.download_button(
            "⬇️ Download selection (CSV)",
            data=top.to_csv(index=False).encode("utf-8"),
            file_name=f"feature_importance_{fname_suffix or 'all'}.csv",
            mime="text/csv",
            key="t17_dl",
        )

# Tab 18 — SL/TP Performance
with tabs[18]:
    st.subheader("🎯 SL/TP Performance Analysis")
    df = load_csv("trade_log.csv", RESULTS_DIR)
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        for c in ["profit", "stop_loss", "take_profit", "exit_price", "entry_price"]:
            if c not in df.columns:
                df[c] = np.nan
        to_numeric(df, ["profit", "stop_loss", "take_profit", "exit_price", "entry_price"])
        df = df[df["profit"].between(-1e12, 1e12)]  # clamp absurd values
        st.metric("Total Trades", len(df))
        if "profit" in df.columns:
            tp_trades = df[df["profit"] > 0]
            sl_trades = df[df["profit"] <= 0]
            st.metric(
                "Avg Profit (TP)",
                round(tp_trades["profit"].mean(), 2) if not tp_trades.empty else 0.0,
            )
            st.metric(
                "Avg Loss (SL)",
                round(sl_trades["profit"].mean(), 2) if not sl_trades.empty else 0.0,
            )
# ──────────────────────────────
# Tab 19 — Sentiment + Signal Fusion (robust + trusted sources)
# ──────────────────────────────
with tabs[19]:
    st.subheader("💬 Sentiment + Signal Fusion")

    # ---------- Load ----------
    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig.empty:
        sig = load_csv("signals.csv", RESULTS_DIR)
    sns = load_csv("news_sentiment.csv", RESULTS_DIR)

    if sig.empty or sns.empty:
        st.info("Need both signals_with_rationale.csv (or signals.csv) and news_sentiment.csv.")
        st.stop()

    # ---------- Signals: clean ----------
    sig = ensure_date(
        sig,
        candidates=["date", "as_of", "timestamp", "time", "datetime", "Date"],
        normalize=True,
    )
    if sig["date"].isna().all():
        sig["date"] = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

    for c in ["ticker", "signal", "rationale"]:
        if c not in sig.columns:
            sig[c] = np.nan

    for c in ["close", "predicted_close", "confidence"]:
        if c not in sig.columns:
            sig[c] = np.nan
    to_numeric(sig, ["close", "predicted_close", "confidence"])

    # If close is missing, try to backfill from {ticker}.parquet
    sig = backfill_close_from_parquet(sig)

    with np.errstate(divide="ignore", invalid="ignore"):
        sig["delta_pct"] = np.where(
            sig["close"].notna() & sig["predicted_close"].notna(),
            (sig["predicted_close"] - sig["close"]) / sig["close"],
            np.nan,
        )

    # ---------- News: clean ----------
    # Normalize date column (naive UTC midnight)
    if "date" not in sns.columns and "publishedAt" in sns.columns:
        sns["date"] = (
            pd.to_datetime(sns["publishedAt"], errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
    else:
        sns["date"] = (
            pd.to_datetime(sns.get("date"), errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )

    # Ensure URL column exists (try alternates if missing)
    if "url" not in sns.columns or sns["url"].isna().all():
        for alt in ["link", "source_url"]:
            if alt in sns.columns and sns[alt].notna().any():
                sns["url"] = sns[alt]
                break

    # Clean HTML in description and extract URL if still missing
    if "description" in sns.columns:
        if "url" not in sns.columns or sns["url"].isna().all():
            sns["url"] = sns["description"].apply(extract_href)
        sns["description"] = sns["description"].apply(strip_html)

    # Sentiment numeric
    if "sentiment" in sns.columns:
        sns["sentiment"] = pd.to_numeric(sns["sentiment"], errors="coerce")

    # Derive domain + canonical source names
    from urllib.parse import urlparse

    def _domain(u: str) -> str:
        try:
            n = urlparse(str(u)).netloc.lower()
            parts = [q for q in n.split(".") if q not in ("", "www", "m")]
            return ".".join(parts[-2:]) if len(parts) >= 2 else n
        except Exception:
            return ""

    CANON = {
        "bloomberg.com": "Bloomberg",
        "finance.yahoo.com": "Yahoo Finance",
        "yahoo.com": "Yahoo Finance",
        "marketwatch.com": "MarketWatch",
        "wsj.com": "WSJ",
        "washingtonpost.com": "Washington Post",
        "reuters.com": "Reuters",
        "apnews.com": "AP News",
        "cnbc.com": "CNBC",
        "ft.com": "Financial Times",
        "seekingalpha.com": "Seeking Alpha",
        "investing.com": "Investing.com",
        "barrons.com": "Barron's",
        "forbes.com": "Forbes",
        "thestreet.com": "TheStreet",
        "fool.com": "Motley Fool",
        "businessinsider.com": "Business Insider",
        "coindesk.com": "CoinDesk",
        "cointelegraph.com": "Cointelegraph",
        "globenewswire.com": "GlobeNewswire",
        "prnewswire.com": "PR Newswire",
        "benzinga.com": "Benzinga",
        "semafor.com": "Semafor",
    }
    TRUSTED = set(CANON.values())

    sns["domain"] = sns.get("url", "").apply(_domain)
    # Prefer canonical name, else fall back to existing 'source'
    sns["source_display"] = sns["domain"].map(CANON).fillna(sns.get("source", ""))

    # Clickable headline
    if {"title", "url"}.issubset(sns.columns):
        sns["news"] = sns.apply(
            lambda r: make_clickable(r.get("title", ""), r.get("url", "")),
            axis=1,
        )
    else:
        sns["news"] = ""

    # ---------- UI controls ----------
    c1, c2, c3 = st.columns([1.4, 1.1, 1.2])
    with c1:
        tickers = sorted(sig.get("ticker", pd.Series(dtype=str)).dropna().astype(str).unique())
        sel_tickers = st.multiselect("Tickers", tickers, default=[], key="t19_tickers")
    with c2:
        days = st.slider("Last N days", 1, 90, 30, 1, key="t19_days")
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.00, 0.01, key="t19_minconf")
        trusted_only = st.checkbox("Trusted outlets only", value=False, key="t19_trusted")
    with c3:
        # Source multiselect populated from canonical names
        all_sources = sorted([s for s in sns["source_display"].dropna().astype(str).unique() if s])
        sel_sources = st.multiselect("Sources", all_sources, default=[], key="t19_sources")
        if "date" in sns.columns and sns["date"].notna().any():
            st.caption(f"Anchor date: {sns['date'].max().date()}")

    cutoff = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=days)

    # ---------- Filter signals ----------
    sig_f = sig.copy()
    if sel_tickers:
        sig_f = sig_f[sig_f["ticker"].astype(str).isin(sel_tickers)]
    if "confidence" in sig_f.columns:
        sig_f = sig_f[sig_f["confidence"].fillna(0) >= min_conf]
    if "date" in sig_f.columns:
        sig_f = sig_f[sig_f["date"] >= cutoff]

    # ---------- Filter news ----------
    sns_f = sns.copy()
    if "date" in sns_f.columns:
        sns_f = sns_f[sns_f["date"].notna() & (sns_f["date"] >= cutoff)]
    if trusted_only:
        sns_f = sns_f[sns_f["source_display"].isin(TRUSTED)]
    if sel_sources:
        sns_f = sns_f[sns_f["source_display"].isin(sel_sources)]
    if sel_tickers and "ticker" in sns_f.columns:
        sns_f = sns_f[sns_f["ticker"].astype(str).isin(sel_tickers)]

    # One article per ticker-day: strongest |sentiment| first, then latest
    if {"ticker", "date"}.issubset(sns_f.columns):
        if "sentiment" in sns_f.columns:
            sns_f = sns_f.sort_values(
                ["ticker", "date", "sentiment"], ascending=[True, True, False]
            ).drop_duplicates(subset=["ticker", "date"], keep="first")
        else:
            sns_f = sns_f.sort_values(["ticker", "date"]).drop_duplicates(
                subset=["ticker", "date"], keep="last"
            )

    # ---------- Merge strategy ----------
    can_join_on_date = sig_f["date"].notna().any() and sns_f["date"].notna().any()
    if can_join_on_date:
        if "ticker" in sns_f.columns and sns_f["ticker"].notna().any():
            merged = pd.merge(
                sig_f, sns_f, on=["ticker", "date"], how="left", suffixes=("", "_news")
            )
        else:
            sns_one = sns_f.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            merged = pd.merge(
                sig_f,
                sns_one.drop(columns=[c for c in ["ticker"] if c in sns_one.columns]),
                on=["date"],
                how="left",
                suffixes=("", "_news"),
            )
    else:
        if "ticker" in sns_f.columns and sns_f["ticker"].notna().any():
            latest_news = sns_f.sort_values("date").groupby("ticker").tail(1)
            merged = pd.merge(sig_f, latest_news, on="ticker", how="left", suffixes=("", "_news"))
        else:
            latest_row = sns_f.sort_values("date").tail(1)
            merged = sig_f.copy()
            if not latest_row.empty:
                for c in latest_row.columns:
                    merged[c] = latest_row.iloc[0][c]

    # ---------- Display (robust to missing 'news') ----------
    # Ensure 'news' exists so slicing never throws
    if "news" not in merged.columns:
        merged["news"] = ""

    # Preferred display columns
    preferred = [
        "date",
        "ticker",
        "news",
        "close",
        "predicted_close",
        "delta_pct",
        "signal",
        "confidence",
        "rationale",
        "sentiment",
        "source_display",
        "domain",
        "author",
        "description",
    ]
    cols_existing = [c for c in preferred if c in merged.columns]

    # Sort newest first if date exists
    if "date" in merged.columns:
        merged = merged.sort_values("date", ascending=False)

    st.markdown(merged[cols_existing].to_html(escape=False, index=False), unsafe_allow_html=True)

    # ---------- KPIs ----------
    cK1, cK2, cK3 = st.columns(3)
    with cK1:
        st.metric("Signals shown", int(len(merged)))
    with cK2:
        if "sentiment" in merged.columns:
            st.metric(
                "Avg Sentiment (news rows)",
                f"{pd.to_numeric(merged['sentiment'], errors='coerce').mean():.2f}",
            )
        else:
            st.metric("Avg Sentiment (news rows)", "—")
    with cK3:
        st.metric(
            "Unique tickers", int(merged["ticker"].nunique()) if "ticker" in merged.columns else 0
        )

    # ---------- Download ----------
    st.download_button(
        "⬇️ Download fused view (CSV)",
        data=merged[cols_existing].to_csv(index=False).encode("utf-8"),
        file_name="sentiment_signal_fusion.csv",
        mime="text/csv",
        key="t19_dl",
    )

# ──────────────────────────────
# Tab 20 — Model Comparison (clean & robust)
# ──────────────────────────────
with tabs[20]:
    st.subheader("📊 Model Comparison")

    # Try primary file, then a common fallback name
    mc = load_csv("model_comparison.csv", RESULTS_DIR)
    if mc.empty:
        mc = load_csv("model_metrics.csv", RESULTS_DIR)

    if mc.empty:
        st.info(
            "No model_comparison.csv (or model_metrics.csv) found. "
            "Expected at least: ['ticker','date','model','close','predicted_close']."
        )
    else:
        # Ensure expected columns exist
        needed_cols = ["ticker", "date", "model", "close", "predicted_close"]
        for c in needed_cols:
            if c not in mc.columns:
                mc[c] = np.nan

        # Dates & numerics
        parse_dates_inplace(mc, ("date",))
        to_numeric(mc, ["close", "predicted_close"])

        # Minimal schema check
        missing = sorted(set(needed_cols) - set(mc.columns))
        if missing:
            st.warning(f"Missing required columns: {missing}")
        else:
            # Drop rows w/o ticker or model
            mc["ticker"] = mc["ticker"].astype(str)
            mc["model"] = mc["model"].astype(str)
            mc = mc.dropna(subset=["ticker", "model"])

            # Ticker selector
            tickers = sorted(mc["ticker"].dropna().unique())
            if not tickers:
                st.info("No tickers available in model comparison file.")
            else:
                sel_ticker = st.selectbox("Select ticker", tickers, key="t20_ticker")

                # Subset and sort
                sub = (
                    mc[mc["ticker"] == sel_ticker]
                    .dropna(subset=["date"])
                    .sort_values("date")
                    .copy()
                )

                models = sorted(sub["model"].dropna().unique())
                if not models:
                    st.info("No models found for the selected ticker.")
                else:
                    sel_models = st.multiselect(
                        "Select models to compare",
                        models,
                        default=models,
                        key="t20_models",
                    )

                    sub = sub[sub["model"].isin(sel_models)].copy()

                    if sub.empty:
                        st.info("No data for the chosen filters.")
                    else:
                        # Compute metrics per model
                        rows = []
                        for m in sel_models:
                            dfm = sub[sub["model"] == m]
                            r2 = r2_score(dfm["close"], dfm["predicted_close"])
                            m_mae = mae(dfm["close"], dfm["predicted_close"])
                            m_rmse = rmse(dfm["close"], dfm["predicted_close"])
                            rows.append({"model": m, "R2": r2, "MAE": m_mae, "RMSE": m_rmse})

                        metrics_df = pd.DataFrame(rows)
                        # Sort by RMSE if available, else by MAE, else by R2 (desc)
                        if "RMSE" in metrics_df:
                            metrics_df = metrics_df.sort_values("RMSE", ascending=True)
                        elif "MAE" in metrics_df:
                            metrics_df = metrics_df.sort_values("MAE", ascending=True)
                        elif "R2" in metrics_df:
                            metrics_df = metrics_df.sort_values("R2", ascending=False)

                        st.subheader("📐 Performance Metrics")
                        st.dataframe(metrics_df, use_container_width=True)

                        # Charts
                        if PLOTLY_OK:
                            # Actual vs predicted
                            st.subheader("📈 Actual vs Predicted")
                            fig = go.Figure()

                            base = (
                                sub[["date", "close"]]
                                .dropna()
                                .drop_duplicates(subset=["date"])
                                .sort_values("date")
                            )
                            if not base.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=base["date"],
                                        y=base["close"],
                                        name="Actual Close",
                                        mode="lines",
                                    )
                                )
                            for m in sel_models:
                                dfm = sub[sub["model"] == m]
                                if dfm["predicted_close"].notna().any():
                                    fig.add_trace(
                                        go.Scatter(
                                            x=dfm["date"],
                                            y=dfm["predicted_close"],
                                            name=f"{m} Predicted",
                                            mode="lines",
                                        )
                                    )
                            fig.update_layout(
                                title=f"{sel_ticker}: Actual vs Predicted (by Model)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                xaxis_rangeslider_visible=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Error distribution (optional but helpful)
                            st.subheader("📉 Error Distribution (Pred - Actual)")
                            err_df = sub.copy()
                            err_df["error"] = err_df["predicted_close"] - err_df["close"]
                            if err_df["error"].notna().any():
                                fig_err = px.histogram(
                                    err_df.dropna(subset=["error"]),
                                    x="error",
                                    color="model",
                                    barmode="overlay",
                                    nbins=40,
                                    title="Prediction Error Distribution",
                                )
                                st.plotly_chart(fig_err, use_container_width=True)
                        else:
                            st.warning("Plotly not installed — `pip install plotly` for charts.")

                        # Download filtered slice
                        st.download_button(
                            label="⬇️ Download filtered comparison (CSV)",
                            data=sub.to_csv(index=False).encode("utf-8"),
                            file_name=f"{sel_ticker}_model_comparison_filtered.csv",
                            mime="text/csv",
                            key="t20_dl",
                        )

# ──────────────────────────────
# Tab 21 — AI Learning Lab (upload + quick strategies)
# ──────────────────────────────
with tabs[21]:
    st.subheader("🧠 AI Learning Lab")
    st.markdown(
        """
        Upload custom OHLC CSV and prototype quick strategies:
        - Moving Average Crossover
        - RSI Oversold/Overbought
        - Bollinger Band Breakout
        """
    )

    uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"], key="t21_upl")
    if uploaded_file:
        try:
            # Flexible schema: accept Date/date/DATE, Close/close etc.
            raw = pd.read_csv(uploaded_file)
            # Try to find a date column
            date_col = next(
                (c for c in ["date", "Date", "timestamp", "time"] if c in raw.columns), None
            )
            if not date_col:
                st.error("CSV must include a date-like column (e.g., 'date' or 'Date').")
                st.stop()
            raw["date"] = pd.to_datetime(raw[date_col], errors="coerce", utc=True).dt.tz_localize(
                None
            )

            # Close price needed for all strategies
            close_col = next(
                (c for c in ["close", "Close", "adj_close", "Adj Close"] if c in raw.columns), None
            )
            if not close_col:
                st.error("CSV must include a price column (e.g., 'close').")
                st.stop()

            df = raw.rename(columns={close_col: "close"}).copy()
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"])

            strategy = st.selectbox(
                "🧠 Choose a Strategy",
                ["Moving Average Crossover", "RSI Strategy", "Bollinger Bands"],
                key="t21_strategy",
            )

            # Helper: safe position → return
            def _returns_from_position(px: pd.Series, pos: pd.Series) -> pd.Series:
                r = px.pct_change().fillna(0.0)
                return r * pos.shift(1).fillna(0.0)

            if strategy == "Moving Average Crossover":
                fast, slow = st.slider("MA windows (fast, slow)", 3, 100, (5, 20), 1, key="t21_ma")
                df["ma_fast"] = df["close"].rolling(fast).mean()
                df["ma_slow"] = df["close"].rolling(slow).mean()
                # Position: long when fast > slow, flat otherwise
                df["position"] = (df["ma_fast"] > df["ma_slow"]).astype(float)
                df["signal_event"] = df["position"].diff().fillna(0.0)  # +1 buy, -1 sell, 0 hold

            elif strategy == "RSI Strategy":
                length = st.slider("RSI length", 5, 30, 14, 1, key="t21_rsi_len")
                lb = st.slider("Oversold (buy below)", 10, 40, 30, 1, key="t21_rsi_lb")
                ub = st.slider("Overbought (sell above)", 60, 90, 70, 1, key="t21_rsi_ub")

                delta = df["close"].diff()
                gain = delta.clip(lower=0).rolling(length).mean()
                loss = (-delta.clip(upper=0)).rolling(length).mean()
                rs = gain / loss.replace(0, np.nan)
                df["rsi"] = 100 - (100 / (1 + rs))
                # Long when RSI < lb, short when RSI > ub, flat otherwise (you can choose long/flat)
                use_short = st.checkbox(
                    "Allow short when RSI > overbought", value=False, key="t21_rsi_short"
                )
                pos = pd.Series(0.0, index=df.index)
                pos[df["rsi"] < lb] = 1.0
                if use_short:
                    pos[df["rsi"] > ub] = -1.0
                df["position"] = pos
                df["signal_event"] = df["position"].diff().fillna(0.0)

            else:  # Bollinger Bands
                window = st.slider("BB window", 10, 60, 20, 1, key="t21_bb_win")
                mult = st.slider("Std dev multiplier", 1.0, 3.5, 2.0, 0.1, key="t21_bb_mult")
                ma = df["close"].rolling(window).mean()
                std = df["close"].rolling(window).std()
                df["upper"] = ma + mult * std
                df["lower"] = ma - mult * std
                # Position: long when close < lower (revert), short when close > upper (optional)
                mean_revert = st.checkbox(
                    "Mean-revert (long lower / short upper)", value=True, key="t21_bb_mr"
                )
                pos = pd.Series(0.0, index=df.index)
                if mean_revert:
                    pos[df["close"] < df["lower"]] = 1.0
                    pos[df["close"] > df["upper"]] = -1.0
                else:
                    # breakout
                    pos[df["close"] > df["upper"]] = 1.0
                    pos[df["close"] < df["lower"]] = -1.0
                df["position"] = pos
                df["signal_event"] = df["position"].diff().fillna(0.0)

            # Strategy performance
            df["strategy_return"] = _returns_from_position(df["close"], df["position"])
            df["cumulative_return"] = (1.0 + df["strategy_return"]).cumprod()

            # KPIs
            stats = perf_stats_from_levels(df.set_index("date")["cumulative_return"])
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Total Return",
                    f"{stats['total_return']:.1%}" if np.isfinite(stats["total_return"]) else "—",
                )
            with c2:
                st.metric("CAGR", f"{stats['cagr']:.1%}" if np.isfinite(stats["cagr"]) else "—")
            with c3:
                st.metric(
                    "Sharpe", f"{stats['sharpe']:.2f}" if np.isfinite(stats["sharpe"]) else "—"
                )
            with c4:
                st.metric(
                    "Max DD", f"{stats['max_dd']:.1%}" if np.isfinite(stats["max_dd"]) else "—"
                )

            st.subheader(f"📈 Strategy Equity Curve — {strategy}")
            if PLOTLY_OK:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"], y=df["cumulative_return"], mode="lines", name="Strategy"
                    )
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return")
                st.plotly_chart(fig, use_container_width=True)
            elif MPL_OK:
                fig, ax = plt.subplots()
                ax.plot(df["date"], df["cumulative_return"], label="Strategy", linewidth=2)
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative Return")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Install plotly or matplotlib to see the chart.")

            with st.expander("Show last 50 rows"):
                st.dataframe(df.tail(50), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


# -------------------------------------------------
# Enhanced guard / posture snapshot helper
# (place this above Tab 22 so both 22 and 23 can call it)
# -------------------------------------------------
def latest_portfolio_status():
    """
    Figure out current guard mode, equity, drawdown, and buying power
    using portfolio_history.csv and live_orders.csv.

    Returns dict:
        {
            "mode": str,
            "reason": str,
            "drawdown_pct": float or nan,
            "buying_power": float or nan,
            "latest_equity": float or nan,
            "reserve_pct": float or nan
        }
    """
    out = {
        "mode": "UNKNOWN",
        "reason": "",
        "drawdown_pct": np.nan,
        "buying_power": np.nan,
        "latest_equity": np.nan,
        "reserve_pct": np.nan,
    }

    # --- Pull equity / drawdown / cash from portfolio_history.csv
    ph = load_csv("portfolio_history.csv", RESULTS_DIR)
    if not ph.empty:
        # make sure we have a clean datetime
        parse_dates_inplace(ph, ("date",))

        # build total_value if it's missing
        ph = derive_total_value(ph)

        # coerce numerics we care about
        for c in ["total_value", "buying_power", "cash"]:
            if c in ph.columns:
                ph[c] = pd.to_numeric(ph[c], errors="coerce")

        # keep only rows that actually have a total_value
        if "total_value" in ph.columns:
            ph = ph.dropna(subset=["total_value"]).sort_values("date").copy()
        else:
            ph = pd.DataFrame()  # fallback: nothing usable

        if not ph.empty:
            # compute running peak + drawdown
            ph["peak"] = ph["total_value"].cummax()
            ph["drawdown"] = ph["total_value"] / ph["peak"] - 1.0

            last = ph.iloc[-1]

            out["latest_equity"] = float(last.get("total_value", np.nan))
            out["drawdown_pct"] = float(last.get("drawdown", np.nan))

            # prefer buying_power if present, else cash
            if "buying_power" in last and pd.notna(last["buying_power"]):
                out["buying_power"] = float(last["buying_power"])
            elif "cash" in last and pd.notna(last["cash"]):
                out["buying_power"] = float(last["cash"])

    # --- Infer guard mode from live_orders.csv
    audit = load_csv("live_orders.csv", RESULTS_DIR)
    if not audit.empty:
        # pick a timestamp column
        time_col = (
            "timestamp"
            if "timestamp" in audit.columns
            else ("ts" if "ts" in audit.columns else None)
        )
        if time_col:
            parse_dates_inplace(audit, (time_col,))
            audit = audit.sort_values(time_col)

        row = audit.iloc[-1]
        note_txt = str(row.get("note", "")).upper()
        stat_txt = str(row.get("status", "")).upper()

        # Priority rules:
        if "LOCKDOWN" in note_txt:
            out["mode"] = "LOCKDOWN"
            out["reason"] = "Explicit LOCKDOWN note"
        elif "DEFENSIVE" in note_txt:
            out["mode"] = "DEFENSIVE"
            out["reason"] = "Explicit DEFENSIVE note"
        elif stat_txt == "OK" or "OK" in note_txt:
            out["mode"] = "NORMAL"
            out["reason"] = "Orders accepted (status OK)"
        else:
            # Fallback: infer based on drawdown
            dd = out.get("drawdown_pct", np.nan)
            if np.isfinite(dd):
                if dd <= -0.10:
                    out["mode"] = "DEFENSIVE"
                    out["reason"] = "Auto DEFENSIVE (>10% drawdown)"
                else:
                    out["mode"] = "NORMAL"
                    out["reason"] = "Stable drawdown (<10%)"

    return out


# -------------------------------------------------
# Tab 22 — Buffett Orders (current)
# -------------------------------------------------
with tabs[22]:
    st.subheader("🧾 Buffett Orders (current)")

    cur = load_csv("buffett_orders.csv", ORDERS_DIR)
    # We'll also optionally read recent execution audit for posture
    exec_audit = load_csv("live_orders.csv", RESULTS_DIR)

    if cur.empty and exec_audit.empty:
        st.info("No buffett_orders.csv in data/orders yet (and no live_orders.csv).")
    else:
        # normalize fields in cur, even if it's empty we keep columns
        if cur.empty:
            cur = pd.DataFrame()

        want_cols = [
            "date",
            "ticker",
            "action",
            "quantity",
            "price",
            "delta_notional",
            "target_weight",
            "current_weight",
            "current_value",
            "target_value",
            "buffett_score",
            "title",
            "url",
            "source",
        ]
        for c in want_cols:
            if c not in cur.columns:
                cur[c] = np.nan

        parse_dates_inplace(cur, ("date",))
        cur["ticker"] = cur["ticker"].astype(str)
        cur["action"] = cur["action"].astype(str).str.upper()
        to_numeric(
            cur,
            [
                "quantity",
                "price",
                "delta_notional",
                "target_weight",
                "current_weight",
                "current_value",
                "target_value",
                "buffett_score",
            ],
        )

        # clickable news/title col
        cur["news"] = cur.apply(
            lambda r: make_clickable(
                r.get("title", r.get("ticker", "")),
                r.get("url", ""),
            ),
            axis=1,
        )

        # KPIs
        if not cur.empty:
            n_syms = cur["ticker"].nunique()
            tw_sum = float(cur["target_weight"].fillna(0).sum())

            buys = cur[cur["action"] == "BUY"] if "action" in cur.columns else pd.DataFrame()
            sells = cur[cur["action"] == "SELL"] if "action" in cur.columns else pd.DataFrame()

            total_buy = sum_safe(buys.get("delta_notional", []))
            total_sell = sum_safe(sells.get("delta_notional", []))
        else:
            n_syms = 0
            tw_sum = np.nan
            total_buy = 0.0
            total_sell = 0.0

        c1, c2, c3a, c3b = st.columns(4)
        with c1:
            st.metric("Symbols", n_syms)
        with c2:
            st.metric("Sum target weights", f"{tw_sum:0.3f}" if np.isfinite(tw_sum) else "—")
        with c3a:
            st.metric("Total BUY $", f"{total_buy:,.0f}")
        with c3b:
            st.metric("Total SELL $", f"{total_sell:,.0f}")

        st.write("**Top BUYS by notional**")
        if not cur.empty:
            buys_sorted = cur[cur["action"] == "BUY"].sort_values("delta_notional", ascending=False)
            if not buys_sorted.empty:
                st.dataframe(
                    buys_sorted.head(10),
                    use_container_width=True,
                )
            else:
                st.caption("No BUY rows.")
        else:
            st.caption("No BUY rows.")

        st.write("**Top SELLS by notional**")
        if not cur.empty:
            sells_sorted = cur[cur["action"] == "SELL"].sort_values(
                "delta_notional", ascending=True
            )
            if not sells_sorted.empty:
                st.dataframe(
                    sells_sorted.head(10),
                    use_container_width=True,
                )
            else:
                st.caption("No SELL rows.")
        else:
            st.caption("No SELL rows.")

        with st.expander("All Orders"):
            show_cols = [
                c
                for c in [
                    "date",
                    "ticker",
                    "action",
                    "quantity",
                    "price",
                    "delta_notional",
                    "target_weight",
                    "current_weight",
                    "news",
                    "source",
                ]
                if (c in cur.columns) or (c == "news")
            ]
            if not cur.empty and show_cols:
                st.markdown(
                    cur[show_cols].to_html(escape=False, index=False),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No order rows available.")

        # Historical buffett_orders_* just as info
        hist = sorted(RESULTS_DIR.glob("buffett_orders_*.csv"))
        if hist:
            st.caption(f"Latest history file: {hist[-1].name} (in {RESULTS_DIR})")

        # ---- Capital posture / Guard snapshot
        st.markdown("### Capital Posture / Guard Status")
        snap = latest_portfolio_status()

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Mode", str(snap.get("mode", "UNKNOWN")))
        with g2:
            dd_val = snap.get("drawdown_pct", np.nan)
            st.metric(
                "Drawdown (est)",
                f"{dd_val:.1%}" if np.isfinite(dd_val) else "—",
            )
        with g3:
            eq_val = snap.get("latest_equity", np.nan)
            st.metric(
                "Equity (est)",
                f"${eq_val:,.0f}" if np.isfinite(eq_val) else "—",
            )
        with g4:
            bp_val = snap.get("buying_power", np.nan)
            st.metric(
                "Buying Power (est)",
                f"${bp_val:,.0f}" if np.isfinite(bp_val) else "—",
            )

        # ---- Execution audit table (what actually got sent)
        st.markdown("### Execution Audit (what actually got sent)")
        a = exec_audit.copy()
        if a.empty:
            st.caption("No live_orders.csv yet.")
        else:
            # Normalize audit columns
            # rename/ensure consistent naming
            if "action" not in a.columns and "side" in a.columns:
                a["action"] = a["side"]

            # handle qty/quantity columns
            if "quantity" not in a.columns and "qty" in a.columns:
                a["quantity"] = a["qty"]

            # 'ts' vs 'timestamp'
            if "timestamp" not in a.columns and "ts" in a.columns:
                a["timestamp"] = a["ts"]

            # coerce numerics
            for c in ["quantity", "qty", "limit_price", "price"]:
                if c in a.columns:
                    a[c] = pd.to_numeric(a[c], errors="coerce")

            # Est BUY capital attempted
            est_buy_notional = 0.0
            try:
                tmp_buy = a[a["action"].astype(str).str.upper() == "BUY"].copy()

                # pick a price col
                price_col = None
                for c in ["limit_price", "price"]:
                    if c in tmp_buy.columns:
                        price_col = c
                        break

                # pick a quantity col
                qty_col = None
                for c in ["quantity", "qty"]:
                    if c in tmp_buy.columns:
                        qty_col = c
                        break

                if price_col and qty_col:
                    px = pd.to_numeric(tmp_buy[price_col], errors="coerce").fillna(0)
                    qx = pd.to_numeric(tmp_buy[qty_col], errors="coerce").fillna(0)
                    est_buy_notional = float((px * qx).sum())
            except Exception:
                pass

            # Sent OK vs skipped/error
            ok_mask = a.get("status", pd.Series([], dtype=object)).astype(str).str.upper() == "OK"
            sent_ok = int(ok_mask.sum()) if "status" in a.columns else 0
            skipped = int((~ok_mask).sum()) if "status" in a.columns else 0

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Sent OK", sent_ok)
            with k2:
                st.metric("Skipped / Error", skipped)
            with k3:
                st.metric("Est. BUY capital attempted", f"${est_buy_notional:,.0f}")

            # show audit table
            show_cols = [
                c
                for c in [
                    "timestamp",
                    "ts",
                    "ticker",
                    "action",
                    "quantity",
                    "qty",
                    "order_type",
                    "tif",
                    "limit_price",
                    "price",
                    "take_profit",
                    "status",
                ]
                if c in a.columns
            ]
            st.dataframe(a[show_cols], use_container_width=True)


# -------------------------------------------------
# Tab 23 — Consolidated Orders (ML × Buffett blend)
# -------------------------------------------------
with tabs[23]:
    st.subheader("🗂️ Consolidated Orders (ML × Buffett blend)")

    ml = load_csv("orders_today.csv", ORDERS_DIR)
    bo = load_csv("buffett_orders.csv", ORDERS_DIR)

    # Also reuse audit + posture for per-idea audit
    audit_df = load_csv("live_orders.csv", RESULTS_DIR)
    snap = latest_portfolio_status()

    if ml.empty and bo.empty and audit_df.empty:
        st.info("No orders_today.csv, no buffett_orders.csv, and no live_orders.csv yet.")
    else:
        # ---------- normalize ML
        if ml.empty:
            ml = pd.DataFrame()
        else:
            need_cols = [
                "date",
                "ticker",
                "action",
                "quantity",
                "price",
                "score",
                "title",
                "url",
                "source",
            ]
            for c in need_cols:
                if c not in ml.columns:
                    ml[c] = np.nan
            parse_dates_inplace(ml, ("date",))
            ml["ticker"] = ml["ticker"].astype(str)
            ml["action"] = ml["action"].astype(str).str.upper()
            to_numeric(ml, ["quantity", "price", "score"])
            ml["order_source"] = "ML"

        # ---------- normalize Buffett
        if bo.empty:
            bo = pd.DataFrame()
        else:
            need_cols = [
                "date",
                "ticker",
                "action",
                "quantity",
                "price",
                "score",
                "title",
                "url",
                "source",
            ]
            for c in need_cols:
                if c not in bo.columns:
                    bo[c] = np.nan
            parse_dates_inplace(bo, ("date",))
            bo["ticker"] = bo["ticker"].astype(str)
            bo["action"] = bo["action"].astype(str).str.upper()
            # we may not have "score" for Buffett, that's fine
            to_numeric(bo, ["quantity", "price", "score"])
            bo["order_source"] = "BUFFETT"

        combo = (
            pd.concat([d for d in [ml, bo] if not d.empty], ignore_index=True)
            if (not ml.empty or not bo.empty)
            else pd.DataFrame()
        )

        if combo.empty:
            st.info("No combined orders to display yet.")
        else:
            combo["news"] = combo.apply(
                lambda r: make_clickable(
                    r.get("title", r.get("ticker", "")),
                    r.get("url", ""),
                ),
                axis=1,
            )

            c1, c2, c3 = st.columns([1.3, 1.1, 1])
            with c1:
                tickers = sorted(combo["ticker"].dropna().unique())
                sel_tickers = st.multiselect("Tickers", tickers, default=tickers, key="t23_tickers")
            with c2:
                sources = sorted(combo["order_source"].dropna().unique())
                sel_sources = st.multiselect("Sources", sources, default=sources, key="t23_sources")
            with c3:
                days = st.slider("Last N days", 1, 60, 14, 1, key="t23_days")

            f = combo.copy()
            if sel_tickers:
                f = f[f["ticker"].isin(sel_tickers)]
            if sel_sources:
                f = f[f["order_source"].isin(sel_sources)]

            if "date" in f.columns and f["date"].notna().any():
                cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=days)
                f = f[pd.to_datetime(f["date"], errors="coerce") >= cutoff]

            f = f.sort_values(["date", "ticker", "order_source"], ascending=[False, True, True])

            # KPI row for this filtered idea list
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Orders", len(f))
            with k2:
                st.metric("Unique tickers", f["ticker"].nunique())
            with k3:
                # try to estimate gross notional if price/quantity exist
                px = pd.to_numeric(f.get("price", np.nan), errors="coerce").fillna(0)
                qx = pd.to_numeric(f.get("quantity", np.nan), errors="coerce").fillna(0)
                gross_notional = float((px * qx).abs().sum())
                st.metric("Gross notional", f"${gross_notional:,.0f}")

            show_cols = [
                c
                for c in [
                    "date",
                    "ticker",
                    "order_source",
                    "action",
                    "quantity",
                    "price",
                    "score",
                    "news",
                ]
                if (c in f.columns) or (c == "news")
            ]
            st.dataframe(f[show_cols], use_container_width=True)

            st.download_button(
                "⬇️ Download consolidated (CSV)",
                data=f.to_csv(index=False).encode("utf-8"),
                file_name="consolidated_orders_filtered.csv",
                mime="text/csv",
                key="t23_dl",
            )

        # -------- Guard snapshot for these blended ideas
        st.markdown("### Capital Posture / Guard Status")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Mode", str(snap.get("mode", "UNKNOWN")))
        with g2:
            dd_val = snap.get("drawdown_pct", np.nan)
            st.metric("Drawdown (est)", f"{dd_val:.1%}" if np.isfinite(dd_val) else "—")
        with g3:
            eq_val = snap.get("latest_equity", np.nan)
            st.metric(
                "Equity (est)",
                f"${eq_val:,.0f}" if np.isfinite(eq_val) else "—",
            )
        with g4:
            bp_val = snap.get("buying_power", np.nan)
            st.metric(
                "Buying Power (est)",
                f"${bp_val:,.0f}" if np.isfinite(bp_val) else "—",
            )

        # -------- Targeted audit for only these filtered tickers/sources
        st.markdown("### Execution Audit for These Ideas")

        a = audit_df.copy()
        if a.empty:
            st.caption("No live_orders.csv yet.")
        else:
            # normalize audit schema just like tab 22
            if "action" not in a.columns and "side" in a.columns:
                a["action"] = a["side"]
            if "quantity" not in a.columns and "qty" in a.columns:
                a["quantity"] = a["qty"]
            if "timestamp" not in a.columns and "ts" in a.columns:
                a["timestamp"] = a["ts"]

            for c in ["quantity", "qty", "limit_price", "price"]:
                if c in a.columns:
                    a[c] = pd.to_numeric(a[c], errors="coerce")

            # filter same tickers/time window
            if "ticker" in a.columns and sel_tickers:
                a = a[a["ticker"].astype(str).isin(sel_tickers)].copy()

            # est BUY notional for this slice
            est_buy_notional = 0.0
            try:
                tmp_buy = a[a["action"].astype(str).str.upper() == "BUY"].copy()
                price_col = None
                for c in ["limit_price", "price"]:
                    if c in tmp_buy.columns:
                        price_col = c
                        break
                qty_col = None
                for c in ["quantity", "qty"]:
                    if c in tmp_buy.columns:
                        qty_col = c
                        break
                if price_col and qty_col:
                    px = pd.to_numeric(tmp_buy[price_col], errors="coerce").fillna(0)
                    qx = pd.to_numeric(tmp_buy[qty_col], errors="coerce").fillna(0)
                    est_buy_notional = float((px * qx).sum())
            except Exception:
                pass

            ok_mask = a.get("status", pd.Series([], dtype=object)).astype(str).str.upper() == "OK"
            sent_ok = int(ok_mask.sum()) if "status" in a.columns else 0
            skipped = int((~ok_mask).sum()) if "status" in a.columns else 0

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Sent OK", sent_ok)
            with k2:
                st.metric("Skipped / Error", skipped)
            with k3:
                st.metric("Est. BUY capital attempted", f"${est_buy_notional:,.0f}")

            show_cols2 = [
                c
                for c in [
                    "timestamp",
                    "ts",
                    "ticker",
                    "action",
                    "quantity",
                    "qty",
                    "order_type",
                    "tif",
                    "limit_price",
                    "price",
                    "take_profit",
                    "status",
                ]
                if c in a.columns
            ]
            st.dataframe(a[show_cols2], use_container_width=True)

# ──────────────────────────────
# Tab 24 — AI Feedback (allocator runs)
# ──────────────────────────────
with tabs[24]:
    st.subheader("🤖 AI Feedback (allocator runs)")
    rows = load_jsonl(RESULTS_DIR / "ai_feedback.jsonl")
    if not rows:
        st.info("No ai_feedback.jsonl yet.")
    else:
        last = rows[-1] if rows else {}
        runs = len(rows)
        total_buy = (last.get("orders", {}) or {}).get("total_buy_notional", 0) or 0
        total_sell = (last.get("orders", {}) or {}).get("total_sell_notional", 0) or 0
        uni_size = (last.get("universe", {}) or {}).get("count", None)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Runs", runs)
        with c2:
            st.metric("Total BUY $ (last)", f"{total_buy:,.0f}")
        with c3:
            st.metric("Total SELL $ (last)", f"{total_sell:,.0f}")
        with c4:
            st.metric("Universe size (last)", uni_size if uni_size is not None else "—")

        with st.expander("Latest run — details"):
            st.json(last)

        def _flatten(d, prefix=""):
            out = {}
            for k, v in (d or {}).items():
                kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    out.update(_flatten(v, kk))
                else:
                    out[kk] = v
            return out

        table = pd.DataFrame([_flatten(r) for r in rows])
        st.write("All feedback records")
        st.dataframe(table, use_container_width=True)

# ──────────────────────────────
# Tab 25 — Equal-Weight Portfolio vs Benchmark (safe indexing)
# ──────────────────────────────
with tabs[25]:
    st.subheader("📚 Equal-Weight Portfolio vs Benchmark")

    df = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if df.empty:
        st.info("No strategy_vs_market.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        df = df.dropna(subset=["date"]).copy().sort_values(["ticker", "date"])

        has_ret = {"strategy_return", "market_return"}.issubset(df.columns)
        has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(df.columns)

        if has_ret and not has_cum:
            df["strategy_return"] = pd.to_numeric(df["strategy_return"], errors="coerce").fillna(
                0.0
            )
            df["market_return"] = pd.to_numeric(df["market_return"], errors="coerce").fillna(0.0)
            df["cumulative_strategy"] = df.groupby("ticker")["strategy_return"].apply(
                lambda s: (1 + s).cumprod()
            )
            df["cumulative_market"] = df.groupby("ticker")["market_return"].apply(
                lambda s: (1 + s).cumprod()
            )
        else:
            to_numeric(df, ["cumulative_strategy", "cumulative_market"])

        sub = (
            df[["date", "ticker", "cumulative_strategy", "cumulative_market"]]
            .dropna(subset=["ticker"])
            .set_index(["date", "ticker"])
            .sort_index()
            .copy()
        )

        # Daily returns by ticker
        sub["strat_ret"] = sub.groupby(level=1)["cumulative_strategy"].pct_change().fillna(0.0)
        sub["mkt_ret"] = sub.groupby(level=1)["cumulative_market"].pct_change().fillna(0.0)

        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            bench_choice = st.selectbox(
                "Benchmark", ["Avg Market (across tickers)", "SPY (ticker)"], key="t25_bench"
            )
        with c2:
            normalize = st.checkbox("Normalize to 1.0 at start", value=True, key="t25_norm")
        with c3:
            show_kpis = st.checkbox("Show KPIs", value=True, key="t25_kpi")

        # Equal-weight portfolio: mean across tickers per day
        eq_strat_ret = sub["strat_ret"].groupby(level=0).mean()
        avg_mkt_ret = sub["mkt_ret"].groupby(level=0).mean()
        dates_index = eq_strat_ret.index

        # Pick benchmark series
        if bench_choice.startswith("SPY") and (
            "SPY" in df.get("ticker", pd.Series(dtype=str)).astype(str).unique()
        ):
            spy = df[df["ticker"] == "SPY"].copy().sort_values("date")
            if has_ret and not has_cum:
                spy["bench_ret"] = pd.to_numeric(spy["market_return"], errors="coerce").fillna(0.0)
            else:
                to_numeric(spy, ["cumulative_market"])
                spy["bench_ret"] = spy["cumulative_market"].pct_change().fillna(0.0)
            bench_ret = spy.set_index("date")["bench_ret"].reindex(dates_index).fillna(0.0)
        else:
            bench_ret = avg_mkt_ret.reindex(dates_index).fillna(0.0)

        # Build cumulative curves
        eq_strat = (1.0 + eq_strat_ret).cumprod()
        bench = (1.0 + bench_ret).cumprod()

        if normalize:
            eq_strat = normalize_to_one(eq_strat)
            bench = normalize_to_one(bench)

        if show_kpis:
            s_stats = perf_stats_from_levels(eq_strat.dropna())
            b_stats = perf_stats_from_levels(bench.dropna())
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            with k1:
                st.metric(
                    "Portfolio Total",
                    (
                        f"{s_stats['total_return']:.1%}"
                        if np.isfinite(s_stats["total_return"])
                        else "—"
                    ),
                )
            with k2:
                st.metric(
                    "Portfolio CAGR",
                    f"{s_stats['cagr']:.1%}" if np.isfinite(s_stats["cagr"]) else "—",
                )
            with k3:
                st.metric(
                    "Portfolio Sharpe",
                    f"{s_stats['sharpe']:.2f}" if np.isfinite(s_stats["sharpe"]) else "—",
                )
            with k4:
                st.metric(
                    "Bench Total",
                    (
                        f"{b_stats['total_return']:.1%}"
                        if np.isfinite(b_stats["total_return"])
                        else "—"
                    ),
                )
            with k5:
                st.metric(
                    "Bench CAGR", f"{b_stats['cagr']:.1%}" if np.isfinite(b_stats["cagr"]) else "—"
                )
            with k6:
                st.metric(
                    "Bench MaxDD",
                    f"{b_stats['max_dd']:.1%}" if np.isfinite(b_stats["max_dd"]) else "—",
                )

        if not PLOTLY_OK:
            st.warning("Plotly not installed — `pip install plotly` for charts.")
        else:
            plot_df = pd.DataFrame({"Equal-Weight Portfolio": eq_strat, "Benchmark": bench}).dropna(
                how="all"
            )
            fig = go.Figure()
            if plot_df["Equal-Weight Portfolio"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=plot_df.index,
                        y=plot_df["Equal-Weight Portfolio"],
                        name="Equal-Weight Portfolio",
                        mode="lines",
                    )
                )
            if plot_df["Benchmark"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=plot_df.index, y=plot_df["Benchmark"], name="Benchmark", mode="lines"
                    )
                )
            ttl = "Equal-Weight Portfolio vs Benchmark" + (" (normalized)" if normalize else "")
            fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📎 Ticker attribution (avg daily return over period)"):
            attr = sub["strat_ret"].groupby(level=1).mean().sort_values(ascending=False)
            st.dataframe(
                attr.reset_index().rename(columns={"strat_ret": "avg_daily_ret"}),
                use_container_width=True,
            )

# ──────────────────────────────
# Tab 26 — Smart-Weight Portfolio vs Benchmark (cleaned)
# ──────────────────────────────
with tabs[26]:
    st.subheader("🧮 Smart-Weight Portfolio vs Benchmark")

    # ——— Local helpers (guarded so order never breaks) ———
    if "ema_smooth_wide" not in globals():

        def ema_smooth_wide(W: pd.DataFrame, span: int) -> pd.DataFrame:
            if W is None or W.empty or span <= 1:
                return W
            return W.ewm(span=span, adjust=False, min_periods=1).mean()

    def _daily_turnover(W: pd.DataFrame) -> pd.Series:
        if W is None or W.empty:
            return pd.Series(dtype=float)
        dW = W.diff().abs()
        return 0.5 * dW.sum(axis=1)

    # --- Load signals we’ll use to derive raw weights
    sig_src = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig_src.empty:
        sig_src = load_csv("signals.csv", RESULTS_DIR)

    if sig_src.empty or "ticker" not in sig_src.columns:
        st.info("No signals found for Smart-Weight tab.")
    else:
        # Clean dates & numeric score fields
        sig_src = ensure_date(
            sig_src,
            candidates=["date", "as_of", "timestamp", "time", "datetime", "Date"],
            normalize=True,
        )
        to_numeric(sig_src, ["confidence", "total_score"])

        # --- UI
        c1, c2, c3 = st.columns(3)
        with c1:
            scheme = st.selectbox(
                "Weighting scheme",
                ["Confidence (daily)", "Total Score (daily)"],
                index=0,
                key="t26_scheme",
                help="Which signal column to convert to weights each day.",
            )
        with c2:
            _ = st.selectbox(
                "Benchmark",
                ["Avg Market (across tickers)"],
                index=0,
                key="t26_bench",
                help="Reference curve for comparison.",
            )
        with c3:
            normalize_curves = st.checkbox(
                "Normalize to 1.0 at start",
                value=True,
                key="t26_norm",
                help="Rescales both curves so they start at 1.0.",
            )

        max_pct = st.slider("Max weight per ticker (%)", 1, 50, 15, 1, key="t26_maxw") / 100.0
        alpha = st.slider("Blend α (smart share)", 0.0, 1.0, 0.30, 0.05, key="t26_alpha")
        cost_bps = st.slider("Trading cost (bps per $ traded)", 0, 50, 5, 1, key="t26_cost") / 1e4
        ema_span = st.slider("Signal smoothing (EMA days)", 1, 10, 1, 1, key="t26_ema")
        show_kpis = st.checkbox("Show KPIs", value=True, key="t26_kpis")

        # Choose the signal column
        score_col = "confidence" if "Confidence" in scheme else "total_score"
        if score_col not in sig_src.columns:
            st.warning(f"'{score_col}' not present; falling back to 'confidence'.")
            score_col = "confidence"
            if score_col not in sig_src.columns:
                sig_src[score_col] = np.nan

        # Pivot to wide: rows=date, cols=ticker, values=score
        sc = sig_src[["date", "ticker", score_col]].dropna(subset=["date", "ticker"]).copy()
        sc["ticker"] = sc["ticker"].astype(str)
        raw_wide = sc.pivot_table(
            index="date", columns="ticker", values=score_col, aggfunc="last"
        ).sort_index()

        # Smooth the raw daily scores before stabilizing -> calmer weights
        raw_wide = ema_smooth_wide(raw_wide, ema_span)

        if raw_wide.shape[1] == 0:
            st.info("No tickers available after processing.")
        else:
            # --- Returns source (per-ticker daily returns) ---
            mkt = load_csv("market_by_ticker.csv", RESULTS_DIR)

            if mkt.empty:
                # Fallback: compute returns from each ticker’s parquet (…/results/{T}.parquet)
                rets_list = []
                for t in raw_wide.columns:
                    pq = RESULTS_DIR / f"{t}.parquet"
                    px = load_parquet(pq)
                    if px.empty or "date" not in px.columns or "close" not in px.columns:
                        continue
                    parse_dates_inplace(px, ("date",))
                    px = px.dropna(subset=["date", "close"]).sort_values("date")
                    px["ret"] = pd.to_numeric(px["close"], errors="coerce").pct_change()
                    tmp = px[["date", "ret"]].copy()
                    tmp["ticker"] = t
                    rets_list.append(tmp)
                mkt = pd.concat(rets_list, ignore_index=True) if rets_list else pd.DataFrame()

            if mkt.empty or not {"date", "ticker"}.issubset(mkt.columns):
                st.info("Could not find per-ticker market returns to build portfolio.")
            else:
                if "ret" not in mkt.columns:
                    cand = next(
                        (
                            c
                            for c in ["return", "daily_return", "market_return"]
                            if c in mkt.columns
                        ),
                        None,
                    )
                    if cand:
                        mkt["ret"] = pd.to_numeric(mkt[cand], errors="coerce")
                mkt = mkt[["date", "ticker", "ret"]].copy()
                mkt["ticker"] = mkt["ticker"].astype(str)
                mkt["ret"] = pd.to_numeric(mkt["ret"], errors="coerce").fillna(0.0)
                mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce", utc=True).dt.tz_localize(
                    None
                )

                R = mkt.pivot_table(
                    index="date", columns="ticker", values="ret", aggfunc="last"
                ).sort_index()

                # Align universe
                common_cols = sorted(set(raw_wide.columns) & set(R.columns))
                if not common_cols:
                    st.info("No overlap between weights universe and returns universe.")
                else:

                    def build_portfolio(max_cap: float, a: float):
                        # Use the robust global stabilize_weights
                        W_s = stabilize_weights(raw_wide[common_cols], max_cap)
                        # equal-weight anchor
                        W_e = pd.DataFrame(
                            1.0 / len(common_cols),
                            index=W_s.index,
                            columns=common_cols,
                        )
                        Wf = a * W_s + (1.0 - a) * W_e
                        Wf = Wf.div(Wf.sum(axis=1), axis=0)

                        idx = Wf.index.intersection(R.index)
                        Wf = Wf.loc[idx, common_cols]
                        Rt = R.loc[idx, common_cols]

                        gross_ret = (Wf * Rt).sum(axis=1).fillna(0.0)
                        tvr = _daily_turnover(Wf).reindex(idx).fillna(0.0)
                        cost = cost_bps * tvr
                        net_ret = gross_ret - cost

                        bench_ret = Rt.mean(axis=1).fillna(0.0)

                        # --- compute stats on RAW (non-normalized) curves
                        port_lvl_g_raw = (1.0 + gross_ret).cumprod()
                        port_lvl_n_raw = (1.0 + net_ret).cumprod()
                        bench_lvl_raw = (1.0 + bench_ret).cumprod()
                        stats_g = perf_stats_from_levels(port_lvl_g_raw)
                        stats_n = perf_stats_from_levels(port_lvl_n_raw)

                        # separate plotting versions (optionally normalized)
                        port_lvl_g_plot = (
                            normalize_to_one(port_lvl_g_raw) if normalize_curves else port_lvl_g_raw
                        )
                        port_lvl_n_plot = (
                            normalize_to_one(port_lvl_n_raw) if normalize_curves else port_lvl_n_raw
                        )
                        bench_lvl_plot = (
                            normalize_to_one(bench_lvl_raw) if normalize_curves else bench_lvl_raw
                        )

                        return (
                            Wf,
                            port_lvl_g_plot,
                            port_lvl_n_plot,
                            bench_lvl_plot,
                            tvr,
                            stats_g,
                            stats_n,
                            bench_lvl_raw,  # for bench stats outside
                        )

                    W, port_lvl_g, port_lvl_n, bench_lvl, tvr, ps_g, ps_n, bench_raw = (
                        build_portfolio(max_pct, alpha)
                    )
                    bs = perf_stats_from_levels(bench_raw)

                    if show_kpis:
                        k1, k2, k3, k4, k5, k6 = st.columns(6)
                        with k1:
                            st.metric(
                                "Net Total",
                                (
                                    f"{ps_n['total_return']:.1%}"
                                    if np.isfinite(ps_n["total_return"])
                                    else "—"
                                ),
                            )
                        with k2:
                            st.metric(
                                "Net CAGR",
                                f"{ps_n['cagr']:.1%}" if np.isfinite(ps_n["cagr"]) else "—",
                            )
                        with k3:
                            st.metric(
                                "Net Sharpe",
                                f"{ps_n['sharpe']:.2f}" if np.isfinite(ps_n["sharpe"]) else "—",
                            )
                        with k4:
                            st.metric(
                                "Bench Total",
                                (
                                    f"{bs['total_return']:.1%}"
                                    if np.isfinite(bs["total_return"])
                                    else "—"
                                ),
                            )
                        with k5:
                            st.metric(
                                "Bench CAGR",
                                f"{bs['cagr']:.1%}" if np.isfinite(bs["cagr"]) else "—",
                            )
                        with k6:
                            avg_mo_tvr = (
                                float(tvr.resample("M").mean().mean()) if not tvr.empty else np.nan
                            )
                            st.metric(
                                "Avg Monthly Turnover",
                                f"{avg_mo_tvr:.1%}" if np.isfinite(avg_mo_tvr) else "—",
                            )

                    if not PLOTLY_OK:
                        st.warning("Plotly not installed — `pip install plotly` for charts.")
                    else:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=port_lvl_n.index,
                                y=port_lvl_n.values,
                                name="Portfolio (Net)",
                                mode="lines",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=port_lvl_g.index,
                                y=port_lvl_g.values,
                                name="Portfolio (Gross)",
                                mode="lines",
                                opacity=0.55,
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=bench_lvl.index,
                                y=bench_lvl.values,
                                name="Benchmark",
                                mode="lines",
                            )
                        )
                        ttl = "Smart-Weight Portfolio vs Benchmark" + (
                            " (normalized)" if normalize_curves else ""
                        )
                        fig.update_layout(
                            title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Average weights by ticker (over period)"):
                        avg_w = W.mean(axis=0).sort_values(ascending=False)
                        st.dataframe(
                            avg_w.rename("avg_weight").to_frame(), use_container_width=True
                        )

                    with st.expander("🗺️ Weights heatmap (last ~120 days)"):
                        if PLOTLY_OK and not W.empty:
                            W_hm = W.tail(120)
                            hm = go.Figure(
                                data=go.Heatmap(
                                    z=W_hm.T.values,
                                    x=W_hm.index.astype(str),
                                    y=W_hm.columns.astype(str),
                                    colorbar=dict(title="Weight"),
                                )
                            )
                            hm.update_layout(
                                title="Daily Weights Heatmap",
                                xaxis_title="Date",
                                yaxis_title="Ticker",
                            )
                            st.plotly_chart(hm, use_container_width=True)
                        else:
                            st.caption("No weights to display.")

                    with st.expander("⬇️ Export data"):
                        curves = pd.DataFrame(
                            {
                                "date": port_lvl_n.index,
                                "portfolio_net": port_lvl_n.values,
                                "portfolio_gross": port_lvl_g.reindex(port_lvl_n.index).values,
                                "benchmark": bench_lvl.reindex(port_lvl_n.index).values,
                            }
                        ).set_index("date")
                        daily = pd.DataFrame(
                            {
                                "date": W.index,
                                "turnover": _daily_turnover(W).reindex(W.index).values,
                            }
                        ).set_index("date")
                        st.download_button(
                            "Download equity curves (CSV)",
                            curves.to_csv().encode("utf-8"),
                            "tab26_curves.csv",
                            "text/csv",
                            key="t26_dl_curves",
                        )
                        st.download_button(
                            "Download turnover (CSV)",
                            daily.to_csv().encode("utf-8"),
                            "tab26_turnover.csv",
                            "text/csv",
                            key="t26_dl_tvr",
                        )

# ──────────────────────────────
# Tab 27 — Confidence Calibration (tight)
# ──────────────────────────────
with tabs[27]:
    st.subheader("🧪 Confidence Calibration")

    # ---- helpers
    def _safe_decile_map(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series(np.nan, index=s.index)
        # add small jitter to avoid all-duplicates causing identical quantiles
        sj = s + np.random.default_rng(42).normal(0, 1e-9, size=len(s))
        qs = np.nanquantile(sj, np.linspace(0, 1, 11))

        def _to_dec(x):
            if not np.isfinite(x):
                return np.nan
            d = int(np.searchsorted(qs, x, side="right"))
            return min(max(d, 1), 10)

        return s.map(_to_dec)

    def _calibration_table(
        df: pd.DataFrame, conf_col="confidence", ret_col="target_ret"
    ) -> pd.DataFrame:
        if df.empty or conf_col not in df.columns or ret_col not in df.columns:
            return pd.DataFrame()
        d = df[[conf_col, ret_col]].copy()
        d[conf_col] = pd.to_numeric(d[conf_col], errors="coerce")
        d[ret_col] = pd.to_numeric(d[ret_col], errors="coerce")
        d = d.dropna(subset=[conf_col, ret_col])
        if d.empty:
            return pd.DataFrame()
        d["decile"] = _safe_decile_map(d[conf_col])
        d = d.dropna(subset=["decile"])
        if d.empty:
            return pd.DataFrame()
        grp = d.groupby("decile", observed=True)
        out = pd.DataFrame(
            {
                "count": grp.size(),
                "avg_return": grp[ret_col].mean(),
                "hitrate_%": grp.apply(lambda g: (g[ret_col] > 0).mean() * 100.0),
            }
        ).reset_index()
        out["decile"] = out["decile"].astype(int)
        return out.sort_values("decile")

    # ---- data
    svs = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if svs.empty:
        st.info("No strategy_vs_market.csv yet.")
        st.stop()

    parse_dates_inplace(svs, ("date",))
    svs = svs.dropna(subset=["date", "ticker"]).copy().sort_values(["ticker", "date"])

    has_ret = {"strategy_return", "market_return"}.issubset(svs.columns)
    has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(svs.columns)
    if has_ret and not has_cum:
        svs["strategy_return"] = pd.to_numeric(svs["strategy_return"], errors="coerce").fillna(0)
        svs["market_return"] = pd.to_numeric(svs["market_return"], errors="coerce").fillna(0)
        svs["cumulative_strategy"] = svs.groupby("ticker")["strategy_return"].apply(
            lambda s: (1 + s).cumprod()
        )
        svs["cumulative_market"] = svs.groupby("ticker")["market_return"].apply(
            lambda s: (1 + s).cumprod()
        )
    else:
        to_numeric(svs, ["cumulative_strategy", "cumulative_market"])

    panel = (
        svs[["date", "ticker", "cumulative_strategy", "cumulative_market"]]
        .set_index(["date", "ticker"])
        .sort_index()
    )
    panel.index = panel.index.set_names(["date", "ticker"])
    panel["strat_ret"] = panel.groupby(level=1)["cumulative_strategy"].pct_change().fillna(0.0)
    panel["mkt_ret"] = panel.groupby(level=1)["cumulative_market"].pct_change().fillna(0.0)
    panel["fwd_strat_ret"] = panel.groupby(level=1)["strat_ret"].shift(-1)
    panel["fwd_mkt_ret"] = panel.groupby(level=1)["mkt_ret"].shift(-1)

    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig.empty:
        sig = load_csv("signals.csv", RESULTS_DIR)
    if sig.empty:
        st.info("Need signals.csv with at least [date, ticker, confidence].")
        st.stop()

    sig = ensure_date(
        sig, ["date", "as_of", "timestamp", "time", "datetime", "Date"], normalize=False
    )
    sig = sig.dropna(subset=["date", "ticker"]).copy()
    to_numeric(sig, ["confidence"])
    sig_idx = sig.set_index(["date", "ticker"]).sort_index()

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        target = st.selectbox(
            "Calibration target",
            ["Next-day Market Return", "Next-day Strategy Return"],
            0,
            key="t27_target",
        )
    with c2:
        min_obs = st.slider("Min obs per decile", 5, 500, 30, 5, key="t27_minobs")
    with c3:
        show_scatter = st.checkbox("Show scatter", True, key="t27_scatter")

    tgt = panel["fwd_mkt_ret"] if target.startswith("Next-day Market") else panel["fwd_strat_ret"]
    merged = sig_idx.join(tgt.rename("target_ret"), how="left").dropna(
        subset=["confidence", "target_ret"]
    )
    if merged.empty:
        st.info("After alignment, no rows found (check dates/tickers overlap).")
        st.stop()

    dec = _calibration_table(merged.reset_index()[["confidence", "target_ret"]])
    if dec.empty:
        st.info("No decile stats available.")
        st.stop()

    dec_f = dec[dec["count"] >= min_obs] if (dec["count"] >= min_obs).any() else dec
    if (dec["count"] < min_obs).any():
        st.caption("Some deciles under threshold were hidden.")

    st.markdown("**Confidence deciles vs next-day return**")
    st.dataframe(dec_f, use_container_width=True)

    if PLOTLY_OK and not dec_f.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dec_f["decile"], y=dec_f["avg_return"], name="Avg next-day return"))
        fig.add_trace(
            go.Scatter(
                x=dec_f["decile"],
                y=dec_f["hitrate_%"],
                mode="lines+markers",
                name="Hitrate (%)",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Calibration: Confidence deciles vs next-day return / hitrate",
            xaxis_title="Confidence decile (1=low … 10=high)",
            yaxis=dict(title="Avg next-day return"),
            yaxis2=dict(title="Hitrate (%)", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

    if show_scatter and PLOTLY_OK:
        samp = (
            merged.sample(min(len(merged), 4000), random_state=42) if len(merged) > 4000 else merged
        )
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=samp["confidence"], y=samp["target_ret"], mode="markers", opacity=0.45, name="obs"
            )
        )
        fig2.update_layout(
            title="Confidence vs next-day return (sampled)",
            xaxis_title="Confidence",
            yaxis_title="Next-day return",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────
# Tab 28 — Confidence-Filtered Portfolio vs Benchmark (tight, raw-KPI)
# ──────────────────────────────
with tabs[28]:
    st.subheader("🧪 Confidence-Filtered Portfolio vs Benchmark")

    # helpers (guarded)
    if "ema_smooth_wide" not in globals():

        def ema_smooth_wide(W: pd.DataFrame, span: int) -> pd.DataFrame:
            if W is None or W.empty or span <= 1:
                return W
            return W.ewm(span=span, adjust=False, min_periods=1).mean()

    if "enforce_min_hold" not in globals():

        def enforce_min_hold(mask_df: pd.DataFrame, min_days: int) -> pd.DataFrame:
            if mask_df is None or mask_df.empty or min_days <= 1:
                return mask_df
            out = mask_df.copy()
            for k in range(1, min_days):
                out |= mask_df.shift(k).fillna(False)
            return out

    def _daily_turnover(W: pd.DataFrame) -> pd.Series:
        if W is None or W.empty:
            return pd.Series(dtype=float)
        dW = W.diff().abs()
        return 0.5 * dW.sum(axis=1)

    svs = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if svs.empty:
        st.info("No strategy_vs_market.csv yet.")
        st.stop()

    parse_dates_inplace(svs, ("date",))
    svs = svs.dropna(subset=["date", "ticker"]).copy().sort_values(["ticker", "date"])

    has_ret = {"strategy_return", "market_return"}.issubset(svs.columns)
    has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(svs.columns)
    if has_ret and not has_cum:
        svs["strategy_return"] = pd.to_numeric(svs["strategy_return"], errors="coerce").fillna(0)
        svs["market_return"] = pd.to_numeric(svs["market_return"], errors="coerce").fillna(0)
        svs["cumulative_strategy"] = svs.groupby("ticker")["strategy_return"].apply(
            lambda s: (1 + s).cumprod()
        )
        svs["cumulative_market"] = svs.groupby("ticker")["market_return"].apply(
            lambda s: (1 + s).cumprod()
        )
    else:
        to_numeric(svs, ["cumulative_strategy", "cumulative_market"])

    panel = (
        svs[["date", "ticker", "cumulative_strategy", "cumulative_market"]]
        .set_index(["date", "ticker"])
        .sort_index()
    )
    panel.index = panel.index.set_names(["date", "ticker"])
    panel["strat_ret"] = panel.groupby(level=1)["cumulative_strategy"].pct_change().fillna(0.0)
    panel["mkt_ret"] = panel.groupby(level=1)["cumulative_market"].pct_change().fillna(0.0)
    panel["fwd_strat_ret"] = panel.groupby(level=1)["strat_ret"].shift(-1)
    panel["fwd_mkt_ret"] = panel.groupby(level=1)["mkt_ret"].shift(-1)

    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig.empty:
        sig = load_csv("signals.csv", RESULTS_DIR)
    if sig.empty:
        st.info("Need signals.csv with [date, ticker, confidence].")
        st.stop()

    sig = ensure_date(
        sig, ["date", "as_of", "timestamp", "time", "datetime", "Date"], normalize=False
    )
    sig = sig.dropna(subset=["date", "ticker"]).copy()
    to_numeric(sig, ["confidence"])
    sidx = sig.set_index(["date", "ticker"]).sort_index()

    c1, c2, c3, c4 = st.columns([1.4, 1.1, 1.1, 1.2])
    with c1:
        bench_choice = st.selectbox(
            "Benchmark", ["Avg Market (across tickers)", "SPY (ticker)"], 0, key="t28_bench"
        )
    with c2:
        thr = st.slider("Confidence threshold (≥)", 0.0, 1.0, 0.70, 0.01, key="t28_thr")
    with c3:
        cost_bps = st.slider("Trading cost (bps per $ traded)", 0, 50, 5, 1, key="t28_cost") / 1e4
    with c4:
        normalize_plot = st.checkbox("Normalize curves for chart", True, key="t28_norm")
    ema_span = st.slider("Confidence smoothing (EMA days)", 1, 10, 1, 1, key="t28_ema")
    min_hold = st.slider("Min hold days", 1, 10, 1, 1, key="t28_hold")
    show_kpis = st.checkbox("Show KPIs", True, key="t28_kpi")

    # build mask → weights
    c_mat = sidx["confidence"].unstack("ticker").sort_index().fillna(0.0)
    c_mat = ema_smooth_wide(c_mat, ema_span)
    mask_wide = enforce_min_hold((c_mat >= thr), min_hold)

    # portfolio returns (equal-weight among passing tickers)
    r = panel["fwd_strat_ret"]
    if mask_wide is None or mask_wide.empty:
        dates = panel.index.get_level_values(0).unique()
        port_ret_gross = pd.Series(0.0, index=dates)
        W = pd.DataFrame(index=dates)
    else:
        mask_wide = mask_wide.reindex(
            index=r.unstack("ticker").index, columns=mask_wide.columns
        ).fillna(False)
        n = mask_wide.sum(axis=1)
        eq_w = mask_wide.div(n.replace(0, np.nan), axis=0).fillna(0.0)
        W = eq_w.copy()
        w_long = eq_w.stack(dropna=False).fillna(0.0)
        w_long.index.set_names(panel.index.names, inplace=True)
        port_ret_gross = (w_long * r.reindex(w_long.index).fillna(0.0)).groupby(level=0).sum()

    tvr = _daily_turnover(W)
    cost = cost_bps * tvr.reindex(port_ret_gross.index).fillna(0.0)
    port_ret_net = port_ret_gross - cost

    # benchmark (avg mkt or SPY)
    avg_mkt_ret = panel["mkt_ret"].groupby(level=0).mean()
    dates_index = port_ret_net.index
    tickers_unique = svs.get("ticker", pd.Series(dtype=str)).astype(str).unique()
    if bench_choice.startswith("SPY") and ("SPY" in tickers_unique):
        spy = svs[svs["ticker"] == "SPY"].copy().sort_values("date")
        if has_ret and not has_cum:
            spy["bench_ret"] = pd.to_numeric(spy["market_return"], errors="coerce").fillna(0)
        else:
            to_numeric(spy, ["cumulative_market"])
            spy["bench_ret"] = spy["cumulative_market"].pct_change().fillna(0)
        bench_ret = spy.set_index("date")["bench_ret"].reindex(dates_index).fillna(0)
    else:
        bench_ret = avg_mkt_ret.reindex(dates_index).fillna(0)

    # RAW curves for KPIs
    port_curve_net_raw = (1 + port_ret_net.fillna(0)).cumprod()
    port_curve_gross_raw = (1 + port_ret_gross.fillna(0)).cumprod()
    bench_curve_raw = (1 + bench_ret.fillna(0)).cumprod()

    # KPI on raw
    if show_kpis:
        s_stats = perf_stats_from_levels(port_curve_net_raw.dropna())
        b_stats = perf_stats_from_levels(bench_curve_raw.dropna())
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.metric(
                "Net Total",
                (
                    f"{s_stats.get('total_return', np.nan):.1%}"
                    if np.isfinite(s_stats.get("total_return", np.nan))
                    else "—"
                ),
            )
        with k2:
            st.metric(
                "Net CAGR",
                (
                    f"{s_stats.get('cagr', np.nan):.1%}"
                    if np.isfinite(s_stats.get("cagr", np.nan))
                    else "—"
                ),
            )
        with k3:
            st.metric(
                "Net Sharpe",
                (
                    f"{s_stats.get('sharpe', np.nan):.2f}"
                    if np.isfinite(s_stats.get("sharpe", np.nan))
                    else "—"
                ),
            )
        with k4:
            st.metric(
                "Bench Total",
                (
                    f"{b_stats.get('total_return', np.nan):.1%}"
                    if np.isfinite(b_stats.get("total_return", np.nan))
                    else "—"
                ),
            )
        with k5:
            st.metric(
                "Bench CAGR",
                (
                    f"{b_stats.get('cagr', np.nan):.1%}"
                    if np.isfinite(b_stats.get("cagr", np.nan))
                    else "—"
                ),
            )
        with k6:
            avg_mo_tvr = float(tvr.resample("M").mean().mean()) if not tvr.empty else np.nan
            st.metric(
                "Avg Monthly Turnover", f"{avg_mo_tvr:.1%}" if np.isfinite(avg_mo_tvr) else "—"
            )

    # chart (optionally normalized for readability)
    if not PLOTLY_OK:
        st.warning("Plotly not installed — `pip install plotly` for charts.")
    else:
        port_curve_net = (
            normalize_to_one(port_curve_net_raw) if normalize_plot else port_curve_net_raw
        )
        port_curve_gross = (
            normalize_to_one(port_curve_gross_raw) if normalize_plot else port_curve_gross_raw
        )
        bench_curve = normalize_to_one(bench_curve_raw) if normalize_plot else bench_curve_raw

        plot_df = pd.DataFrame(
            {
                "Portfolio (Net)": port_curve_net,
                "Portfolio (Gross)": port_curve_gross.reindex(port_curve_net.index),
                "Benchmark": bench_curve.reindex(port_curve_net.index),
            }
        ).dropna(how="all")
        fig = go.Figure()
        if plot_df["Portfolio (Net)"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["Portfolio (Net)"],
                    name="Portfolio (Net)",
                    mode="lines",
                )
            )
        if plot_df["Portfolio (Gross)"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["Portfolio (Gross)"],
                    name="Portfolio (Gross)",
                    mode="lines",
                    opacity=0.55,
                )
            )
        if plot_df["Benchmark"].notna().any():
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["Benchmark"], name="Benchmark", mode="lines")
            )
        ttl = f"Confidence ≥ {thr:.2f} — Portfolio vs Benchmark" + (
            " (normalized)" if normalize_plot else ""
        )
        fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🗺️ Weights heatmap (last ~120 days)"):
        if PLOTLY_OK and not W.empty:
            W_hm = W.tail(120)
            hm = go.Figure(
                data=go.Heatmap(
                    z=W_hm.T.values,
                    x=W_hm.index.astype(str),
                    y=W_hm.columns.astype(str),
                    colorbar=dict(title="Weight"),
                )
            )
            hm.update_layout(
                title="Daily Weights Heatmap", xaxis_title="Date", yaxis_title="Ticker"
            )
            st.plotly_chart(hm, use_container_width=True)
        else:
            st.caption("No weights to display.")

    with st.expander("⬇️ Export data"):
        curves = pd.DataFrame(
            {
                "date": port_curve_net_raw.index,
                "portfolio_net": port_curve_net_raw.values,
                "portfolio_gross": port_curve_gross_raw.reindex(port_curve_net_raw.index).values,
                "benchmark": bench_curve_raw.reindex(port_curve_net_raw.index).values,
            }
        ).set_index("date")
        daily = pd.DataFrame(
            {
                "date": port_ret_gross.index,
                "gross_ret": port_ret_gross.values,
                "net_ret": port_ret_net.values,
                "bench_ret": bench_ret.values,
                "turnover": tvr.reindex(port_ret_gross.index).values,
                "cost_applied": (port_ret_net - port_ret_gross)
                .reindex(port_ret_gross.index)
                .values,
            }
        ).set_index("date")
        st.download_button(
            "Download equity curves (CSV)",
            curves.to_csv().encode("utf-8"),
            "tab28_curves.csv",
            "text/csv",
            key="t28_dl_curves",
        )
        if not daily.empty:
            st.download_button(
                "Download daily returns (CSV)",
                daily.to_csv().encode("utf-8"),
                "tab28_daily.csv",
                "text/csv",
                key="t28_dl_daily",
            )


# ──────────────────────────────
# Tab 29 — Confidence × Sharpe Portfolio vs Benchmark (tight, raw-KPI)
# ──────────────────────────────
with tabs[29]:
    st.subheader("📊 Confidence × Sharpe Portfolio vs Benchmark")

    # helpers (guarded)
    if "ema_smooth_wide" not in globals():

        def ema_smooth_wide(W: pd.DataFrame, span: int) -> pd.DataFrame:
            if W is None or W.empty or span <= 1:
                return W
            return W.ewm(span=span, adjust=False, min_periods=1).mean()

    if "enforce_min_hold" not in globals():

        def enforce_min_hold(mask_df: pd.DataFrame, min_days: int) -> pd.DataFrame:
            if mask_df is None or mask_df.empty or min_days <= 1:
                return mask_df
            out = mask_df.copy()
            for k in range(1, min_days):
                out |= mask_df.shift(k).fillna(False)
            return out

    def _daily_turnover_from_series(weights_long: pd.Series) -> pd.Series:
        if weights_long is None or weights_long.empty:
            return pd.Series(dtype=float)
        W = weights_long.unstack("ticker").fillna(0.0).sort_index()
        dW = W.diff().abs()
        return 0.5 * dW.sum(axis=1)

    # data
    svs = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if svs.empty:
        st.info("No strategy_vs_market.csv yet.")
        st.stop()

    parse_dates_inplace(svs, ("date",))
    svs = svs.dropna(subset=["date", "ticker"]).copy().sort_values(["ticker", "date"])

    has_ret = {"strategy_return", "market_return"}.issubset(svs.columns)
    has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(svs.columns)
    if has_ret and not has_cum:
        svs["strategy_return"] = pd.to_numeric(svs["strategy_return"], errors="coerce").fillna(0)
        svs["market_return"] = pd.to_numeric(svs["market_return"], errors="coerce").fillna(0)
        svs["cumulative_strategy"] = svs.groupby("ticker")["strategy_return"].apply(
            lambda s: (1 + s).cumprod()
        )
        svs["cumulative_market"] = svs.groupby("ticker")["market_return"].apply(
            lambda s: (1 + s).cumprod()
        )
    else:
        to_numeric(svs, ["cumulative_strategy", "cumulative_market"])

    panel = (
        svs[["date", "ticker", "cumulative_strategy", "cumulative_market"]]
        .set_index(["date", "ticker"])
        .sort_index()
    )
    panel.index = panel.index.set_names(["date", "ticker"])
    panel["strat_ret"] = panel.groupby(level=1)["cumulative_strategy"].pct_change().fillna(0.0)
    panel["mkt_ret"] = panel.groupby(level=1)["cumulative_market"].pct_change().fillna(0.0)
    panel["fwd_strat_ret"] = panel.groupby(level=1)["strat_ret"].shift(-1)
    panel["fwd_mkt_ret"] = panel.groupby(level=1)["mkt_ret"].shift(-1)

    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig.empty:
        sig = load_csv("signals.csv", RESULTS_DIR)
    if not sig.empty:
        sig = ensure_date(
            sig, ["date", "as_of", "timestamp", "time", "datetime", "Date"], normalize=False
        )
        sig = sig.dropna(subset=["date", "ticker"])
        to_numeric(sig, ["confidence"])
        conf = sig.set_index(["date", "ticker"])["confidence"].sort_index()
    else:
        conf = pd.Series(np.nan, index=panel.index)

    # UI
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        lookback = st.slider("Sharpe lookback (days)", 10, 120, 30, 5, key="t29_lb")
    with c2:
        max_w_pct = st.slider("Max weight per ticker (%)", 1, 50, 15, 1, key="t29_cap")
    with c3:
        cost_bps = st.slider("Trading cost (bps per $ traded)", 0, 50, 5, 1, key="t29_cost") / 1e4

    c4, c5 = st.columns([1.2, 1.2])
    with c4:
        return_source = st.selectbox(
            "Return family used for Sharpe & PnL",
            ["Market returns", "Strategy returns"],
            0,
            key="t29_source",
        )
    with c5:
        bench_choice = st.selectbox(
            "Benchmark", ["Avg Market (across tickers)", "SPY (ticker)"], 0, key="t29_bench"
        )

    ema_span = st.slider("Weight smoothing (EMA days)", 1, 10, 1, 1, key="t29_ema")
    min_hold = st.slider("Min hold days (soft)", 1, 10, 1, 1, key="t29_hold")
    normalize_plot = st.checkbox("Normalize curves for chart", True, key="t29_norm")
    show_kpis = st.checkbox("Show KPIs", True, key="t29_kpis")

    base_ret = panel["mkt_ret"] if return_source.startswith("Market") else panel["strat_ret"]
    base_fwd_ret = (
        panel["fwd_mkt_ret"] if return_source.startswith("Market") else panel["fwd_strat_ret"]
    )

    r_mat = base_ret.unstack("ticker")
    # robust rolling (min periods avoids NaN walls)
    minp = max(5, lookback // 3)
    mu = r_mat.rolling(lookback, min_periods=minp).mean()
    sd = r_mat.rolling(lookback, min_periods=minp).std()
    sharpe_mat = (mu / sd.replace(0, np.nan)) * np.sqrt(252)
    sharpe_mat = sharpe_mat.clip(lower=0.0).shift(1).fillna(0.0)

    sharpe = sharpe_mat.stack(dropna=False).fillna(0.0)
    sharpe.index.set_names(panel.index.names, inplace=True)

    c_mat = conf.reindex(panel.index).fillna(0.0).unstack("ticker")
    # min-max scale per day to [0,1]
    c_min = c_mat.min(axis=1)
    c_span = (c_mat.max(axis=1) - c_min).replace(0, np.nan)
    conf01 = (c_mat.sub(c_min, axis=0)).div(c_span, axis=0).fillna(0.0).stack(dropna=False)
    conf01.index.set_names(panel.index.names, inplace=True)

    # raw weights = Sharpe × scaled confidence
    raw_w = (sharpe * conf01).clip(lower=0.0)
    cap = max_w_pct / 100.0
    raw_w = raw_w.clip(upper=cap)

    # smooth & enforce hold
    w_mat = raw_w.unstack("ticker").fillna(0.0)
    w_mat = ema_smooth_wide(w_mat, ema_span)
    if min_hold > 1 and not w_mat.empty:
        pos = w_mat.gt(0)
        pos = enforce_min_hold(pos, min_hold)
        w_mat = w_mat.where(pos, 0.0)

    # cap & row-normalize, with equal-weight fallback
    w_mat = w_mat.clip(lower=0.0, upper=cap)
    row_sums = w_mat.sum(axis=1)
    w_norm = pd.DataFrame(0.0, index=w_mat.index, columns=w_mat.columns)
    pos_rows = row_sums > 0
    if pos_rows.any():
        w_norm.loc[pos_rows] = w_mat.loc[pos_rows].div(row_sums.loc[pos_rows], axis=0)
    if (~pos_rows).any():
        w0 = w_mat.loc[~pos_rows]

        def _eq_row(r):
            n = (r > 0).sum()
            return (
                pd.Series(np.where(r > 0, 1.0 / n, 0.0), index=r.index)
                if n > 0
                else pd.Series(0.0, index=r.index)
            )

        w_norm.loc[~pos_rows] = w0.apply(_eq_row, axis=1).fillna(0.0)

    weights = w_norm.stack(dropna=False).fillna(0.0)
    weights.index.set_names(panel.index.names, inplace=True)

    port_ret_gross = (
        (weights * base_fwd_ret.reindex(weights.index).fillna(0.0)).groupby(level=0).sum()
    )
    tvr = _daily_turnover_from_series(weights)
    cost = cost_bps * tvr.reindex(port_ret_gross.index).fillna(0.0)
    port_ret_net = port_ret_gross - cost

    # benchmark
    avg_mkt_ret = panel["mkt_ret"].groupby(level=0).mean()
    dates_index = port_ret_net.index
    tickers_unique = svs.get("ticker", pd.Series(dtype=str)).astype(str).unique()
    if bench_choice.startswith("SPY") and ("SPY" in tickers_unique):
        spy = svs[svs["ticker"] == "SPY"].copy().sort_values("date")
        if has_ret and not has_cum:
            spy["bench_ret"] = pd.to_numeric(spy["market_return"], errors="coerce").fillna(0)
        else:
            to_numeric(spy, ["cumulative_market"])
            spy["bench_ret"] = spy["cumulative_market"].pct_change().fillna(0)
        bench_ret = spy.set_index("date")["bench_ret"].reindex(dates_index).fillna(0)
    else:
        bench_ret = avg_mkt_ret.reindex(dates_index).fillna(0)

    # RAW curves for KPIs
    port_curve_net_raw = (1 + port_ret_net.fillna(0)).cumprod()
    port_curve_gross_raw = (1 + port_ret_gross.fillna(0)).cumprod()
    bench_curve_raw = (1 + bench_ret.fillna(0)).cumprod()

    # KPIs from raw curves
    if show_kpis:
        s_stats = perf_stats_from_levels(port_curve_net_raw.dropna())
        b_stats = perf_stats_from_levels(bench_curve_raw.dropna())
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.metric(
                "Net Total",
                (
                    f"{s_stats.get('total_return', np.nan):.1%}"
                    if np.isfinite(s_stats.get("total_return", np.nan))
                    else "—"
                ),
            )
        with k2:
            st.metric(
                "Net CAGR",
                (
                    f"{s_stats.get('cagr', np.nan):.1%}"
                    if np.isfinite(s_stats.get("cagr", np.nan))
                    else "—"
                ),
            )
        with k3:
            st.metric(
                "Net Sharpe",
                (
                    f"{s_stats.get('sharpe', np.nan):.2f}"
                    if np.isfinite(s_stats.get("sharpe", np.nan))
                    else "—"
                ),
            )
        with k4:
            st.metric(
                "Bench Total",
                (
                    f"{b_stats.get('total_return', np.nan):.1%}"
                    if np.isfinite(b_stats.get("total_return", np.nan))
                    else "—"
                ),
            )
        with k5:
            st.metric(
                "Bench CAGR",
                (
                    f"{b_stats.get('cagr', np.nan):.1%}"
                    if np.isfinite(b_stats.get("cagr", np.nan))
                    else "—"
                ),
            )
        with k6:
            avg_mo_tvr = float(tvr.resample("M").mean().mean()) if not tvr.empty else np.nan
            st.metric(
                "Avg Monthly Turnover", f"{avg_mo_tvr:.1%}" if np.isfinite(avg_mo_tvr) else "—"
            )

    # chart (normalize for readability only)
    if not PLOTLY_OK:
        st.warning("Plotly not installed — `pip install plotly` for charts.")
    else:
        port_curve_net = (
            normalize_to_one(port_curve_net_raw) if normalize_plot else port_curve_net_raw
        )
        port_curve_gross = (
            normalize_to_one(port_curve_gross_raw) if normalize_plot else port_curve_gross_raw
        )
        bench_curve = normalize_to_one(bench_curve_raw) if normalize_plot else bench_curve_raw

        plot_df = pd.DataFrame(
            {
                "Portfolio (Net)": port_curve_net,
                "Portfolio (Gross)": port_curve_gross.reindex(port_curve_net.index),
                "Benchmark": bench_curve.reindex(port_curve_net.index),
            }
        ).dropna(how="all")

        fig = go.Figure()
        if plot_df["Portfolio (Net)"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["Portfolio (Net)"],
                    name="Portfolio (Net)",
                    mode="lines",
                )
            )
        if plot_df["Portfolio (Gross)"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["Portfolio (Gross)"],
                    name="Portfolio (Gross)",
                    mode="lines",
                    opacity=0.55,
                )
            )
        if plot_df["Benchmark"].notna().any():
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["Benchmark"], name="Benchmark", mode="lines")
            )
        ttl = "Confidence × Sharpe Portfolio vs Benchmark" + (
            " (normalized)" if normalize_plot else ""
        )
        fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📎 Average weights by ticker (over period)"):
        w_avg = weights.groupby(level=1).mean().sort_values(ascending=False)
        st.dataframe(
            w_avg.reset_index().rename(columns={0: "avg_weight"}), use_container_width=True
        )

    with st.expander("🗺️ Weights heatmap (last ~120 days)"):
        if PLOTLY_OK and not weights.empty:
            W_hm = weights.unstack("ticker").fillna(0.0).tail(120)
            hm = go.Figure(
                data=go.Heatmap(
                    z=W_hm.T.values,
                    x=W_hm.index.astype(str),
                    y=W_hm.columns.astype(str),
                    colorbar=dict(title="Weight"),
                )
            )
            hm.update_layout(
                title="Daily Weights Heatmap", xaxis_title="Date", yaxis_title="Ticker"
            )
            st.plotly_chart(hm, use_container_width=True)
        else:
            st.caption("No weights to display.")

# ──────────────────────────────
# Tab 30 — Stress Test Reports + runner button + aggregated chart
# ──────────────────────────────
with tabs[30]:
    st.subheader("🧪 Stress Test Reports & Runner")
    st.caption(f"Looking for JSON results in: {STRESS_DIR}")

    # Runner: quick stress test via subprocess
    st.markdown("**Run quick stress tests**")
    col_run, col_run_notes = st.columns([1, 3])
    with col_run_notes:
        st.caption(
            "Runs `services/stress_test.py --quick` in a subprocess. May take a few minutes."
        )
    run_now = col_run.button("▶️ Run quick stress tests (quick)")

    if run_now:
        runner_path = PROJECT_ROOT / "run_stress_test.py"
        services_runner = PROJECT_ROOT / "services" / "stress_test.py"

        if runner_path.exists():
            cmd = [sys.executable, str(runner_path), "--quick"]
        elif services_runner.exists():
            cmd = [sys.executable, str(services_runner), "--quick"]
        else:
            cmd = None

        if not cmd:
            st.error(
                "Runner not found. Expected `run_stress_test.py` or `services/stress_test.py`."
            )
        else:
            st.info(f"Running: {' '.join(cmd)}")
            try:
                with st.spinner("Running stress tests (quick)…"):
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                out = proc.stdout or ""
                err = proc.stderr or ""
                status_code = proc.returncode

                st.subheader("Runner output")
                if out.strip():
                    st.markdown("**stdout**")
                    st.code(out.strip())
                if err.strip():
                    st.markdown("**stderr**")
                    st.code(err.strip())

                if status_code == 0:
                    st.success(
                        "Runner finished with exit code 0. Refresh the list below to see new results."
                    )
                else:
                    st.error(f"Runner exited with code {status_code}. Check stderr above.")
            except Exception as e:
                st.error(f"Failed to run stress test runner: {e}")

    # Discover result files
    files = sorted(STRESS_DIR.glob("stress_test_results_*.json"))
    if not files:
        st.info(
            "No stress test result files found in stress_test_results/. Run the quick test above."
        )
    else:
        # Parse + flatten rows
        rows = []
        for p in files:
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                rows.append(
                    {
                        "file": p.name,
                        "file_path": str(p),
                        "scenario": "(invalid JSON)",
                        "test": "(parse_error)",
                        "survived": False,
                        "total_return": np.nan,
                        "max_drawdown": np.nan,
                        "final_value": np.nan,
                        "volatility": np.nan,
                        "notes": f"JSON parse error: {e}",
                    }
                )
                continue

            for scenario_name, scenario_results in j.items():
                if isinstance(scenario_results, dict):
                    for test_name, metrics in scenario_results.items():
                        if not isinstance(metrics, dict):
                            rows.append(
                                {
                                    "file": p.name,
                                    "file_path": str(p),
                                    "scenario": scenario_name,
                                    "test": test_name,
                                    "survived": False,
                                    "total_return": np.nan,
                                    "max_drawdown": np.nan,
                                    "final_value": np.nan,
                                    "volatility": np.nan,
                                    "notes": "Unexpected metrics format",
                                }
                            )
                            continue
                        rows.append(
                            {
                                "file": p.name,
                                "file_path": str(p),
                                "scenario": scenario_name,
                                "test": test_name,
                                "survived": bool(metrics.get("survived", False)),
                                "total_return": (
                                    float(metrics.get("total_return", np.nan))
                                    if metrics.get("total_return") is not None
                                    else np.nan
                                ),
                                "max_drawdown": (
                                    float(metrics.get("max_drawdown", np.nan))
                                    if metrics.get("max_drawdown") is not None
                                    else np.nan
                                ),
                                "final_value": (
                                    float(metrics.get("final_value", np.nan))
                                    if metrics.get("final_value") is not None
                                    else np.nan
                                ),
                                "volatility": (
                                    float(metrics.get("volatility", np.nan))
                                    if metrics.get("volatility") is not None
                                    else np.nan
                                ),
                                "notes": "",
                            }
                        )
                else:
                    rows.append(
                        {
                            "file": p.name,
                            "file_path": str(p),
                            "scenario": scenario_name,
                            "test": "(unexpected format)",
                            "survived": False,
                            "total_return": np.nan,
                            "max_drawdown": np.nan,
                            "final_value": np.nan,
                            "volatility": np.nan,
                            "notes": "Scenario entry is not a dict",
                        }
                    )

        summary_df = pd.DataFrame(rows)
        if summary_df.empty:
            st.info("Stress result files were found but no usable rows could be parsed.")
        else:
            # KPIs
            total_runs = int(summary_df["file"].nunique())
            total_tests = int(len(summary_df))
            passed = int(summary_df["survived"].sum())
            failed = total_tests - passed
            success_rate = passed / total_tests if total_tests else 0.0

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Result files", total_runs)
            with k2:
                st.metric("Total tests", total_tests)
            with k3:
                st.metric("Passed", passed)
            with k4:
                st.metric("Success rate", f"{success_rate:.1%}")

            # By-scenario aggregation
            agg = (
                summary_df.groupby(["scenario"])["survived"]
                .agg(["count", "sum"])
                .rename(columns={"count": "tests", "sum": "passed"})
                .reset_index()
            )
            agg["failed"] = agg["tests"] - agg["passed"]
            st.markdown("**Scenario summary**")
            st.dataframe(
                agg.sort_values("failed", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )

            # Chart
            st.markdown("**Aggregated visualization**")
            agg_chart = agg.sort_values("tests", ascending=False).reset_index(drop=True)
            if PLOTLY_OK:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=agg_chart["scenario"], y=agg_chart["passed"], name="Passed"))
                fig.add_trace(go.Bar(x=agg_chart["scenario"], y=agg_chart["failed"], name="Failed"))
                fig.update_layout(
                    barmode="stack",
                    title="Passed / Failed counts by Scenario",
                    xaxis_title="Scenario",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(agg_chart.set_index("scenario")[["passed", "failed"]])

            # Main table + filters
            st.markdown("**All test rows — filter and explore**")
            filt_scenarios = sorted(summary_df["scenario"].dropna().unique())
            sel_scenario = st.selectbox(
                "Filter scenario (optional)", ["(all)"] + filt_scenarios, key="t30_scen"
            )
            sel_file = st.selectbox(
                "Select a result file to inspect",
                ["(none)"] + sorted(summary_df["file"].unique()),
                key="t30_file",
            )

            df_view = summary_df.copy()
            if sel_scenario and sel_scenario != "(all)":
                df_view = df_view[df_view["scenario"] == sel_scenario]

            st.dataframe(
                df_view[
                    [
                        "file",
                        "scenario",
                        "test",
                        "survived",
                        "total_return",
                        "max_drawdown",
                        "final_value",
                        "volatility",
                        "notes",
                    ]
                ],
                use_container_width=True,
            )

            # Per-file inspector
            st.markdown("**Inspect selected file**")
            if sel_file and sel_file != "(none)":
                sel_path = STRESS_DIR / sel_file
                try:
                    raw_text = sel_path.read_text(encoding="utf-8")
                    sel_json = json.loads(raw_text)
                    st.write(f"**File:** {sel_file}")
                    with st.expander("Show raw JSON"):
                        st.json(sel_json)

                    rows_f = [r for r in rows if r["file"] == sel_file]
                    df_f = pd.DataFrame(rows_f)
                    if not df_f.empty:
                        st.markdown("**Selected run — test breakdown**")
                        st.dataframe(
                            df_f[
                                [
                                    "scenario",
                                    "test",
                                    "survived",
                                    "total_return",
                                    "max_drawdown",
                                    "final_value",
                                    "volatility",
                                ]
                            ],
                            use_container_width=True,
                        )

                    st.download_button(
                        "⬇️ Download raw JSON",
                        data=raw_text.encode("utf-8"),
                        file_name=sel_file,
                        mime="application/json",
                        key="t30_dl_json",
                    )
                except Exception as e:
                    st.error(f"Could not read selected file: {e}")

            # Combined export
            st.download_button(
                "⬇️ Download combined summary (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="stress_tests_summary.csv",
                mime="text/csv",
                key="t30_dl_combined",
            )

            with st.expander("🛈 Tips & quick actions"):
                st.write(
                    "- Files should be named `stress_test_results_YYYYMMDD_HHMMSS.json`.\n"
                    "- If a row shows `survived: false`, drill into that file's JSON.\n"
                    "- Re-run quick tests above to generate new results.\n"
                    "- To run a *full* suite from Streamlit, change the runner to omit `--quick` (may be slow)."
                )

# ──────────────────────────────
# Tab 31 — Market Sentinels
# ──────────────────────────────
with tabs[31]:
    st.subheader("🩺 Market Sentinels (quick market health)")

    ms = load_csv("market_sentinels.csv", RESULTS_DIR)
    if ms.empty:
        st.info("No market_sentinels.csv yet. Run: python services/market_sentinels.py")
        st.stop()

    # Coerce numeric columns
    for c in ["last_close", "ret_5d", "ret_20d", "vol_20d", "rsi_14"]:
        if c in ms.columns:
            ms[c] = pd.to_numeric(ms[c], errors="coerce")

    # Traffic-light summary
    def _status_row(r):
        badges = []
        v = r.get("ret_5d", np.nan)
        if pd.notna(v):
            badges.append("🟢" if v > 0 else "🔴" if v < 0 else "🟡")
        rsi = r.get("rsi_14", np.nan)
        if pd.notna(rsi):
            badges.append("🟢" if 45 <= rsi <= 65 else "🔴" if (rsi < 30 or rsi > 70) else "🟡")
        ma_ok = r.get("ma_20_above_ma_50", None)
        if isinstance(ma_ok, bool):
            badges.append("🟢" if ma_ok else "🔴")
        return " ".join(badges)

    ms_disp = ms.copy()
    ms_disp["status"] = ms_disp.apply(_status_row, axis=1)

    c1, c2 = st.columns([1.25, 1])
    with c1:
        cols_to_show = [
            c
            for c in [
                "symbol",
                "last_date",
                "last_close",
                "ret_5d",
                "ret_20d",
                "vol_20d",
                "rsi_14",
                "ma_20_above_ma_50",
                "status",
            ]
            if c in ms_disp.columns
        ]
        st.dataframe(ms_disp[cols_to_show], use_container_width=True)

    with c2:

        def metric_for(sym, label, col, fmt=None):
            row = ms[ms["symbol"] == sym]
            if not row.empty and col in row.columns and pd.notna(row.iloc[0][col]):
                val = row.iloc[0][col]
                st.metric(f"{label} — {sym}", fmt.format(val) if fmt else f"{val:,.2f}")
            else:
                st.metric(f"{label} — {sym}", "—")

        metric_for("SPY", "5D Return", "ret_5d", fmt="{:.2%}")
        metric_for("SPY", "20D Return", "ret_20d", fmt="{:.2%}")
        metric_for("^VIX", "Last Close", "last_close")

    st.download_button(
        "⬇️ Download sentinels (CSV)",
        data=ms.to_csv(index=False).encode("utf-8"),
        file_name="market_sentinels.csv",
        mime="text/csv",
        key="t31_dl",
    )


# ──────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────
def _sidebar_controls():
    """
    Returns: (section_choice, sub_choice)
    Sets PROJECT_ROOT / DATA_ROOT / RESULTS_DIR / ORDERS_DIR / PRED_DIR / STRESS_DIR.
    """
    global PROJECT_ROOT, DATA_ROOT, RESULTS_DIR, ORDERS_DIR, PRED_DIR, STRESS_DIR

    with st.sidebar:
        st.title("TRITON Nav")

        # Advanced / project root selector
        st.subheader("⚙️ Advanced")
        root_mode = st.radio(
            "Project root",
            ["Auto (this file’s folder)", "Manual path"],
            index=0,
            key="sb_root_mode",  # unique
        )

        if root_mode == "Manual path":
            default_text = str(st.session_state.get("custom_root", DEFAULT_PROJECT_ROOT))
            custom_root = st.text_input(
                "Enter absolute path to repo root",
                value=default_text,
                key="sb_root_manual_input",  # unique
            )
            try:
                project_root_candidate = Path(custom_root).expanduser().resolve()
                st.session_state["custom_root"] = str(project_root_candidate)
                PROJECT_ROOT = project_root_candidate
                st.caption(f"Using custom root: {PROJECT_ROOT}")
            except Exception as e:
                st.error(f"Invalid custom root: {e}")
                PROJECT_ROOT = DEFAULT_PROJECT_ROOT
                st.caption(f"Using auto root (fallback): {PROJECT_ROOT}")
        else:
            PROJECT_ROOT = DEFAULT_PROJECT_ROOT
            st.caption(f"Using auto root: {PROJECT_ROOT}")

        st.caption(f"Build: {APP_VERSION}")

        # Ensure canonical dirs exist
        DATA_ROOT = PROJECT_ROOT / "data"
        RESULTS_DIR = DATA_ROOT / "results"
        ORDERS_DIR = DATA_ROOT / "orders"
        PRED_DIR = DATA_ROOT / "predictions"
        STRESS_DIR = DATA_ROOT / "stress_test_results"
        for p in (RESULTS_DIR, ORDERS_DIR, PRED_DIR, STRESS_DIR):
            p.mkdir(parents=True, exist_ok=True)

        # Section / subpage navigation
        section_choice = st.selectbox(
            "Section",
            list(SECTIONS.keys()),
            index=0,
            help="High-level area of Triton",
            key="sb_section",  # unique
        )

        subpages = SECTIONS[section_choice]
        sub_choice = st.radio(
            "View",
            subpages,
            index=0,
            help="Which view inside that section",
            key="sb_subpage",  # unique
        )

    return section_choice, sub_choice


# ──────────────────────────────
# APP ENTRYPOINT
# ──────────────────────────────
def _render_main():
    section_choice, sub_choice = _sidebar_controls()
    render_global_header()
    render_body(section_choice, sub_choice)


_render_main()
