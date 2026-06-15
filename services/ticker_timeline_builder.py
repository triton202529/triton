"""
TRITON Phase 166 — Ticker Decision Timeline Builder.

Reconstructs a chronological BUY / ADD / HOLD / TRIM / EXIT decision timeline
for every ticker by merging:

  - the canonical fill ledger (actual executions)        -> BUY/ADD/TRIM/EXIT
  - paper_execution_audit.csv (decision-time context)    -> lifecycle/risk flags
  - signals_lifecycle.csv (daily lifecycle history)      -> HOLD/WAIT records
  - signal_lifecycle_effective.csv (current snapshot)    -> latest state
  - trade_outcomes.csv (realized P&L per closed trade)
  - processed price parquets (decision grading, post-exit analysis)

Observability only — no trading, execution, or lifecycle logic is touched.

Outputs:
    data/results/ticker_decision_timelines.parquet
    data/results/ticker_decision_timelines.csv

Run:
    python -m services.ticker_timeline_builder
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.pnl_ledger_builder import (
    _load_fill_ledger,
    _load_price_series,
    _norm_sym,
    _safe_read_csv,
    _safe_read_json,
    _to_float,
)

RESULTS = ROOT / "data" / "results"
OUT_PARQUET = RESULTS / "ticker_decision_timelines.parquet"
OUT_CSV = RESULTS / "ticker_decision_timelines.csv"

TIMELINE_COLUMNS = [
    "timestamp",
    "ticker",
    "action",
    "signal",
    "lifecycle_state",
    "effective_lifecycle",
    "confidence",
    "expected_return",
    "risk_flags",
    "rationale",
    "position_size",
    "market_value",
    "unrealized_pnl",
    "realized_pnl",
    "execution_session",
    "price",
    "qty",
    "event_source",
    "decision_grade",
    "forward_return_5d",
    "post_exit_return_5d",
    "post_exit_return_10d",
    "post_exit_return_20d",
    "exit_outcome",
]

_CACHE: Dict[str, Any] = {"fingerprint": None, "df": None}

_SOURCE_FILES = [
    RESULTS / "live_orders_log.csv",
    RESULTS / "recent_orders.csv",
    RESULTS / "paper_execution_audit.csv",
    RESULTS / "signals_lifecycle.csv",
    RESULTS / "signal_lifecycle_effective.csv",
    RESULTS / "signal_lifecycle.csv",
    RESULTS / "signals_with_rationale.csv",
    RESULTS / "trade_outcomes.csv",
    RESULTS / "trade_log.csv",
]


def _log(msg: str) -> None:
    print(f"[ticker_timeline_builder] {msg}", flush=True)


def _fingerprint() -> str:
    parts = []
    for p in _SOURCE_FILES:
        try:
            parts.append(f"{p.name}:{p.stat().st_mtime_ns}")
        except OSError:
            parts.append(f"{p.name}:absent")
    return "|".join(parts)


def empty_timeline() -> pd.DataFrame:
    return pd.DataFrame(columns=TIMELINE_COLUMNS)


# ──────────────────────────────────────────────
# Context tables
# ──────────────────────────────────────────────
def _load_audit_context() -> pd.DataFrame:
    """Decision-time context per (symbol, session) from the execution audit."""
    a = _safe_read_csv(RESULTS / "paper_execution_audit.csv")
    if a.empty:
        return pd.DataFrame()
    a.columns = [str(c).strip().lower() for c in a.columns]
    a["symbol"] = a.get("symbol", pd.Series(dtype=str)).map(_norm_sym)
    a["ts"] = pd.to_datetime(a.get("timestamp"), errors="coerce", utc=True)
    return a.dropna(subset=["ts"])


def _audit_context_for(
    audit: pd.DataFrame, symbol: str, session: str, ts: pd.Timestamp
) -> Dict[str, str]:
    """Nearest audit row for a fill: same symbol+session preferred, else nearest in time."""
    if audit.empty:
        return {}
    rows = audit[(audit["symbol"] == symbol) & (audit.get("session") == session)]
    if rows.empty:
        rows = audit[audit["symbol"] == symbol]
        if rows.empty or pd.isna(ts):
            return {}
        rows = rows.iloc[(rows["ts"] - ts).abs().argsort()[:1]]
    r = rows.iloc[-1]
    flags = []
    for col in ("skip_reason", "duplicate_check_result"):
        v = str(r.get(col) or "").strip()
        if v and v.lower() not in ("nan", "ok", "none", "passed"):
            flags.append(v)
    return {
        "lifecycle_state": str(r.get("lifecycle_state") or "").strip(),
        "signal": str(r.get("signal") or "").strip(),
        "risk_flags": "; ".join(flags),
    }


def _load_current_snapshot() -> pd.DataFrame:
    snap = _safe_read_csv(RESULTS / "signal_lifecycle_effective.csv")
    if snap.empty:
        return pd.DataFrame()
    snap.columns = [str(c).strip().lower() for c in snap.columns]
    snap["ticker"] = snap.get("ticker", pd.Series(dtype=str)).map(_norm_sym)
    return snap


def _outcome_realized_map() -> Dict[str, List[Dict[str, Any]]]:
    """Closed trades per symbol from trade_outcomes.csv (realized P&L attribution)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    t = _safe_read_csv(RESULTS / "trade_outcomes.csv")
    if t.empty:
        return out
    t.columns = [str(c).strip().lower() for c in t.columns]
    t = t[t.get("outcome_type", "").astype(str).str.upper() == "REALIZED"]
    for _, r in t.iterrows():
        sym = _norm_sym(r.get("symbol"))
        closed = pd.to_datetime(r.get("closed_at"), errors="coerce", utc=True)
        out.setdefault(sym, []).append(
            {
                "closed_at": closed,
                "realized_pl": _to_float(r.get("realized_pl"), 0.0),
                "qty": _to_float(r.get("qty"), np.nan),
            }
        )
    return out


def _watchdog_flags() -> str:
    """Global risk flags from risk_watchdog.json / adaptive risk state (optional)."""
    flags: List[str] = []
    wd = _safe_read_json(RESULTS / "risk_watchdog.json")
    if isinstance(wd, dict):
        for k in ("status", "alert", "mode"):
            v = str(wd.get(k) or "").strip()
            if v and v.upper() not in ("OK", "NORMAL", "NONE"):
                flags.append(f"watchdog:{v}")
    ars = _safe_read_json(RESULTS / "adaptive_risk_state.json")
    if isinstance(ars, dict):
        mode = str(ars.get("mode") or "").strip()
        if mode and mode.upper() not in ("NORMAL", "OK"):
            flags.append(f"risk_mode:{mode}")
    return "; ".join(flags)


# ──────────────────────────────────────────────
# Event extraction
# ──────────────────────────────────────────────
def _events_from_fills(
    fills: pd.DataFrame,
    audit: pd.DataFrame,
    realized_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Actual executions become timeline events. FIFO running inventory decides
    BUY vs ADD (buys) and TRIM vs EXIT (sells), and yields per-event realized P&L.
    """
    events: List[Dict[str, Any]] = []
    if fills.empty:
        return events

    for sym, grp in fills.groupby("symbol"):
        grp = grp.sort_values("ts")
        lots: List[List[float]] = []  # [qty, price]
        for _, r in grp.iterrows():
            qty = _to_float(r.get("qty"), 0.0)
            px = _to_float(r.get("price"), np.nan)
            session = str(r.get("session") or "")
            ts = r.get("ts")
            open_qty = sum(l[0] for l in lots)
            realized = np.nan

            if str(r.get("side")) == "buy":
                action = "BUY" if open_qty <= 1e-9 else "ADD"
                lots.append([qty, px])
            else:
                realized = 0.0
                left = qty
                while left > 1e-9 and lots:
                    lot = lots[0]
                    take = min(left, lot[0])
                    if np.isfinite(px):
                        realized += (px - lot[1]) * take
                    lot[0] -= take
                    left -= take
                    if lot[0] <= 1e-9:
                        lots.pop(0)
                action = "EXIT" if sum(l[0] for l in lots) <= 1e-9 else "TRIM"

            ctx = _audit_context_for(audit, sym, session, ts)
            net_qty = sum(l[0] for l in lots)
            events.append(
                {
                    "timestamp": ts,
                    "ticker": sym,
                    "action": action,
                    "signal": ctx.get("signal", ""),
                    "lifecycle_state": ctx.get("lifecycle_state", action),
                    "effective_lifecycle": "",
                    "confidence": np.nan,
                    "expected_return": np.nan,
                    "risk_flags": ctx.get("risk_flags", ""),
                    "rationale": "",
                    "position_size": net_qty,
                    "market_value": net_qty * px if np.isfinite(px) else np.nan,
                    "unrealized_pnl": np.nan,
                    "realized_pnl": realized,
                    "execution_session": session,
                    "price": px,
                    "qty": qty,
                    "event_source": "fill_ledger",
                }
            )
    return events


def _events_from_lifecycle_history() -> List[Dict[str, Any]]:
    """Daily lifecycle records (HOLD/ADD/WAIT/...) from signals_lifecycle.csv."""
    events: List[Dict[str, Any]] = []
    h = _safe_read_csv(RESULTS / "signals_lifecycle.csv")
    if h.empty:
        return events
    h.columns = [str(c).strip().lower() for c in h.columns]
    if "ticker" not in h.columns:
        return events
    h["ticker"] = h["ticker"].map(_norm_sym)
    h["ts"] = pd.to_datetime(h.get("date"), errors="coerce", utc=True)
    h = h.dropna(subset=["ts"])
    for _, r in h.iterrows():
        action = str(r.get("lifecycle_action") or "").strip().upper() or "HOLD"
        events.append(
            {
                "timestamp": r["ts"],
                "ticker": r["ticker"],
                "action": action,
                "signal": str(r.get("signal") or ""),
                "lifecycle_state": str(r.get("position_state") or ""),
                "effective_lifecycle": "",
                "confidence": _to_float(r.get("confidence"), np.nan),
                "expected_return": _to_float(
                    r.get("edge_pct"), _to_float(r.get("delta_pct"), np.nan)
                ),
                "risk_flags": "",
                "rationale": str(r.get("rationale") or ""),
                "position_size": np.nan,
                "market_value": np.nan,
                "unrealized_pnl": np.nan,
                "realized_pnl": np.nan,
                "execution_session": "",
                "price": _to_float(r.get("close"), np.nan),
                "qty": np.nan,
                "event_source": "lifecycle_history",
            }
        )
    return events


def _events_from_current_snapshot(snap: pd.DataFrame, global_flags: str) -> List[Dict[str, Any]]:
    """One 'current state' event per ticker from signal_lifecycle_effective.csv."""
    events: List[Dict[str, Any]] = []
    if snap.empty:
        return events
    for _, r in snap.iterrows():
        ts = pd.to_datetime(r.get("generated_at_utc"), errors="coerce", utc=True)
        if pd.isna(ts):
            ts = pd.to_datetime(r.get("date"), errors="coerce", utc=True)
        action = str(r.get("lifecycle_action") or "").strip().upper() or "HOLD"
        flags = []
        for col in ("lifecycle_decision_reason", "execution_block_reason", "heal_reason"):
            v = str(r.get(col) or "").strip()
            if v and v.lower() not in ("nan", "none", ""):
                flags.append(v)
        if global_flags:
            flags.append(global_flags)
        events.append(
            {
                "timestamp": ts,
                "ticker": r.get("ticker"),
                "action": action,
                "signal": str(r.get("signal") or ""),
                "lifecycle_state": str(r.get("position_state") or ""),
                "effective_lifecycle": str(
                    r.get("effective_position_state") or r.get("effective_stance") or ""
                ),
                "confidence": _to_float(r.get("confidence"), np.nan),
                "expected_return": _to_float(r.get("delta_pct"), np.nan),
                "risk_flags": "; ".join(flags),
                "rationale": str(r.get("rationale") or ""),
                "position_size": np.nan,
                "market_value": np.nan,
                "unrealized_pnl": np.nan,
                "realized_pnl": np.nan,
                "execution_session": "current_snapshot",
                "price": _to_float(r.get("close"), np.nan),
                "qty": np.nan,
                "event_source": "current_snapshot",
            }
        )
    return events


# ──────────────────────────────────────────────
# Decision grading & post-exit analysis
# ──────────────────────────────────────────────
def _forward_return(sym: str, ts: pd.Timestamp, days: int) -> float:
    prices = _load_price_series(sym)
    if prices.empty or pd.isna(ts):
        return np.nan
    base_date = pd.Timestamp(ts).tz_localize(None).normalize()
    after = prices[prices["date"] >= base_date]
    if after.empty:
        return np.nan
    start = float(after["close"].iloc[0])
    fwd = after[after["date"] >= base_date + pd.Timedelta(days=days)]
    end = float(fwd["close"].iloc[0]) if not fwd.empty else float(after["close"].iloc[-1])
    if start <= 0:
        return np.nan
    return (end - start) / start * 100.0


def _grade_from_return(action: str, fwd_ret_pct: float) -> str:
    """
    Entries (BUY/ADD/HOLD) are good when the price subsequently rises;
    exits (TRIM/EXIT) are good when the price subsequently falls.
    """
    if not np.isfinite(fwd_ret_pct):
        return ""
    aligned = fwd_ret_pct if action in ("BUY", "ADD", "HOLD") else -fwd_ret_pct
    if aligned >= 3.0:
        return "A"
    if aligned >= 1.0:
        return "B"
    if aligned >= -1.0:
        return "C"
    if aligned >= -3.0:
        return "D"
    return "F"


def _apply_grades_and_post_exit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grades: List[str] = []
    fwd5: List[float] = []
    pe5: List[float] = []
    pe10: List[float] = []
    pe20: List[float] = []
    outcome: List[str] = []

    for _, r in df.iterrows():
        action = str(r["action"])
        sym = str(r["ticker"])
        ts = r["timestamp"]
        f5 = (
            _forward_return(sym, ts, 5)
            if action in ("BUY", "ADD", "HOLD", "TRIM", "EXIT")
            else np.nan
        )
        fwd5.append(f5)
        grades.append(_grade_from_return(action, f5))

        if action == "EXIT":
            r5 = _forward_return(sym, ts, 5)
            r10 = _forward_return(sym, ts, 10)
            r20 = _forward_return(sym, ts, 20)
            pe5.append(r5)
            pe10.append(r10)
            pe20.append(r20)
            ref = r10 if np.isfinite(r10) else (r5 if np.isfinite(r5) else r20)
            if not np.isfinite(ref):
                outcome.append("NEUTRAL_EXIT")
            elif ref <= -2.0:
                outcome.append("CORRECT_EXIT")
            elif ref >= 5.0:
                outcome.append("EARLY_EXIT")
            else:
                # Late if the trade gave back a big run-up before exiting.
                late = False
                px = _to_float(r.get("price"), np.nan)
                prices = _load_price_series(sym)
                if np.isfinite(px) and not prices.empty and not pd.isna(ts):
                    base = pd.Timestamp(ts).tz_localize(None).normalize()
                    window = prices[
                        (prices["date"] >= base - pd.Timedelta(days=30)) & (prices["date"] <= base)
                    ]
                    if not window.empty and float(window["close"].max()) > px * 1.05:
                        late = True
                outcome.append("LATE_EXIT" if late else "NEUTRAL_EXIT")
        else:
            pe5.append(np.nan)
            pe10.append(np.nan)
            pe20.append(np.nan)
            outcome.append("")

    df = df.copy()
    df["decision_grade"] = grades
    df["forward_return_5d"] = fwd5
    df["post_exit_return_5d"] = pe5
    df["post_exit_return_10d"] = pe10
    df["post_exit_return_20d"] = pe20
    df["exit_outcome"] = outcome
    return df


# ──────────────────────────────────────────────
# Build + persist
# ──────────────────────────────────────────────
def build_ticker_timelines(force: bool = False) -> pd.DataFrame:
    fp = _fingerprint()
    if not force and _CACHE["fingerprint"] == fp and _CACHE["df"] is not None:
        return _CACHE["df"]

    fills = _load_fill_ledger()
    audit = _load_audit_context()
    realized_map = _outcome_realized_map()
    snap = _load_current_snapshot()
    global_flags = _watchdog_flags()

    events: List[Dict[str, Any]] = []
    events += _events_from_fills(fills, audit, realized_map)
    events += _events_from_lifecycle_history()
    events += _events_from_current_snapshot(snap, global_flags)

    df = pd.DataFrame(events)
    if df.empty:
        df = empty_timeline()
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp", "ticker"])
        df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        df = _apply_grades_and_post_exit(df)
        for c in TIMELINE_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[TIMELINE_COLUMNS]

    _CACHE["fingerprint"] = fp
    _CACHE["df"] = df
    return df


def write_outputs(df: pd.DataFrame) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if not out.empty:
        out["timestamp"] = out["timestamp"].astype(str)
    try:
        out.to_parquet(OUT_PARQUET, index=False)
        _log(f"wrote {OUT_PARQUET.name}: {len(out)} events")
    except Exception as e:
        _log(f"parquet write failed ({type(e).__name__}: {e}) — CSV still written")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    _log(f"wrote {OUT_CSV.name}: {len(out)} events")


def load_timelines() -> pd.DataFrame:
    """Dashboard entry point: build (cached) and return the full timeline frame."""
    try:
        return build_ticker_timelines()
    except Exception as e:
        _log(f"build failed: {type(e).__name__}: {e}")
        return empty_timeline()


if __name__ == "__main__":
    timeline = build_ticker_timelines(force=True)
    write_outputs(timeline)
    if not timeline.empty:
        _log(
            f"{timeline['ticker'].nunique()} tickers | "
            f"actions: {timeline['action'].value_counts().to_dict()}"
        )
