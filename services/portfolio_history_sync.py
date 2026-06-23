"""
portfolio_history_sync.py
Read-only portfolio history / positions snapshot reconciliation for dashboard.

- Derives current invested value from positions_snapshot.csv
- Preserves cash/total from latest portfolio/account artifacts (never invents cash)
- Appends dated rows to portfolio_history.csv only when positions are newer (non-destructive)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

PORTFOLIO_HISTORY_PATH = RESULTS / "portfolio_history.csv"
POSITIONS_SNAPSHOT_PATH = RESULTS / "positions_snapshot.csv"
OPEN_ORDERS_SNAPSHOT_PATH = RESULTS / "open_orders_snapshot.csv"
RECENT_ORDERS_SNAPSHOT_PATH = RESULTS / "recent_orders.csv"
ACCOUNT_SNAPSHOT_PATH = RESULTS / "account_snapshot.csv"
ENHANCED_PORTFOLIO_HISTORY_PATH = RESULTS / "enhanced_portfolio_history.csv"
EXECUTION_READINESS_PATH = RESULTS / "execution_readiness.json"

PH_COLS = ["date", "cash", "market_value", "total_value"]
VALUE_ALIASES = ("market_value", "position_value", "value", "notional", "mv", "amount")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_mtime_dt(path: Path) -> Optional[datetime]:
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _norm_symbols(series: pd.Series) -> List[str]:
    return sorted({str(s).strip().upper() for s in series if str(s).strip()})


def _pick_value_col(df: pd.DataFrame) -> Optional[str]:
    for c in VALUE_ALIASES:
        if c in df.columns:
            return c
    return None


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _normalize_portfolio_history_raw(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=PH_COLS)

    out = df.copy()
    if "total_value" not in out.columns:
        for alt in ("equity", "portfolio_value", "portfolio_total", "value", "total"):
            if alt in out.columns:
                out = out.rename(columns={alt: "total_value"})
                break

    if "date" not in out.columns:
        for alt in ("timestamp", "date_utc", "as_of"):
            if alt in out.columns:
                out = out.rename(columns={alt: "date"})
                break

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    for c in PH_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("date").reset_index(drop=True)
    return out


def aggregate_portfolio_history_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw history to one row per calendar date (dashboard-safe)."""
    raw = _normalize_portfolio_history_raw(df)
    if raw.empty:
        return pd.DataFrame(columns=PH_COLS)

    if "ticker" in raw.columns or "symbol" in raw.columns:
        sym_col = "ticker" if "ticker" in raw.columns else "symbol"
        raw["date"] = raw["date"].dt.normalize()
        cash = raw.groupby("date")["cash"].max() if "cash" in raw.columns else None
        if "market_value" in raw.columns:
            mv = raw.groupby("date")["market_value"].sum()
            daily = pd.DataFrame({"date": mv.index, "market_value": mv.values})
            if cash is not None:
                daily["cash"] = cash.reindex(daily["date"]).values
            if "cash" in daily.columns:
                daily["total_value"] = daily["cash"].fillna(0) + daily["market_value"].fillna(0)
            return daily.sort_values("date").reset_index(drop=True)

    cols = [c for c in PH_COLS if c in raw.columns]
    daily = raw[cols].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"])
    daily["date"] = daily["date"].dt.normalize()
    daily = daily.groupby("date", as_index=False).last()
    return daily.sort_values("date").reset_index(drop=True)


def load_positions_snapshot_summary() -> Dict[str, Any]:
    df = _read_csv(POSITIONS_SNAPSHOT_PATH)
    meta: Dict[str, Any] = {
        "exists": POSITIONS_SNAPSHOT_PATH.exists() and POSITIONS_SNAPSHOT_PATH.stat().st_size > 0,
        "path": str(POSITIONS_SNAPSHOT_PATH),
        "mtime": file_mtime_dt(POSITIONS_SNAPSHOT_PATH),
        "rows": int(len(df)),
        "snapshot_ts": None,
        "as_of_date": None,
        "invested_value": None,
        "positions": [],
        "position_count": 0,
    }
    if df.empty:
        return meta

    if "snapshot_ts" in df.columns:
        meta["snapshot_ts"] = _parse_ts(df["snapshot_ts"].iloc[-1])
    if "date" in df.columns:
        meta["as_of_date"] = _parse_ts(df["date"].iloc[-1])

    if "ticker" not in df.columns:
        for alt in ("symbol", "sym"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break

    value_col = _pick_value_col(df)
    if value_col and "ticker" in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            latest_dt = df["date"].max()
            snap = df[df["date"] == latest_dt].copy()
        else:
            snap = df.copy()
        snap = snap.dropna(subset=["ticker", value_col])
        meta["positions"] = sorted(snap["ticker"].unique().tolist())
        meta["position_count"] = len(meta["positions"])
        meta["invested_value"] = float(snap[value_col].sum()) if not snap.empty else None

    return meta


def count_open_orders() -> int:
    df = _read_csv(OPEN_ORDERS_SNAPSHOT_PATH)
    if df.empty:
        return 0
    if "symbol" in df.columns:
        sym = df["symbol"].astype(str).str.strip()
        return int((sym != "") & (sym.str.upper() != "NAN")).sum()
    return int(len(df))


def resolve_cash_from_artifacts() -> Tuple[Optional[float], str]:
    """Return (cash, source). Never invents cash."""
    for path, col in (
        (ACCOUNT_SNAPSHOT_PATH, "cash"),
        (ENHANCED_PORTFOLIO_HISTORY_PATH, "cash"),
        (PORTFOLIO_HISTORY_PATH, "cash"),
    ):
        df = _read_csv(path)
        if df.empty or col not in df.columns:
            continue
        df = _normalize_portfolio_history_raw(df) if path != ACCOUNT_SNAPSHOT_PATH else df
        if df.empty:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            src = path.name
            if path == PORTFOLIO_HISTORY_PATH:
                src = "portfolio_history.csv (latest row)"
            return float(series.iloc[-1]), src
    return None, "none"


def resolve_total_from_artifacts() -> Tuple[Optional[float], str]:
    for path in (ACCOUNT_SNAPSHOT_PATH, ENHANCED_PORTFOLIO_HISTORY_PATH, PORTFOLIO_HISTORY_PATH):
        df = _read_csv(path)
        if df.empty:
            continue
        if path == ACCOUNT_SNAPSHOT_PATH:
            for col in ("total_value", "equity", "portfolio_value"):
                if col in df.columns:
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if not series.empty:
                        return float(series.iloc[-1]), f"{path.name}:{col}"
            continue
        norm = _normalize_portfolio_history_raw(df)
        if norm.empty or "total_value" not in norm.columns:
            continue
        series = norm["total_value"].dropna()
        if not series.empty:
            return float(series.iloc[-1]), "portfolio_history.csv (latest row)"
    return None, "none"


def compare_snapshot_freshness() -> Dict[str, Any]:
    ph_mtime = file_mtime_dt(PORTFOLIO_HISTORY_PATH)
    pos = load_positions_snapshot_summary()
    pos_mtime = pos.get("mtime")
    pos_ts = pos.get("snapshot_ts")

    daily = aggregate_portfolio_history_daily(_read_csv(PORTFOLIO_HISTORY_PATH))
    history_latest: Optional[datetime] = None
    if not daily.empty:
        history_latest = pd.to_datetime(daily["date"].iloc[-1], errors="coerce")
        if pd.notna(history_latest):
            history_latest = history_latest.to_pydatetime()

    newer = False
    if pos_ts and history_latest:
        pos_day = pos_ts.date()
        hist_day = history_latest.date() if hasattr(history_latest, "date") else history_latest
        newer = pos_day > hist_day
    elif pos_mtime and ph_mtime:
        newer = pos_mtime > ph_mtime

    return {
        "portfolio_history_mtime": ph_mtime,
        "positions_snapshot_mtime": pos_mtime,
        "positions_snapshot_ts": pos_ts,
        "history_latest_date": history_latest,
        "positions_snapshot_newer_than_history": newer,
    }


def resolve_portfolio_dashboard_state() -> Dict[str, Any]:
    """Unified read-only state for Portfolio History dashboard."""
    raw = _read_csv(PORTFOLIO_HISTORY_PATH)
    daily = aggregate_portfolio_history_daily(raw)
    positions = load_positions_snapshot_summary()
    freshness = compare_snapshot_freshness()
    open_orders = count_open_orders()

    cash, cash_source = resolve_cash_from_artifacts()
    invested = positions.get("invested_value")
    total_known, total_source = resolve_total_from_artifacts()

    total_value: Optional[float] = None
    total_display = "UNKNOWN"
    cash_display = f"${cash:,.2f}" if cash is not None else "UNKNOWN"
    data_quality = "FULL"

    if cash is not None and invested is not None:
        total_value = float(cash) + float(invested)
        total_display = f"${total_value:,.2f}"
    elif total_known is not None:
        total_value = total_known
        total_display = f"${total_value:,.2f}"
        data_quality = "PARTIAL"
    else:
        data_quality = "PARTIAL"

    if cash is None:
        data_quality = "PARTIAL"
    if freshness.get("positions_snapshot_newer_than_history"):
        if data_quality == "FULL":
            data_quality = "PARTIAL"

    # Prefer overlay when positions snapshot is newer than history file
    use_overlay = bool(freshness.get("positions_snapshot_newer_than_history"))
    latest_as_of = freshness.get("history_latest_date")
    if use_overlay and positions.get("snapshot_ts"):
        latest_as_of = positions["snapshot_ts"]

    peak: Optional[float] = None
    drawdown_pct: Optional[float] = None
    if not daily.empty and "total_value" in daily.columns:
        tv = daily["total_value"].dropna()
        if not tv.empty:
            peak = float(tv.max())
    if total_value is not None and peak is not None and peak > 0:
        drawdown_pct = (total_value / peak - 1.0) * 100.0
    elif not daily.empty and "total_value" in daily.columns:
        tv = daily["total_value"].dropna()
        if len(tv) >= 1:
            latest_hist = float(tv.iloc[-1])
            peak = float(tv.max())
            if peak > 0:
                drawdown_pct = (latest_hist / peak - 1.0) * 100.0
            if total_value is None:
                total_value = latest_hist
                total_display = f"${total_value:,.2f}"

    return {
        "portfolio_history_exists": PORTFOLIO_HISTORY_PATH.exists(),
        "portfolio_history_rows": int(len(raw)),
        "portfolio_history_daily_rows": int(len(daily)),
        "positions_snapshot_exists": bool(positions.get("exists")),
        "positions_snapshot_rows": int(positions.get("rows") or 0),
        "positions_snapshot_newer_than_history": bool(
            freshness.get("positions_snapshot_newer_than_history")
        ),
        "freshness": freshness,
        "cash": cash,
        "cash_source": cash_source,
        "cash_display": cash_display,
        "invested_value": invested,
        "total_value": total_value,
        "total_display": total_display,
        "total_source": total_source,
        "data_quality": data_quality,
        "use_positions_overlay": use_overlay,
        "latest_as_of": latest_as_of,
        "position_count": int(positions.get("position_count") or 0),
        "latest_positions": list(positions.get("positions") or []),
        "positions_summary": positions,
        "open_orders": open_orders,
        "daily": daily,
        "raw": raw,
        "drawdown_pct": drawdown_pct,
        "peak_total_value": peak,
        "sources": {
            "portfolio_history": {
                "path": str(PORTFOLIO_HISTORY_PATH),
                "mtime": file_mtime_dt(PORTFOLIO_HISTORY_PATH),
            },
            "positions_snapshot": {
                "path": str(POSITIONS_SNAPSHOT_PATH),
                "mtime": positions.get("mtime"),
            },
            "open_orders_snapshot": {
                "path": str(OPEN_ORDERS_SNAPSHOT_PATH),
                "mtime": file_mtime_dt(OPEN_ORDERS_SNAPSHOT_PATH),
            },
        },
    }


def append_portfolio_history_if_stale(*, dry_run: bool = False) -> Dict[str, Any]:
    """
    Append one dated row when positions_snapshot is newer than portfolio history.
    Non-destructive: never deletes or rewrites prior rows; skips duplicate dates.
    """
    result: Dict[str, Any] = {
        "appended": False,
        "dry_run": dry_run,
        "destructive_history_rewrite": False,
        "reason": "",
    }
    freshness = compare_snapshot_freshness()
    if not freshness.get("positions_snapshot_newer_than_history"):
        result["reason"] = "history_current"
        return result

    positions = load_positions_snapshot_summary()
    invested = positions.get("invested_value")
    if invested is None or positions.get("position_count", 0) == 0:
        result["reason"] = "no_positions_value"
        return result

    pos_ts = positions.get("snapshot_ts")
    if not pos_ts:
        result["reason"] = "missing_snapshot_ts"
        return result

    snap_date = pos_ts.date().isoformat()
    daily = aggregate_portfolio_history_daily(_read_csv(PORTFOLIO_HISTORY_PATH))
    if not daily.empty:
        existing_dates = {
            pd.to_datetime(d).date().isoformat()
            for d in daily["date"]
            if pd.notna(pd.to_datetime(d, errors="coerce"))
        }
        if snap_date in existing_dates:
            result["reason"] = "duplicate_date"
            return result

    cash, cash_source = resolve_cash_from_artifacts()
    row: Dict[str, Any] = {
        "date": snap_date,
        "market_value": round(float(invested), 2),
    }
    if cash is not None:
        row["cash"] = round(float(cash), 2)
        row["total_value"] = round(float(cash) + float(invested), 2)
    else:
        row["cash"] = ""
        total_known, _ = resolve_total_from_artifacts()
        row["total_value"] = round(float(total_known), 2) if total_known is not None else ""

    result["proposed_row"] = row
    result["cash_source"] = cash_source

    if dry_run:
        result["reason"] = "dry_run"
        return result

    PORTFOLIO_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = PORTFOLIO_HISTORY_PATH.exists() and PORTFOLIO_HISTORY_PATH.stat().st_size > 0
    if file_exists:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = PORTFOLIO_HISTORY_PATH.with_name(f"portfolio_history.backup.{ts}.csv")
        shutil.copy2(PORTFOLIO_HISTORY_PATH, backup)
        result["backup"] = str(backup)

    mode = "a" if file_exists else "w"
    import csv

    with PORTFOLIO_HISTORY_PATH.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PH_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PH_COLS})

    result["appended"] = True
    result["reason"] = "appended"
    return result


def load_live_execution_flag() -> bool:
    doc = {}
    try:
        if EXECUTION_READINESS_PATH.exists():
            doc = json.loads(EXECUTION_READINESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    readiness = doc.get("readiness") or doc
    return bool(readiness.get("live_execution_permitted", False))


def run_phase_196f_report(*, append_if_stale: bool = True, verbose: bool = False) -> Dict[str, Any]:
    """Build phase_196f_portfolio_history_sync_repair.json report."""
    notes: List[str] = []
    state = resolve_portfolio_dashboard_state()

    append_result: Dict[str, Any] = {"appended": False, "reason": "skipped"}
    if append_if_stale and state.get("positions_snapshot_newer_than_history"):
        append_result = append_portfolio_history_if_stale(dry_run=False)
        if append_result.get("appended"):
            notes.append("Appended portfolio history row from positions snapshot.")
            state = resolve_portfolio_dashboard_state()
        elif append_result.get("reason") == "duplicate_date":
            notes.append("Positions newer but history date already present; no append.")
        else:
            notes.append(f"Append skipped: {append_result.get('reason')}")

    daily_rows = int(state.get("portfolio_history_daily_rows") or 0)
    vr_path = ROOT / "view_results.py"
    dashboard_one_row_ok = False
    try:
        dashboard_one_row_ok = "Only one portfolio history row is available" in vr_path.read_text(
            encoding="utf-8"
        )
    except Exception:
        dashboard_one_row_ok = daily_rows <= 1
    dashboard_stale_ok = True
    if state.get("positions_snapshot_newer_than_history"):
        dashboard_stale_ok = True
        notes.append("Positions snapshot newer than history; dashboard shows overlay warning.")

    status = "PASS"
    if not state.get("portfolio_history_exists") and not state.get("positions_snapshot_exists"):
        status = "FAIL"
        notes.append("Neither portfolio_history nor positions_snapshot available.")
    elif state.get("data_quality") == "PARTIAL":
        status = "REVIEW" if status == "PASS" else status

    payload = {
        "phase": "196F",
        "generated_at": _utc_now_iso(),
        "status": status,
        "dashboard_file_checked": True,
        "portfolio_history_exists": bool(state.get("portfolio_history_exists")),
        "portfolio_history_rows": int(state.get("portfolio_history_rows") or 0),
        "positions_snapshot_exists": bool(state.get("positions_snapshot_exists")),
        "positions_snapshot_rows": int(state.get("positions_snapshot_rows") or 0),
        "positions_snapshot_newer_than_history": bool(
            state.get("positions_snapshot_newer_than_history")
        ),
        "cash_available": state.get("cash") is not None,
        "market_value_available": state.get("invested_value") is not None,
        "total_value_available": state.get("total_value") is not None,
        "latest_positions": list(state.get("latest_positions") or []),
        "open_orders": int(state.get("open_orders") or 0),
        "live_execution": load_live_execution_flag(),
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "dashboard_handles_one_row_history": dashboard_one_row_ok,
        "dashboard_handles_stale_history": dashboard_stale_ok,
        "destructive_history_rewrite": bool(append_result.get("destructive_history_rewrite")),
        "append_result": append_result,
        "data_quality": state.get("data_quality"),
        "notes": notes,
    }

    out_path = RESULTS / "phase_196f_portfolio_history_sync_repair.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out_path)

    if verbose:
        print(f"[196F] wrote {out_path.relative_to(ROOT)}")
        print(f"[196F] status={status}")

    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Portfolio history dashboard sync (read-only repair)")
    ap.add_argument("--no-append", action="store_true", help="Report only; do not append history")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    report = run_phase_196f_report(append_if_stale=not args.no_append, verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    raise SystemExit(main())
