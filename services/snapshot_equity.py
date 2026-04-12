# services/snapshot_equity.py
from __future__ import annotations

import csv
import os
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.broker_alpaca import AlpacaBroker  # noqa: E402

OUT_PATH = Path("data/results/portfolio_history.csv")

COLS: List[str] = [
    "date",
    "timestamp",
    "portfolio_value",
    "equity",
    "buying_power",
    "cash",
    "long_mv",
    "short_mv",
    "net_mv",
    "date_utc",
    "timestamp_utc",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _f(x: Any) -> float:
    try:
        return float(x or 0)
    except Exception:
        return 0.0


def _safe_get(acct: Dict[str, Any], *keys: str) -> float:
    for k in keys:
        if k in acct and acct[k] is not None:
            return _f(acct[k])
    return 0.0


def _read_header(path: Path) -> List[str]:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, [])
        return [h.strip() for h in header if h is not None]
    except Exception:
        return []


def _upgrade_schema_if_needed() -> None:
    if not OUT_PATH.exists():
        return

    header = _read_header(OUT_PATH)
    if not header or header == COLS:
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = OUT_PATH.parent / f"portfolio_history.backup.{ts}.csv"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_PATH, backup)

    rows: List[Dict[str, Any]] = []
    try:
        with OUT_PATH.open("r", newline="", encoding="utf-8") as f:
            dr = csv.DictReader(f)
            for r in dr:
                if r:
                    rows.append(dict(r))
    except Exception:
        rows = []

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()

        for r in rows:
            date = r.get("date") or r.get("date_utc") or ""
            timestamp = r.get("timestamp") or r.get("timestamp_utc") or ""
            equity = r.get("equity") or r.get("portfolio_value") or r.get("account_equity") or ""
            pv = r.get("portfolio_value") or equity

            out = {
                "date": date,
                "timestamp": timestamp,
                "portfolio_value": pv,
                "equity": equity or pv,
                "buying_power": r.get("buying_power", ""),
                "cash": r.get("cash", ""),
                "long_mv": r.get("long_mv", "") or r.get("long_market_value", ""),
                "short_mv": r.get("short_mv", "") or r.get("short_market_value", ""),
                "net_mv": r.get("net_mv", "") or r.get("position_market_value", ""),
                "date_utc": r.get("date_utc", "") or date,
                "timestamp_utc": r.get("timestamp_utc", "") or timestamp,
            }
            w.writerow(out)

    print(f"[snapshot_equity] Upgraded portfolio_history.csv schema. Backup: {backup}")


def snapshot_equity(broker: AlpacaBroker) -> Dict[str, Any]:
    acct = broker.get_account()

    now = _utc_now()
    date_utc = now.strftime("%Y-%m-%d")
    ts_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    equity = _safe_get(acct, "equity", "portfolio_value", "last_equity")
    buying_power = _safe_get(acct, "buying_power")
    cash = _safe_get(acct, "cash", "cash_withdrawable")
    long_mv = _safe_get(acct, "long_market_value")
    short_mv = _safe_get(acct, "short_market_value")
    net_mv = _safe_get(acct, "position_market_value")

    return {
        "date": date_utc,
        "timestamp": ts_utc,
        "portfolio_value": equity,
        "equity": equity,
        "buying_power": buying_power,
        "cash": cash,
        "long_mv": long_mv,
        "short_mv": short_mv,
        "net_mv": net_mv,
        "date_utc": date_utc,
        "timestamp_utc": ts_utc,
    }


def _last_timestamp_utc() -> str:
    """Read the last non-empty line and return its timestamp_utc (or timestamp)."""
    try:
        if not OUT_PATH.exists() or OUT_PATH.stat().st_size == 0:
            return ""
        with OUT_PATH.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            if end < 2:
                return ""
            # read last ~2KB
            size = min(2048, end)
            f.seek(end - size)
            chunk = f.read(size).decode("utf-8", errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if len(lines) < 2:
            return ""
        last = lines[-1]
        # Our CSV layout: date,timestamp,portfolio_value,...,date_utc,timestamp_utc
        parts = [p.strip() for p in last.split(",")]
        if len(parts) >= 11:
            return parts[-1] or parts[1]  # timestamp_utc else timestamp
        if len(parts) >= 2:
            return parts[1]
        return ""
    except Exception:
        return ""


def append_snapshot(row: Dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _upgrade_schema_if_needed()

    # ✅ De-dupe: if same-second timestamp already exists as last row, skip.
    ts = str(row.get("timestamp_utc") or row.get("timestamp") or "").strip()
    if ts:
        last_ts = _last_timestamp_utc()
        if last_ts == ts:
            return

    file_exists = OUT_PATH.exists()
    with OUT_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})


def main() -> None:
    import argparse

    default_mode = os.getenv("TRITON_BROKER_MODE", "paper").strip().lower()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=default_mode, choices=["paper", "live"])
    args = ap.parse_args()

    broker = AlpacaBroker(mode=args.mode)
    row = snapshot_equity(broker)
    append_snapshot(row)
    print("Wrote snapshot:", row)


if __name__ == "__main__":
    main()
