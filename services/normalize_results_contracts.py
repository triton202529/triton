# services/normalize_results_contracts.py
"""
Normalize dashboard-facing result artifacts to deterministic schemas.

Goal:
- Keep data/results/portfolio_history.csv STRICT and dashboard-safe:
    date,cash,market_value,total_value
- Keep data/results/trade_log.csv stable (rename qty->quantity if needed).
- Preserve any extra columns by writing enhanced artifacts instead of polluting
  dashboard contracts:
    data/results/enhanced_portfolio_history.csv

This prevents the "pipeline flapping" where one stage writes a 4-col portfolio_history
and a later stage reintroduces extra columns (e.g., 'regime') causing health checks to go red.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PORTFOLIO = RESULTS_DIR / "portfolio_history.csv"
ENHANCED_PORTFOLIO = RESULTS_DIR / "enhanced_portfolio_history.csv"
TRADE_LOG = RESULTS_DIR / "trade_log.csv"

PH_COLS: List[str] = ["date", "cash", "market_value", "total_value"]


def _normalize_column_names(cols: List[object]) -> List[str]:
    out: List[str] = []
    for c in cols:
        s = str(c).replace("\ufeff", "").strip()
        out.append(s)
    return out


def _log_contract_check(
    path: Path,
    df: pd.DataFrame,
    required: List[str],
    missing: List[str],
) -> None:
    cols = list(df.columns) if not df.empty else []
    exists = path.is_file()
    print(
        f"[CONTRACT_CHECK] path={path.resolve()} exists={exists} "
        f"columns={cols} required={required} missing={missing}",
        flush=True,
    )


def _backup(path: Path) -> None:
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.backup.{ts}{path.suffix}")
    shutil.copy2(path, backup)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = _normalize_column_names(list(df.columns))
    return df


def _apply_portfolio_aliases(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PH_COLS if c not in df.columns]
    if "equity" in df.columns and "total_value" in missing:
        df["total_value"] = df["equity"]
        missing = [c for c in PH_COLS if c not in df.columns]
    if "portfolio_value" in df.columns and "total_value" in missing:
        df["total_value"] = df["portfolio_value"]
        missing = [c for c in PH_COLS if c not in df.columns]
    return df


def _repair_portfolio_market_value(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PH_COLS if c not in df.columns]
    if "market_value" in missing and "total_value" in df.columns and "cash" in df.columns:
        total = pd.to_numeric(df["total_value"], errors="coerce")
        cash = pd.to_numeric(df["cash"], errors="coerce")
        df["market_value"] = total - cash
        print(
            "[CONTRACT_REPAIR] portfolio_history.csv derived market_value = total_value - cash",
            flush=True,
        )
        _backup(PORTFOLIO)
        _atomic_write_csv(df, PORTFOLIO)
    return df


def _normalize_portfolio_history() -> None:
    df = _read_csv(PORTFOLIO)
    if df.empty:
        _log_contract_check(PORTFOLIO, df, PH_COLS, [])
        return

    df = _apply_portfolio_aliases(df)
    missing = [c for c in PH_COLS if c not in df.columns]
    _log_contract_check(PORTFOLIO, df, PH_COLS, missing)

    if "market_value" in missing:
        df = _repair_portfolio_market_value(df)
        missing = [c for c in PH_COLS if c not in df.columns]
        _log_contract_check(PORTFOLIO, df, PH_COLS, missing)

    if missing:
        raise SystemExit(
            f"[normalize_results_contracts] portfolio_history.csv missing required columns: {missing}"
        )

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Coerce numeric
    for c in ["cash", "market_value", "total_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Preserve enhanced copy if extra columns exist
    extra_cols = [c for c in df.columns if c not in PH_COLS]
    if extra_cols:
        _backup(ENHANCED_PORTFOLIO)
        df.to_csv(ENHANCED_PORTFOLIO, index=False)

    # Write STRICT dashboard file
    strict = df[PH_COLS].copy()
    strict["cash"] = strict["cash"].round(2)
    strict["market_value"] = strict["market_value"].round(2)
    strict["total_value"] = strict["total_value"].round(2)

    _backup(PORTFOLIO)
    _atomic_write_csv(strict, PORTFOLIO)


def _normalize_trade_log() -> None:
    df = _read_csv(TRADE_LOG)
    if df.empty:
        return

    # Normalize qty -> quantity
    if "qty" in df.columns and "quantity" not in df.columns:
        df = df.rename(columns={"qty": "quantity"})

    # Normalize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()

    _backup(TRADE_LOG)
    _atomic_write_csv(df, TRADE_LOG)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _normalize_trade_log()
    _normalize_portfolio_history()

    print("[normalize_results_contracts] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
