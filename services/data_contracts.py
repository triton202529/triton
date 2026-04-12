# services/data_contracts.py
# ------------------------------------------------------------
# TRITON — Data Contracts Validator (Phase 1.5)
#
# Goal:
#   Validate saved artifacts (CSV/JSON/Parquet) against expected schemas.
#   Produce human-readable FAIL blocks + structured results.
#
# Key hardening:
#   - CSV header normalization (strip + remove BOM)
#   - Contract-aware column aliasing (symbol <- ticker, total_value <- equity, etc.)
# ------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class DataContract:
    name: str
    path: Path
    fmt: str  # "csv" | "json" | "parquet"
    required_cols: List[str] = field(default_factory=list)
    optional_cols: List[str] = field(default_factory=list)
    min_rows: int = 0
    unique_keys: Optional[List[str]] = None
    date_col: Optional[str] = "date"  # if present, validate min/max
    allow_empty: bool = False


def _normalize_columns(cols: List[Any]) -> List[str]:
    out: List[str] = []
    for c in cols:
        s = str(c)
        s = s.strip().lstrip("\ufeff")
        out.append(s)
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    # utf-8-sig removes BOM automatically if present
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = _normalize_columns(list(df.columns))
    return df


def _read_json(path: Path) -> Any:
    txt = path.read_text(encoding="utf-8")
    return json.loads(txt)


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Parquet columns can still be odd; normalize anyway
    df.columns = _normalize_columns(list(df.columns))
    return df


def _apply_contract_aliases(contract: DataContract, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes schema drift at read-time so validators don't lie.

    We only add/alias columns. We do NOT drop columns.
    """
    p = str(contract.path).replace("\\", "/").lower()

    # Positions snapshot: validator requires `symbol` but some writers use `ticker`
    if p.endswith("/positions_snapshot.csv"):
        if "symbol" not in df.columns and "ticker" in df.columns:
            df["symbol"] = df["ticker"]

    # Portfolio history: validator requires `total_value` but some writers use alternates
    if p.endswith("/portfolio_history.csv") or p.endswith("/enhanced_portfolio_history.csv"):
        if "total_value" not in df.columns:
            for alt in (
                "equity",
                "portfolio_value",
                "value",
                "account_value",
                "account_equity",
                "net_mv",
                "net_liquidation",
            ):
                if alt in df.columns:
                    df["total_value"] = df[alt]
                    break

    return df


def _date_range(df: pd.DataFrame, date_col: str) -> Tuple[Optional[str], Optional[str]]:
    if date_col not in df.columns or df.empty:
        return None, None
    try:
        dd = pd.to_datetime(df[date_col], errors="coerce")
        dd = dd.dropna()
        if dd.empty:
            return None, None
        return str(dd.min()), str(dd.max())
    except Exception:
        return None, None


def validate_one(project_root: Path, contract: DataContract) -> Dict[str, Any]:
    """
    Returns a structured result with:
      ok: bool
      name, path, fmt
      rows
      missing_columns: list[str]
      date_range: {min_date, max_date} (if applicable)
      notes/errors
    """
    path = contract.path
    if not path.is_absolute():
        path = project_root / path

    result: Dict[str, Any] = {
        "ok": True,
        "name": contract.name,
        "path": str(path),
        "fmt": contract.fmt,
        "rows": 0,
        "missing_columns": [],
        "date_range": None,
        "errors": [],
    }

    if not path.exists() or path.stat().st_size <= 0:
        if contract.allow_empty:
            result["ok"] = True
            result["rows"] = 0
            return result
        result["ok"] = False
        result["errors"].append("Missing or empty file.")
        return result

    try:
        if contract.fmt == "csv":
            df = _read_csv(path)
            df = _apply_contract_aliases(contract, df)
            result["rows"] = int(len(df))

            # required columns
            missing = [c for c in contract.required_cols if c not in df.columns]
            result["missing_columns"] = missing
            if missing:
                result["ok"] = False

            # min rows
            if contract.min_rows and len(df) < contract.min_rows and not contract.allow_empty:
                result["ok"] = False
                result["errors"].append(f"Too few rows: {len(df)} < {contract.min_rows}")

            # date range
            if contract.date_col:
                mn, mx = _date_range(df, contract.date_col)
                if mn or mx:
                    result["date_range"] = {"min_date": mn, "max_date": mx}

            # unique keys
            if contract.unique_keys:
                keys = [k for k in contract.unique_keys if k in df.columns]
                if keys:
                    dupes = df.duplicated(subset=keys).sum()
                    if dupes > 0:
                        result["errors"].append(f"Duplicate rows by keys {keys}: {int(dupes)}")

            return result

        if contract.fmt == "json":
            _ = _read_json(path)
            result["rows"] = 1
            return result

        if contract.fmt == "parquet":
            df = _read_parquet(path)
            result["rows"] = int(len(df))
            missing = [c for c in contract.required_cols if c not in df.columns]
            result["missing_columns"] = missing
            if missing:
                result["ok"] = False
            return result

        result["ok"] = False
        result["errors"].append(f"Unknown contract format: {contract.fmt}")
        return result

    except Exception as e:
        result["ok"] = False
        result["errors"].append(str(e))
        return result


def validate_all(project_root: Path, contracts: List[DataContract]) -> List[Dict[str, Any]]:
    return [validate_one(project_root, c) for c in contracts]


def print_report(results: List[Dict[str, Any]]) -> None:
    """
    Human-readable output similar to your ❌ blocks.
    """
    for r in results:
        name = r.get("name", "Unknown")
        path = r.get("path", "")
        rows = r.get("rows", 0)
        ok = bool(r.get("ok", False))

        if ok:
            print(f"✅ {name} — {rows} rows")
            continue

        print(f"❌ {name} — {rows} rows\n")
        print(path + "\n")

        missing = r.get("missing_columns") or []
        if missing:
            print(f"[MISSING_COLUMNS] Missing required columns: {missing}\n")
            print("Hint: Fix upstream generator or update contract intentionally.\n")
            print(json.dumps({"missing": list(enumerate(missing))}, indent=2))
            print()

        dr = r.get("date_range")
        if isinstance(dr, dict) and (dr.get("min_date") or dr.get("max_date")):
            print(f"[DATE_RANGE] {dr.get('min_date')} \u2192 {dr.get('max_date')}\n")
            print(json.dumps(dr, indent=2))
            print()

        errs = r.get("errors") or []
        for e in errs:
            print(f"[ERROR] {e}")
        print("-" * 60)
