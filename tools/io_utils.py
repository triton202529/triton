#!/usr/bin/env python3
"""
I/O helpers used by the loaders.
- Prefer <file>.fixed.csv if present
- Ensure returned DataFrame has ['date','close'] cleaned + sorted
"""

from pathlib import Path
import pandas as pd


def _read_candidate(p: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        if "date" in cols and "close" in cols:
            out = df.rename(columns={cols["date"]: "date", cols["close"]: "close"})
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out["close"] = pd.to_numeric(out["close"], errors="coerce")
            out = out.dropna(subset=["date", "close"]).sort_values("date")
            return out if not out.empty else None
    except Exception:
        return None
    return None


def smart_read_price_csv(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    fixed = Path(str(path) + ".fixed.csv")
    return _read_candidate(fixed) or _read_candidate(path)
