"""
Lightweight schema helpers: dedupe columns, validate required columns, safe merges.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pandas as pd


def dedupe_columns(df: pd.DataFrame, *, keep: str = "first", warn_label: str = "") -> pd.DataFrame:
    cols = df.columns
    dupes = cols[cols.duplicated()].tolist()
    if dupes:
        label = warn_label or "frame"
        print(f"[schema_guard] duplicate columns removed in {label}: {dupes}")
        df = df.loc[:, ~cols.duplicated(keep=keep)].copy()
    return df


def find_duplicate_columns(df: pd.DataFrame) -> List[Any]:
    return df.columns[df.columns.duplicated()].tolist()


def require_columns(
    df: pd.DataFrame,
    required: List[str],
    *,
    label: str = "df",
    hard_fail: bool = False,
) -> List[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"[schema_guard] missing required columns in {label}: {missing}"
        if hard_fail:
            raise ValueError(msg)
        print(msg)
    return missing


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: Optional[Any] = None,
    left_on: Optional[Any] = None,
    right_on: Optional[Any] = None,
    how: str = "left",
    suffixes: Tuple[str, str] = ("_x", "_y"),
    label: str = "merge",
) -> pd.DataFrame:
    merge_kw: dict = {"how": how, "suffixes": suffixes}
    if on is not None:
        merge_kw["on"] = on
    if left_on is not None:
        merge_kw["left_on"] = left_on
    if right_on is not None:
        merge_kw["right_on"] = right_on

    df = pd.merge(left, right, **merge_kw)

    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        print(f"[schema_guard] duplicate columns after {label}: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()].copy()

    return df


def schema_snapshot(df: pd.DataFrame, *, label: str = "df") -> None:
    dupes = find_duplicate_columns(df)
    print(f"[schema_guard] {label} rows={len(df)} cols={len(df.columns)} dupes={dupes}")
    print(f"[schema_guard] columns preview: {list(df.columns[:20])}")
