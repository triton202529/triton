"""Read-only dashboard file cache.

Entries are keyed by path plus ``(mtime_ns, size)`` and are dropped by Refresh data.
This module does not cache an order authority, paper arm, broker permission,
hold decision, or runtime process scan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

MISSING_STAMP = (-1, -1)


def source_stamp(path: Path) -> tuple[int, int]:
    try:
        info = Path(path).stat()
    except OSError:
        return MISSING_STAMP
    return (int(info.st_mtime_ns), int(info.st_size))


def _stamp_matches(path: Path, mtime_ns: int, size: int) -> None:
    info = path.stat()
    if int(info.st_mtime_ns) != int(mtime_ns) or int(info.st_size) != int(size):
        raise OSError("source stamp changed before read")


@st.cache_data(show_spinner=False)
def read_local_bytes(path_text: str, mtime_ns: int, size: int) -> bytes:
    """Return file bytes for one path and stamp. A new stamp is a cache miss."""
    path = Path(path_text)
    _stamp_matches(path, mtime_ns, size)
    data = path.read_bytes()
    if len(data) != int(size):
        raise OSError("source size changed during read")
    return data


@st.cache_data(show_spinner=False)
def read_jsonl_tail(
    path_text: str, mtime_ns: int, size: int, max_lines: int
) -> List[Dict[str, Any]]:
    """Parse only the last ``max_lines`` complete JSON objects in a local JSONL file."""
    if int(max_lines) < 1:
        return []
    path = Path(path_text)
    _stamp_matches(path, mtime_ns, size)
    lines = path.read_bytes().splitlines()
    selected = lines[-int(max_lines) :]
    rows: List[Dict[str, Any]] = []
    for line in selected:
        text = line.decode("utf-8").strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


@st.cache_data(show_spinner=False)
def read_csv_records(path_text: str, mtime_ns: int, size: int) -> List[Dict[str, Any]]:
    path = Path(path_text)
    _stamp_matches(path, mtime_ns, size)
    import pandas as pd

    frame = pd.read_csv(path)
    return frame.to_dict("records")


@st.cache_data(show_spinner=False)
def _cached_certified_totals(
    root_text: str,
    mtime_ns: int,
    size: int,
    expected_sha256: str,
) -> Optional[Dict[str, Any]]:
    from dashboard.performance_presentation import try_certified_page_totals

    ledger = Path(root_text) / "data" / "results" / "performance_accounting_events.jsonl"
    if mtime_ns >= 0:
        _stamp_matches(ledger, mtime_ns, size)
    return try_certified_page_totals(
        root=Path(root_text),
        expected_sha256=expected_sha256 or None,
    )


def load_cached_certified_totals(
    root: Path,
    expected_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Certified display totals for the current ledger stamp. Not a hold decision."""
    ledger = Path(root) / "data" / "results" / "performance_accounting_events.jsonl"
    stamp = source_stamp(ledger)
    return _cached_certified_totals(
        str(Path(root)),
        stamp[0],
        stamp[1],
        expected_sha256 or "",
    )


def clear_dashboard_read_cache() -> None:
    """Drop stamp-keyed dashboard reads. Refresh data calls this."""
    read_local_bytes.clear()
    read_jsonl_tail.clear()
    read_csv_records.clear()
    _cached_certified_totals.clear()
    try:
        st.cache_data.clear()
    except Exception:
        pass
