"""Fail closed if a test reads or writes production operational artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

PROD = Path(r"C:\Users\akimw\triton").resolve()
ISO = Path(r"C:\Users\akimw\triton-bn1-performance-isolated").resolve()
BM6A = Path(r"C:\Users\akimw\triton-bm6a-ui-isolated").resolve()

MUTABLE_NAMES = frozenset(
    {
        "paper_order_events.jsonl",
        "paper_order_events.lock",
        "performance_accounting_events.jsonl",
        "performance_accounting_events.lock",
        "paper_sell_arm.sqlite3",
        "paper_sell_arm.sqlite3-wal",
        "paper_sell_arm.sqlite3-shm",
        "paper_sell_arm_authority.json",
        "portfolio_history.csv",
        "recent_orders.csv",
        "positions_snapshot.csv",
    }
)

WRITE_MODES = frozenset({"w", "a", "x", "r+", "w+", "a+", "wb", "ab", "xb", "r+b", "w+b", "a+b"})


class ProductionIsolationError(RuntimeError):
    pass


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _under(path: Path, root: Path) -> bool:
    text = str(_resolve(path)).lower()
    prefix = str(root).lower().rstrip("\\/") + os.sep
    return text.startswith(prefix)


def is_temp_path(path: Path) -> bool:
    resolved = str(_resolve(path)).lower()
    temp_roots = [
        os.environ.get("TEMP", ""),
        os.environ.get("TMP", ""),
        r"c:\users\akimw\appdata\local\temp",
        r"c:\users\akimw\appdata\local\pytest-of-akimw",
    ]
    return any(
        resolved.startswith(str(root).lower().rstrip("\\/") + os.sep) for root in temp_roots if root
    )


def is_mutable_repo_data(path: Path) -> bool:
    resolved = _resolve(path)
    if resolved.name.lower() in {n.lower() for n in MUTABLE_NAMES}:
        return (
            _under(resolved, PROD / "data")
            or _under(resolved, ISO / "data")
            or _under(resolved, BM6A / "data")
        )
    return (
        _under(resolved, PROD / "data")
        or _under(resolved, ISO / "data")
        or _under(resolved, BM6A / "data")
    )


def check_open(path: Path, mode: str) -> None:
    resolved = _resolve(Path(path))
    normalized = str(mode or "r").replace("+", "+")
    writing = any(flag in str(mode) for flag in ("w", "a", "x", "+"))
    if writing and is_mutable_repo_data(resolved) and not is_temp_path(resolved):
        raise ProductionIsolationError(f"forbidden_production_or_repo_data_write:{resolved}")
    if (not writing) and resolved.name.lower() in {n.lower() for n in MUTABLE_NAMES}:
        if (
            _under(resolved, PROD / "data")
            or _under(resolved, ISO / "data")
            or _under(resolved, BM6A / "data")
        ):
            raise ProductionIsolationError(f"forbidden_production_artifact_read:{resolved}")


def assert_allowed_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if is_mutable_repo_data(Path(path)) and not is_temp_path(Path(path)):
            raise ProductionIsolationError(f"forbidden_repo_data_path:{path}")
