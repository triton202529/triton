"""Materialize frozen BM5 evidence. Isolated writes only. No synthetic orders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tests.fixtures.bm5.authoritative import (
    GE_AUTHORITY_ID,
    SAME_CYCLE,
    AuthoritativeLedgerError,
    materialize_authoritative_fixture,
    write_preview_capture,
)

__all__ = [
    "GE_AUTHORITY_ID",
    "SAME_CYCLE",
    "AuthoritativeLedgerError",
    "materialize_bm5_fixture",
    "write_malformed_packet",
]


def write_malformed_packet(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")


def materialize_bm5_fixture(root: Path, *, include_sqlite: bool = False) -> Dict[str, Any]:
    """Copy frozen authoritative evidence. include_sqlite is accepted for compatibility."""
    del include_sqlite
    result = materialize_authoritative_fixture(root)
    write_preview_capture(Path(root))
    return result
