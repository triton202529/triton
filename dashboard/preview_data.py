"""Isolated preview capture banner and path guard. Presentation only."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

PROD_ROOT = Path(r"C:\Users\akimw\triton")
EXPECTED_LEDGER_SHA256 = "6ce03a05bf553eb8a9dce5938f964f2a24dd1f1e4ee1437e5f9f87d4cc6e9aee"
EXPECTED_ACCOUNTING_LEDGER_SHA256 = (
    "9227fb2c802261434877860d250df4a45575c141a6ec6605a955afd5b11a73d6"
)
CAPTURE_RELATIVE = "tmp/preview_capture.json"
MARKER_RELATIVE = "tmp/ui_a_isolation_marker.txt"
FIXTURE_RELATIVE = "tmp/phase_198T_BP1_fixtures/frozen"


def isolation_root() -> Path:
    return Path(__file__).resolve().parents[1]


def isolation_active(root: Optional[Path] = None) -> bool:
    """True only for the inspected root (or this worktree when omitted)."""
    target = Path(root) if root is not None else isolation_root()
    return (target / MARKER_RELATIVE).is_file()


def preview_data_root(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else isolation_root()
    fixture = (isolation_root() if isolation_active(base) else base) / FIXTURE_RELATIVE
    if isolation_active(base) and fixture.is_dir():
        return fixture
    return base


def load_preview_capture(root: Optional[Path] = None) -> Dict[str, Any]:
    path = (root or isolation_root()) / CAPTURE_RELATIVE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def expected_ledger_sha256(root: Optional[Path] = None) -> Optional[str]:
    """Pinned hash for isolated preview/tests only.

    Production must validate the current canonical ledger and must not inherit
    the BM5 fixture hash. Isolated preview keeps its configured fixture pin.
    """
    if not isolation_active(root):
        return None
    captured = str(load_preview_capture(root).get("ledger_sha256") or "").strip()
    return captured or EXPECTED_LEDGER_SHA256


def expected_accounting_ledger_sha256(root: Optional[Path] = None) -> str:
    captured = str(load_preview_capture(root).get("accounting_ledger_sha256") or "").strip()
    return captured or EXPECTED_ACCOUNTING_LEDGER_SHA256


def preview_capture_label(captured_at: str) -> str:
    text = str(captured_at or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(ZoneInfo("America/New_York"))
        stamp = local.strftime("%Y-%m-%d %H:%M:%S ET")
    except ValueError:
        stamp = captured_at
    return f"Preview data captured at {stamp}"


def preview_as_of_datetime(root: Optional[Path] = None) -> Optional[datetime]:
    captured = str(load_preview_capture(root).get("preview_data_captured_at") or "").strip()
    if not captured:
        return None
    if captured.endswith("Z"):
        captured = captured[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(captured)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def preview_banner_text(root: Optional[Path] = None) -> str:
    captured = str(load_preview_capture(root).get("preview_data_captured_at") or "").strip()
    if captured:
        return preview_capture_label(captured)
    return "Preview data captured at unavailable"


def assert_not_production_path(path: Path, *, root: Optional[Path] = None) -> None:
    if not isolation_active(root):
        return
    resolved = Path(path).resolve()
    prod = PROD_ROOT.resolve()
    if resolved == prod or str(resolved).startswith(str(prod) + os.sep):
        raise RuntimeError("isolated preview refuses production path")
