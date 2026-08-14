"""
Shared Windows directory-durability test stubs for paper-sell suites.

Production on Windows fail-closes broker-usable permits: FlushFileBuffers
success is not accepted as proven OS-crash/power-loss directory-entry
durability. Happy-path unit tests may stub both `_fsync_dir` and
`durability_meets_dispatch_standard` so non-durability logic can be exercised.
Tests that intentionally inject directory-durability failure must re-monkeypatch
after this stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import services.paper_sell_arm_authority as auth_mod


def _test_durability_meets(meta: Dict[str, Any]) -> bool:
    if meta.get("flush_ok") is False:
        return False
    if not meta.get("file_fsync_ok"):
        return False
    return bool(meta.get("dir_fsync_ok"))


def stub_dir_fsync_ok(monkeypatch: Any) -> None:
    monkeypatch.setattr(auth_mod, "_fsync_dir", lambda _p: True)
    monkeypatch.setattr(auth_mod, "durability_meets_dispatch_standard", _test_durability_meets)


def stub_dir_fsync_ok_inprocess() -> None:
    """For spawn workers / threads that cannot use pytest monkeypatch."""
    auth_mod._fsync_dir = lambda _p: True  # type: ignore[assignment]
    auth_mod.durability_meets_dispatch_standard = _test_durability_meets  # type: ignore[assignment]


def assert_not_real_governance(path: Path) -> None:
    text = str(path.resolve()).lower().replace("\\", "/")
    assert "/data/governance/" not in text or "/tmp" in text or "pytest" in text
