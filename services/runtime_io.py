# services/runtime_io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def atomic_write_json(path: Path, obj: Any) -> None:
    """
    Atomic JSON write to avoid partial reads by Streamlit/dashboard.
    Writes to *.tmp then replaces target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def safe_read_json(path: Path) -> Dict[str, Any]:
    """
    Safe JSON read. Returns {} on missing or error.
    """
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
