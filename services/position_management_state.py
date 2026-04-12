# services/position_management_state.py
"""Lightweight per-symbol state for manage_positions (weak-cycle counting). Best-effort; never raises."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "results" / "position_management_state.json"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> Dict[str, Any]:
    try:
        if STATE_PATH.is_file() and STATE_PATH.stat().st_size > 0:
            o = json.loads(STATE_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(o, dict) and isinstance(o.get("symbols"), dict):
                return o
    except Exception:
        pass
    return {"version": 1, "updated_at": "", "symbols": {}}


def save_state(state: Dict[str, Any]) -> None:
    try:
        state["updated_at"] = _utc_iso()
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception:
        try:
            STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass


def get_symbol_state(state: Dict[str, Any], sym: str) -> Dict[str, Any]:
    syms = state.setdefault("symbols", {})
    if sym not in syms or not isinstance(syms[sym], dict):
        syms[sym] = {
            "last_seen_cycle_ts": "",
            "consecutive_weak_cycles": 0,
            "last_management_action": "NONE",
            "last_effective_stance": "",
            "last_trim_ts_utc": "",
        }
    return syms[sym]
