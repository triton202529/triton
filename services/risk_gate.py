# services/risk_gate.py
"""
Risk Gate (Phase 1.5)

Single source of truth:
- Reads data/results/adaptive_risk_state.json
- Decides whether Triton may place new orders
- Provides caps (max_gross_exposure, max_position_weight)
- Designed to be used by ALL execution entrypoints

No broker calls. No network calls. Artifact-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

STATE_PATH = Path("data/results/adaptive_risk_state.json")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _to_float(x: Any) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _read_json_bom_safe(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (obj, err). Uses utf-8-sig to tolerate BOM from PowerShell Set-Content.
    """
    if not path.exists():
        return None, f"Missing: {path.as_posix()}"
    try:
        if path.stat().st_size == 0:
            return None, f"Empty file: {path.as_posix()}"
    except Exception as e:
        return None, f"Stat failed: {e}"

    # Try utf-8-sig first (handles BOM), then fallback utf-8
    for enc in ("utf-8-sig", "utf-8"):
        try:
            raw = path.read_text(encoding=enc)
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj, None
            return None, f"JSON root not dict: {path.name}"
        except Exception as e:
            last_err = str(e)

    return None, f"JSON read/parse failed: {last_err}"


# ─────────────────────────────────────────────────────────────
# Decision object
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RiskGateDecision:
    ok: bool
    reason: str
    block_reason: str
    risk_on: bool
    allow_new_orders: bool
    global_kill_switch: bool

    max_gross_exposure: float
    max_position_weight: float

    regime: str
    mode: str
    timestamp: str
    raw: Dict[str, Any]


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def read_risk_state(state_path: Path = STATE_PATH) -> RiskGateDecision:
    """
    Read state and return a normalized decision object.

    Conservative behavior:
    - If state file missing/unreadable/invalid -> BLOCK (global_kill_switch=True)
    """
    state, err = _read_json_bom_safe(state_path)
    state = state or {}

    ts = str(state.get("timestamp", "")) if isinstance(state, dict) else ""
    regime = str(state.get("regime", "UNKNOWN")) if isinstance(state, dict) else "UNKNOWN"
    mode = str(state.get("mode", "UNKNOWN")) if isinstance(state, dict) else "UNKNOWN"
    reason = str(state.get("reason", "") or "")

    # If file unreadable/missing, hard-block with an actionable reason
    if err is not None:
        return RiskGateDecision(
            ok=False,
            reason=f"Risk state unreadable: {err}",
            block_reason="STATE_UNREADABLE",
            risk_on=False,
            allow_new_orders=False,
            global_kill_switch=True,  # conservative
            max_gross_exposure=0.0,
            max_position_weight=0.0,
            regime=regime,
            mode=mode,
            timestamp=ts,
            raw=state,
        )

    controls = state.get("controls", {}) if isinstance(state.get("controls"), dict) else {}

    # Conservative defaults if keys missing
    gks = bool(controls.get("global_kill_switch", True))
    risk_on = bool(controls.get("risk_on", False))
    allow_new = bool(controls.get("allow_new_orders", False))
    block_reason = str(controls.get("block_reason", "") or "")

    max_gross = _to_float(controls.get("max_gross_exposure", 0.0))
    max_pos = _to_float(controls.get("max_position_weight", 0.0))

    # Clamp NaNs to zero (conservative)
    if not np.isfinite(max_gross):
        max_gross = 0.0
    if not np.isfinite(max_pos):
        max_pos = 0.0

    # Hard kill switch override
    if gks:
        return RiskGateDecision(
            ok=False,
            reason=reason or "Global kill switch enabled",
            block_reason=block_reason or "GLOBAL_KILL_SWITCH",
            risk_on=False,
            allow_new_orders=False,
            global_kill_switch=True,
            max_gross_exposure=0.0,
            max_position_weight=0.0,
            regime=regime,
            mode=mode,
            timestamp=ts,
            raw=state,
        )

    ok = bool(risk_on and allow_new)

    # Fill a default block reason if blocked
    if not ok and not block_reason:
        if not risk_on:
            block_reason = "RISK_OFF"
        elif not allow_new:
            block_reason = "NEW_ORDERS_BLOCKED"
        else:
            block_reason = "BLOCKED"

    return RiskGateDecision(
        ok=ok,
        reason=reason or ("All systems nominal" if ok else "Blocked by controls"),
        block_reason=block_reason,
        risk_on=risk_on,
        allow_new_orders=allow_new,
        global_kill_switch=False,
        max_gross_exposure=float(max_gross),
        max_position_weight=float(max_pos),
        regime=regime,
        mode=mode,
        timestamp=ts,
        raw=state,
    )


def assert_can_place_orders(state_path: Path = STATE_PATH) -> RiskGateDecision:
    """
    Use this at the top of any order-placement path.
    Raises RuntimeError if blocked.
    """
    d = read_risk_state(state_path)
    if not d.ok:
        raise RuntimeError(
            f"RISK GATE BLOCKED: {d.block_reason} | regime={d.regime} mode={d.mode} | {d.reason}"
        )
    return d
