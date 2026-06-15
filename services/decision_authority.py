"""
decision_authority.py
THP-H006 — Authoritative lifecycle and risk artifact readers for execution consumers.

Institutional rule:
  Lifecycle → signal_lifecycle_effective.csv
  Execution risk → master_execution_gate.json (snapshot from MasterExecutionGate.evaluate)

adaptive_risk_state.json and signal_lifecycle.csv are supporting/historical intelligence only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"

# ── Lifecycle authority ──
LIFECYCLE_AUTHORITY_PATH = RESULTS / "signal_lifecycle_effective.csv"
LIFECYCLE_HISTORICAL_PATH = RESULTS / "signal_lifecycle.csv"
TRADE_OPPORTUNITIES_DERIVED_PATH = RESULTS / "trade_opportunities.csv"

# ── Risk authority (execution restrictions) ──
RISK_AUTHORITY_PATH = RESULTS / "master_execution_gate.json"
RISK_INTELLIGENCE_PATH = RESULTS / "adaptive_risk_state.json"

# ── Allocation authority ──
ALLOCATION_AUTHORITY_PATH = RESULTS / "portfolio_allocation_recommendations.csv"
ALLOCATION_DERIVED_SUMMARY_PATH = RESULTS / "portfolio_allocation_summary.json"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def lifecycle_authority_path() -> Path:
    """Return path to authoritative lifecycle artifact."""
    return LIFECYCLE_AUTHORITY_PATH


def lifecycle_authority_exists() -> bool:
    p = LIFECYCLE_AUTHORITY_PATH
    return p.is_file() and p.stat().st_size > 0


def load_risk_execution_controls() -> Tuple[bool, bool, float, float, float]:
    """
    Read execution risk controls from master_execution_gate.json snapshot.

    Returns: (allow_new_orders, allow_new_trades, exposure_multiplier, max_position_weight, max_gross_exposure)
    """
    allow_new = True
    allow_trades = True
    exp_mult = 1.0
    max_pos_w = 0.10
    max_gross = 1.0

    snap = _load_json(RISK_AUTHORITY_PATH)
    if not snap:
        return allow_new, allow_trades, exp_mult, max_pos_w, max_gross

    details = snap.get("details") if isinstance(snap.get("details"), dict) else {}
    risk = details.get("risk") if isinstance(details.get("risk"), dict) else {}
    cpm = details.get("cpm") if isinstance(details.get("cpm"), dict) else {}

    if risk.get("global_kill_switch"):
        allow_new = False
    if risk.get("risk_on") is not None:
        try:
            allow_new = allow_new and bool(risk.get("risk_on", True))
        except Exception:
            pass
    if risk.get("allow_new_orders") is not None:
        try:
            allow_new = allow_new and bool(risk.get("allow_new_orders", True))
        except Exception:
            pass

    if cpm.get("allow_new_trades") is not None:
        try:
            allow_trades = bool(cpm.get("allow_new_trades", True))
        except Exception:
            pass

    try:
        em = cpm.get("exposure_multiplier")
        if em is not None:
            exp_mult = float(em)
    except (TypeError, ValueError):
        pass

    try:
        mw = risk.get("max_position_weight")
        if mw is not None and float(mw) > 0:
            max_pos_w = float(mw)
    except (TypeError, ValueError):
        pass

    try:
        mg = risk.get("max_gross_exposure")
        if mg is not None and float(mg) > 0:
            max_gross = float(mg)
    except (TypeError, ValueError):
        pass

    return allow_new, allow_trades, exp_mult, max_pos_w, max_gross


def risk_authority_snapshot() -> Optional[Dict[str, Any]]:
    """Return latest master execution gate snapshot (authoritative for batch execution)."""
    return _load_json(RISK_AUTHORITY_PATH)
