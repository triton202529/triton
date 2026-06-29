"""
order_guard_snapshot_authority.py
Phase 197P — Reconcile guard_snapshot freshness authority with broker/open-orders snapshots.
Read-only classification; optional guard artifact refresh from broker authority (no orders submitted).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"

GUARD_PATHS = [RESULTS / "guard_snapshot.json", LIVE / "guard_snapshot.json"]
OPEN_ORDERS_JSON = RESULTS / "open_orders_snapshot.csv"
RECENT_ORDERS_JSON = RESULTS / "recent_orders.csv"
POSITIONS_JSON = RESULTS / "positions_snapshot.csv"
LIVE_ORDERS_RESULTS = RESULTS / "live_orders.csv"
LIVE_ORDERS_LIVE = LIVE / "live_orders.csv"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"

ORDER_GUARD_FRESH = "ORDER_GUARD_FRESH"
ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS = "ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS"
ORDER_GUARD_STALE_AND_UNVERIFIED = "ORDER_GUARD_STALE_AND_UNVERIFIED"
ORDER_GUARD_OPEN_ORDERS_PRESENT = "ORDER_GUARD_OPEN_ORDERS_PRESENT"
ORDER_GUARD_BROKER_SNAPSHOT_FAILED = "ORDER_GUARD_BROKER_SNAPSHOT_FAILED"

DEFAULT_GUARD_MAX_AGE_MINUTES = 30.0
DEFAULT_OPEN_ORDERS_MAX_AGE_MINUTES = 120.0

GUARD_WRITERS = [
    "services/reconcile_state.py::_write_guard_snapshot",
    "services/guard_control.py",
    "services/guard_set.py",
    "services/guard_auto.py::write_guard_snapshot",
    "services/write_guard_snapshot.py",
    "services/master_execution_gate.py::try_refresh_stale_guard_snapshot_if_clear",
    "services/order_guard_snapshot_authority.py::refresh_guard_snapshot_from_broker_authority",
]

GUARD_READERS = [
    "services/master_execution_gate.py::_check_guard",
    "services/paper_entry_loop_stage.py",
    "services/preflight_checks.py",
    "services/safety_gate.py",
    "services/live_gate.py",
    "services/defensive_exit_market_hours_governance_recheck.py",
    "services/order_guard_snapshot_authority.py",
    "services/dashboard_status_sync.py",
    "services/reconcile_state.py",
    "services/capital_deployment_engine.py",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _file_age_minutes(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (_utc_now() - mtime).total_seconds() / 60.0


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _resolve_guard_path() -> Optional[Path]:
    existing = [p for p in GUARD_PATHS if p.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _guard_blocked(doc: Dict[str, Any]) -> bool:
    if not doc:
        return False
    if doc.get("blocked") or doc.get("kill_switch"):
        return True
    mode = str(doc.get("mode") or "").upper()
    return mode in ("FREEZE", "LOCKDOWN", "BLOCKED")


def guard_snapshot_inventory() -> Dict[str, Any]:
    """Document writers/readers and why snapshot_live_orders does not historically refresh guard."""
    return {
        "guard_snapshot_paths": [str(p.relative_to(ROOT)).replace("\\", "/") for p in GUARD_PATHS],
        "writers": GUARD_WRITERS,
        "readers": GUARD_READERS,
        "why_snapshot_live_orders_did_not_refresh_guard": (
            "snapshot_live_orders writes open_orders_snapshot.csv, recent_orders.csv, and "
            "positions_snapshot.csv only; guard_snapshot.json is maintained by reconcile/guard "
            "pipelines. Phase 197P adds optional broker-authority refresh when broker snapshot "
            "succeeds with zero open orders."
        ),
        "live_orders_event_log_paths": [
            str(LIVE_ORDERS_RESULTS.relative_to(ROOT)).replace("\\", "/"),
            str(LIVE_ORDERS_LIVE.relative_to(ROOT)).replace("\\", "/"),
        ],
        "open_orders_authority_path": str(OPEN_ORDERS_JSON.relative_to(ROOT)).replace("\\", "/"),
    }


def refresh_guard_snapshot_from_broker_authority(
    *,
    broker_snapshot_ok: bool,
    open_orders_count: int,
    mode: str = "paper",
    broker_orders_submitted: bool = False,
) -> Tuple[bool, str, Optional[Path]]:
    """
    Refresh guard_snapshot.json from broker open-orders authority (no orders submitted).
    Returns (updated, reason, path).
    """
    if not broker_snapshot_ok:
        return False, "broker_snapshot_not_ok", None
    if open_orders_count != 0:
        return False, f"open_orders_count={open_orders_count}", None
    if broker_orders_submitted:
        return False, "broker_orders_submitted", None

    path = RESULTS / "guard_snapshot.json"
    existing = _read_json(path)
    if _guard_blocked(existing):
        return False, "guard_blocked_cannot_refresh", path

    ts = _utc_now_iso()
    payload = {
        **existing,
        "timestamp": ts,
        "generated_at": ts,
        "updated_at": ts,
        "mode": mode,
        "reason": "Broker open-orders snapshot authority refresh (zero open orders)",
        "blocked": False,
        "kill_switch": False,
        "ok": True,
        "clear": True,
        "broker_snapshot_ok": True,
        "open_orders_count": 0,
        "source_paths": {
            "open_orders_snapshot": str(OPEN_ORDERS_JSON.relative_to(ROOT)).replace("\\", "/"),
            "recent_orders_snapshot": str(RECENT_ORDERS_JSON.relative_to(ROOT)).replace("\\", "/"),
            "positions_snapshot": str(POSITIONS_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "source_authority": "BROKER_OPEN_ORDERS_SNAPSHOT",
        "broker_orders_submitted": False,
        "phase": "197P",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return True, "refreshed_from_broker_authority", path


def classify_order_guard_authority(
    *,
    broker_snapshot_ok: bool,
    open_orders_count: Optional[int] = None,
    guard_max_age_minutes: float = DEFAULT_GUARD_MAX_AGE_MINUTES,
    open_orders_max_age_minutes: float = DEFAULT_OPEN_ORDERS_MAX_AGE_MINUTES,
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify order guard authority state. Returns (state, reason, details)."""
    guard_path = _resolve_guard_path()
    guard_age = _file_age_minutes(guard_path) if guard_path else None
    guard_fresh = guard_age is not None and guard_age <= guard_max_age_minutes
    oo_age = _file_age_minutes(OPEN_ORDERS_JSON)
    oo_fresh = oo_age is not None and oo_age <= open_orders_max_age_minutes
    oo_count = (
        open_orders_count if open_orders_count is not None else _count_csv_rows(OPEN_ORDERS_JSON)
    )
    recent_count = _count_csv_rows(RECENT_ORDERS_JSON)

    details: Dict[str, Any] = {
        "guard_snapshot_path": (
            str(guard_path.relative_to(ROOT)).replace("\\", "/") if guard_path else None
        ),
        "guard_snapshot_exists": guard_path is not None,
        "guard_snapshot_age_minutes": round(guard_age, 2) if guard_age is not None else None,
        "guard_snapshot_fresh": guard_fresh if guard_path else None,
        "open_orders_snapshot_path": str(OPEN_ORDERS_JSON.relative_to(ROOT)).replace("\\", "/"),
        "open_orders_snapshot_exists": OPEN_ORDERS_JSON.is_file(),
        "open_orders_snapshot_age_minutes": round(oo_age, 2) if oo_age is not None else None,
        "open_orders_snapshot_fresh": oo_fresh if OPEN_ORDERS_JSON.is_file() else None,
        "open_orders_count": oo_count,
        "recent_orders_count": recent_count,
        "broker_snapshot_ok": broker_snapshot_ok,
    }

    if not broker_snapshot_ok:
        return ORDER_GUARD_BROKER_SNAPSHOT_FAILED, "broker_snapshot_failed", details

    if oo_count > 0:
        return ORDER_GUARD_OPEN_ORDERS_PRESENT, f"open_orders_count={oo_count}", details

    if guard_fresh:
        return ORDER_GUARD_FRESH, "guard_snapshot_within_ttl", details

    if oo_fresh and oo_count == 0:
        return (
            ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS,
            "guard_stale_broker_open_orders_authoritative_zero",
            details,
        )

    return ORDER_GUARD_STALE_AND_UNVERIFIED, "guard_stale_broker_unverified", details


def evaluate_order_guard_authority(
    *,
    broker_snapshot_ok: bool = True,
    open_orders_count: Optional[int] = None,
    refresh_guard_if_eligible: bool = False,
    mode: str = "paper",
) -> Dict[str, Any]:
    """Full order guard authority evaluation."""
    from services.live_block_classifier import evaluate_live_block_status

    state, reason, details = classify_order_guard_authority(
        broker_snapshot_ok=broker_snapshot_ok,
        open_orders_count=open_orders_count,
    )

    updated_guard = False
    update_mode = "none"
    update_reason = ""
    if refresh_guard_if_eligible and state == ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS:
        updated_guard, update_reason, _ = refresh_guard_snapshot_from_broker_authority(
            broker_snapshot_ok=broker_snapshot_ok,
            open_orders_count=int(details.get("open_orders_count") or 0),
            mode=mode,
        )
        update_mode = "broker_authority_refresh" if updated_guard else "refresh_skipped"
        if updated_guard:
            state, reason, details = classify_order_guard_authority(
                broker_snapshot_ok=broker_snapshot_ok,
                open_orders_count=open_orders_count,
            )

    live_block = evaluate_live_block_status(paper_mode=True)
    eligible_for_execution = False

    gate_freshness_note = ""
    if state == ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS:
        gate_freshness_note = (
            "guard_snapshot stale but broker/open-orders snapshot fresh with zero open orders — "
            "broker authority overrides stale guard timestamp for paper observation"
        )
    elif state == ORDER_GUARD_FRESH:
        gate_freshness_note = "guard_snapshot fresh"
    elif state == ORDER_GUARD_OPEN_ORDERS_PRESENT:
        gate_freshness_note = "open orders present — in-flight protection active"
    elif state == ORDER_GUARD_BROKER_SNAPSHOT_FAILED:
        gate_freshness_note = "broker snapshot failed — cannot verify order guard"
    elif state == ORDER_GUARD_STALE_AND_UNVERIFIED:
        gate_freshness_note = "guard stale and broker/open-orders authority unverified"

    return {
        "timestamp": _utc_now_iso(),
        "phase": "197P",
        "order_guard_authority_state": state,
        "order_guard_authority_reason": reason,
        "eligible_for_execution": eligible_for_execution,
        "updated_guard_snapshot": updated_guard,
        "update_mode": update_mode,
        "update_reason": update_reason,
        "source_authority": (
            "BROKER_OPEN_ORDERS_SNAPSHOT" if updated_guard else details.get("source_authority")
        ),
        "live_blocked": live_block.get("live_blocked"),
        "live_block_reason": live_block.get("live_block_reason"),
        "real_approval_artifact_exists": REAL_APPROVAL_PATH.is_file(),
        "gate_freshness_note": gate_freshness_note,
        "inventory": guard_snapshot_inventory(),
        **details,
    }
