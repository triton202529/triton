"""Same-cycle F-stage identity sidecar for paper SELL readiness (Phase 198I-ZB).

Data-authored only. Never uses filesystem mtime as freshness or cycle identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from services.paper_sell_arm_authority import (
    BUY_ACTIONS,
    SELL_ACTIONS,
    _atomic_write_json,
    _parse_iso,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "results"
BINDING_NAME = "manage_positions_cycle_binding.json"
INTERLOCK_NAME = "paper_sell_exit_before_entry.json"
INVOCATION_ENV = "TRITON_MARKET_OPEN_INVOCATION_ID"
SCHEMA_VERSION = "198I-ZB-F-STAGE-BINDING-1"

CLASS_AUTHORITY_READY = "AUTHORITY_READY"
CLASS_ALREADY_SUBMITTED = "ALREADY_SUBMITTED"
CLASS_OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
CLASS_BLOCKED_OTHER_GATE = "BLOCKED_OTHER_GATE"
CLASS_INVALID = "INVALID"
CLASS_NOT_SELECTED = "NOT_SELECTED"

BLOCKER_MISSING_SELL_AUTHORITY = "MISSING_PAPER_SELL_AUTHORITY"
BLOCKER_OTHER_GATE = "OTHER_GATE"
BLOCKER_OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
BLOCKER_NONE = "NONE"

DECISION_MAY_CONTINUE = "MAY_CONTINUE"
DECISION_BLOCK = "BLOCK"
DECISION_CANDIDATE_AWARE = "CANDIDATE_AWARE"

SCOPE_NONE = "NONE"
SCOPE_GLOBAL_SAFETY_BLOCK = "GLOBAL_SAFETY_BLOCK"
SCOPE_CANDIDATE_FILTER = "CANDIDATE_FILTER"

REASON_PENDING_SELL_ARM = "ENTRY_BLOCKED_PENDING_SELL_ARM"
REASON_SELL_READINESS_UNKNOWN = "ENTRY_BLOCKED_SELL_READINESS_UNKNOWN"
REASON_SELL_OUTCOME_UNKNOWN = "ENTRY_BLOCKED_SELL_OUTCOME_UNKNOWN"
REASON_SELL_PRECONDITION = "ENTRY_BLOCKED_SELL_PRECONDITION"
REASON_CRITICAL_EXIT = "ENTRY_BLOCKED_CRITICAL_EXIT_PENDING"
REASON_INVOCATION_MISMATCH = "ENTRY_BLOCKED_INVOCATION_MISMATCH"
REASON_INVOCATION_MISSING = "ENTRY_BLOCKED_INVOCATION_MISSING"

INTERLOCK_SCHEMA = "198I-ZB-R-EXIT-BEFORE-ENTRY-1"

F2_OK = 0
F2_PENDING_SELL_ARM = 10
F2_OUTCOME_UNKNOWN = 11
F2_OTHER_GATE = 12
F2_GENERATION_FAILURE = 13
F2_SOURCE_UNBOUND = 14


def binding_path(results_root: Optional[Path] = None) -> Path:
    return Path(results_root or DEFAULT_RESULTS) / BINDING_NAME


def interlock_path(results_root: Optional[Path] = None) -> Path:
    return Path(results_root or DEFAULT_RESULTS) / INTERLOCK_NAME


def resolve_invocation_id() -> str:
    env = str(os.environ.get(INVOCATION_ENV) or "").strip()
    if env:
        return env
    return f"fstage-{uuid.uuid4().hex}"


def sha256_file_bytes(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def selected_row_hash(rows: Sequence[Dict[str, Any]]) -> str:
    blob = "|".join(
        sorted(
            f"{r.get('symbol')}:{r.get('final_action')}:{r.get('qty')}:"
            f"{r.get('selected_for_execution')}:{r.get('reason_code')}:{r.get('skip_reason')}:"
            f"{r.get('status')}"
            for r in rows
        )
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("timestamp_must_be_timezone_aware")
    return dt.astimezone(timezone.utc).isoformat()


def write_cycle_binding(
    *,
    results_root: Path,
    invocation_id: str,
    plan_generated_at: str,
    f_stage_status: str,
    now: datetime,
    mode: str = "paper",
    live_execution_blocked: bool = True,
    orders_executed: int = 0,
    submitted_symbols: Optional[Sequence[str]] = None,
    submitted_outcomes: Optional[Dict[str, str]] = None,
    block_reasons: Optional[Sequence[str]] = None,
    dispatch_interlock_state: Optional[str] = None,
    operator_execution_required: bool = False,
    remaining_blocker: str = BLOCKER_NONE,
    outcome_unknown: bool = False,
) -> Dict[str, Any]:
    results_root = Path(results_root)
    csv_path = results_root / "manage_positions_plan.csv"
    json_path = results_root / "manage_positions_plan.json"
    rows: List[Dict[str, Any]] = []
    if csv_path.is_file():
        import csv

        rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    plan_ts = _parse_iso(plan_generated_at)
    if plan_ts is None:
        raise ValueError("plan_generated_at_unparseable_or_naive")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "invocation_id": str(invocation_id),
        "cycle_id": str(invocation_id),
        "market_open_invocation_id": str(invocation_id),
        "plan_generated_at": plan_ts.astimezone(timezone.utc).isoformat(),
        "f_stage_completed_at": _iso(now),
        "f_stage_status": str(f_stage_status),
        "mode": str(mode or "").upper(),
        "live_execution_blocked": bool(live_execution_blocked),
        "source_plan_sha256": sha256_file_bytes(csv_path) or sha256_file_bytes(json_path),
        "selected_row_hash": selected_row_hash(rows),
        "positions_snapshot_sha256": sha256_file_bytes(results_root / "positions_snapshot.csv"),
        "open_orders_snapshot_sha256": sha256_file_bytes(results_root / "open_orders_snapshot.csv"),
        "orders_executed": int(orders_executed or 0),
        "submitted_symbols": [str(s).upper() for s in (submitted_symbols or [])],
        "submitted_outcomes": {
            str(k).upper(): str(v).upper() for k, v in (submitted_outcomes or {}).items()
        },
        "block_reasons": list(block_reasons or []),
        "dispatch_interlock_state": dispatch_interlock_state,
        "operator_execution_required": bool(operator_execution_required),
        "remaining_blocker": remaining_blocker,
        "outcome_unknown": bool(outcome_unknown),
        "creates_no_authority": True,
        "orders_submitted": False,
    }
    _atomic_write_json(binding_path(results_root), payload, require_full_durability=False)
    return payload


def load_cycle_binding(results_root: Path) -> Dict[str, Any]:
    path = binding_path(results_root)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def derive_remaining_blocker(
    *,
    block_reasons: Sequence[str],
    operator_execution_required: bool,
    orders_executed: int,
    outcome_unknown: bool,
    dispatch_interlock_state: Optional[str],
) -> str:
    if outcome_unknown:
        return BLOCKER_OUTCOME_UNKNOWN
    joined = " ".join(str(x) for x in block_reasons)
    state = str(dispatch_interlock_state or "")
    if (
        operator_execution_required
        or "DISPATCH_BLOCKED_APPROVAL_REQUIRED" in joined
        or ("APPROVAL_REQUIRED" in state)
    ):
        return BLOCKER_MISSING_SELL_AUTHORITY
    if int(orders_executed or 0) > 0 and not block_reasons:
        return BLOCKER_NONE
    if block_reasons or state:
        return BLOCKER_OTHER_GATE
    return BLOCKER_NONE


def classify_plan_rows(
    rows: Sequence[Dict[str, Any]],
    binding: Dict[str, Any],
) -> List[Dict[str, Any]]:
    submitted = {str(s).upper() for s in (binding.get("submitted_symbols") or [])}
    outcomes = binding.get("submitted_outcomes") or {}
    blocker = str(binding.get("remaining_blocker") or BLOCKER_NONE)
    outcome_unknown = bool(binding.get("outcome_unknown"))
    out: List[Dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper().strip()
        selected = str(row.get("selected_for_execution") or "").lower() == "true"
        action = str(
            row.get("final_action") or row.get("management_action_candidate") or ""
        ).upper()
        rec: Dict[str, Any] = {
            "symbol": symbol,
            "action": action,
            "qty": row.get("qty"),
            "selected_for_execution": selected,
            "reason_code": row.get("reason_code"),
            "force_exit_override": row.get("force_exit_override"),
            "rotation_id": row.get("rotation_id") or row.get("rotation_batch_id"),
            "risk_objective": row.get("risk_objective") or row.get("objective"),
        }
        if not selected:
            rec["classification"] = CLASS_NOT_SELECTED
            out.append(rec)
            continue
        if action in BUY_ACTIONS:
            rec["classification"] = CLASS_INVALID
            rec["reason"] = "buy_or_add_in_sell_cycle"
            out.append(rec)
            continue
        if action not in SELL_ACTIONS:
            rec["classification"] = CLASS_INVALID
            rec["reason"] = "action_not_sell"
            out.append(rec)
            continue
        try:
            qty = float(row.get("qty") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if not symbol or qty <= 0:
            rec["classification"] = CLASS_INVALID
            rec["reason"] = "invalid_symbol_or_qty"
            out.append(rec)
            continue
        if symbol in submitted or str(outcomes.get(symbol) or "").upper() == "SUBMITTED":
            rec["classification"] = CLASS_ALREADY_SUBMITTED
            rec["outcome"] = str(outcomes.get(symbol) or "SUBMITTED")
            out.append(rec)
            continue
        if outcome_unknown or blocker == BLOCKER_OUTCOME_UNKNOWN:
            rec["classification"] = CLASS_OUTCOME_UNKNOWN
            rec["reason"] = "reconciliation_required"
            out.append(rec)
            continue
        if blocker == BLOCKER_MISSING_SELL_AUTHORITY:
            rec["classification"] = CLASS_AUTHORITY_READY
            rec["remaining_blocker"] = BLOCKER_MISSING_SELL_AUTHORITY
            out.append(rec)
            continue
        if blocker == BLOCKER_NONE:
            rec["classification"] = CLASS_OUTCOME_UNKNOWN
            rec["reason"] = "selected_sell_without_terminal_outcome"
            out.append(rec)
            continue
        rec["classification"] = CLASS_BLOCKED_OTHER_GATE
        rec["remaining_blocker"] = blocker or BLOCKER_OTHER_GATE
        out.append(rec)
    return out


def write_entry_interlock(
    *,
    results_root: Path,
    invocation_id: str,
    now: datetime,
    decision: str,
    reason: str,
    packet_status: str,
    interlock_scope: str = SCOPE_NONE,
    global_block_reason: Optional[str] = None,
    pending_sell_symbols: Optional[Sequence[str]] = None,
    pending_sell_actions: Optional[Sequence[Dict[str, Any]]] = None,
    risk_criticality: str = "NONE",
    affected_sectors: Optional[Sequence[str]] = None,
    source_plan_sha256: Optional[str] = None,
    selected_row_hash_value: Optional[str] = None,
) -> Dict[str, Any]:
    pending_actions = list(pending_sell_actions or [])
    pending_syms = [str(s).upper() for s in (pending_sell_symbols or [])]
    if not pending_syms:
        pending_syms = [
            str(a.get("symbol") or "").upper() for a in pending_actions if a.get("symbol")
        ]
    payload = {
        "schema_version": INTERLOCK_SCHEMA,
        "generated_at": _iso(now),
        "invocation_id": invocation_id,
        "cycle_id": invocation_id,
        "market_open_invocation_id": invocation_id,
        "decision": decision,
        "reason": reason,
        "packet_status": packet_status,
        "interlock_scope": interlock_scope,
        "global_block_reason": global_block_reason,
        "pending_sell_symbols": pending_syms,
        "pending_sell_actions": pending_actions,
        "risk_criticality": risk_criticality,
        "affected_sectors": list(affected_sectors or []),
        "proceeds_available_for_entry": 0,
        "entry_dependency_policy": "SELL_UNEXECUTED_ZERO_PROCEEDS",
        "source_plan_sha256": source_plan_sha256,
        "selected_row_hash": selected_row_hash_value,
        "orders_submitted": False,
        "creates_no_authority": True,
    }
    _atomic_write_json(interlock_path(results_root), payload, require_full_durability=False)
    return payload


def evaluate_exit_before_entry(
    *,
    results_root: Path,
    now: Optional[datetime] = None,
    ttl_minutes: int = 30,
    expected_invocation_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    expected = str(
        expected_invocation_id
        if expected_invocation_id is not None
        else (os.environ.get(INVOCATION_ENV) or "")
    ).strip()
    path = interlock_path(results_root)
    if not expected:
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": REASON_INVOCATION_MISSING,
            "packet_status": "INVOCATION_MISSING",
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": REASON_INVOCATION_MISSING,
        }
    if not path.is_file():
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": REASON_SELL_READINESS_UNKNOWN,
            "packet_status": "MISSING_INTERLOCK",
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": REASON_SELL_READINESS_UNKNOWN,
        }
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": REASON_SELL_READINESS_UNKNOWN,
            "packet_status": "MALFORMED_INTERLOCK",
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": REASON_SELL_READINESS_UNKNOWN,
        }
    if not isinstance(doc, dict):
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": REASON_SELL_READINESS_UNKNOWN,
            "packet_status": "MALFORMED_INTERLOCK",
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": REASON_SELL_READINESS_UNKNOWN,
        }
    generated = _parse_iso(doc.get("generated_at"))
    if generated is None:
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": REASON_SELL_READINESS_UNKNOWN,
            "packet_status": "SOURCE_UNBOUND",
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": REASON_SELL_READINESS_UNKNOWN,
        }
    age = now - generated
    if age.total_seconds() > ttl_minutes * 60 or generated > now:
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": REASON_SELL_READINESS_UNKNOWN,
            "packet_status": "STALE_INTERLOCK",
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": REASON_SELL_READINESS_UNKNOWN,
        }
    inv = str(
        doc.get("market_open_invocation_id")
        or doc.get("invocation_id")
        or doc.get("cycle_id")
        or ""
    ).strip()
    if not inv:
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": REASON_INVOCATION_MISSING,
            "packet_status": "INTERLOCK_INVOCATION_MISSING",
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": REASON_INVOCATION_MISSING,
        }
    if inv != expected:
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": REASON_INVOCATION_MISMATCH,
            "packet_status": "INVOCATION_MISMATCH",
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": REASON_INVOCATION_MISMATCH,
            "invocation_id": inv,
            "expected_invocation_id": expected,
        }
    decision = str(doc.get("decision") or "")
    reason = str(doc.get("reason") or REASON_SELL_PRECONDITION)
    scope = str(doc.get("interlock_scope") or "")
    shared = {
        "pending_sell_symbols": list(doc.get("pending_sell_symbols") or []),
        "pending_sell_actions": list(doc.get("pending_sell_actions") or []),
        "risk_criticality": str(doc.get("risk_criticality") or ""),
        "affected_sectors": list(doc.get("affected_sectors") or []),
        "proceeds_available_for_entry": 0,
        "entry_dependency_policy": doc.get("entry_dependency_policy")
        or "SELL_UNEXECUTED_ZERO_PROCEEDS",
        "invocation_id": inv,
        "cycle_id": inv,
        "market_open_invocation_id": inv,
        "source_plan_sha256": doc.get("source_plan_sha256"),
        "selected_row_hash": doc.get("selected_row_hash"),
        "generated_at": doc.get("generated_at"),
    }
    if decision == DECISION_BLOCK or scope == SCOPE_GLOBAL_SAFETY_BLOCK:
        return {
            "ok": False,
            "decision": DECISION_BLOCK,
            "reason": reason or REASON_SELL_PRECONDITION,
            "packet_status": str(doc.get("packet_status") or ""),
            "interlock_scope": SCOPE_GLOBAL_SAFETY_BLOCK,
            "global_block_reason": doc.get("global_block_reason") or reason,
            **shared,
        }
    if decision == DECISION_CANDIDATE_AWARE or scope == SCOPE_CANDIDATE_FILTER:
        return {
            "ok": True,
            "decision": DECISION_CANDIDATE_AWARE,
            "reason": reason or "ok",
            "packet_status": str(doc.get("packet_status") or "AUTHORITY_READY"),
            "interlock_scope": SCOPE_CANDIDATE_FILTER,
            "global_block_reason": None,
            **shared,
            "risk_criticality": str(doc.get("risk_criticality") or "ORDINARY"),
        }
    return {
        "ok": True,
        "decision": DECISION_MAY_CONTINUE,
        "reason": reason or "ok",
        "packet_status": str(doc.get("packet_status") or "VALID_EMPTY"),
        "interlock_scope": SCOPE_NONE,
        "global_block_reason": None,
        **shared,
        "pending_sell_symbols": [],
        "pending_sell_actions": [],
        "risk_criticality": "NONE",
    }
