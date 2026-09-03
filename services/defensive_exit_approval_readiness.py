"""Canonical paper SELL readiness packet producer.

Writes only data/results/defensive_exit_approval_readiness.json (or an injected
results_root). Creates no authority, challenge, permit, or broker call.
Failed generation does not overwrite a previous packet, except a same-cycle
protected SELL dispatch which must overwrite any armable packet with a
non-armable reconciliation-required packet.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.manage_positions_cycle_binding import (
    BLOCKER_MISSING_SELL_AUTHORITY,
    BLOCKER_NONE,
    BLOCKER_OTHER_GATE,
    BLOCKER_OUTCOME_UNKNOWN,
    CLASS_ALREADY_SUBMITTED,
    CLASS_AUTHORITY_READY,
    CLASS_BLOCKED_OTHER_GATE,
    CLASS_INVALID,
    CLASS_OUTCOME_UNKNOWN,
    DECISION_BLOCK,
    DECISION_CANDIDATE_AWARE,
    DECISION_MAY_CONTINUE,
    F2_GENERATION_FAILURE,
    F2_OK,
    F2_OTHER_GATE,
    F2_OUTCOME_UNKNOWN,
    F2_PENDING_SELL_ARM,
    F2_SOURCE_UNBOUND,
    INVOCATION_ENV,
    REASON_CRITICAL_EXIT,
    REASON_PENDING_SELL_ARM,
    REASON_SELL_OUTCOME_UNKNOWN,
    REASON_SELL_PRECONDITION,
    REASON_SELL_READINESS_UNKNOWN,
    SCOPE_CANDIDATE_FILTER,
    SCOPE_GLOBAL_SAFETY_BLOCK,
    SCOPE_NONE,
    classify_plan_rows,
    load_cycle_binding,
    selected_row_hash,
    sha256_file_bytes,
    write_entry_interlock,
)
from services.paper_sell_entry_dependency import (
    is_critical_pending_sell,
    pending_sell_sector,
)
from services.paper_sell_arm_authority import (
    APPROVAL_CLI_CAP,
    BUY_ACTIONS,
    READINESS_REFERENCE_TTL_MINUTES,
    READINESS_STATUS_OK,
    SELL_ACTIONS,
    _atomic_write_json,
    _parse_iso,
    readiness_packet_valid_until,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "results"
PACKET_NAME = "defensive_exit_approval_readiness.json"
PRE_PLAN_DISPATCH_EVIDENCE_NAME = "manage_positions_pre_plan_dispatch.json"
PRE_PLAN_DISPATCH_SCHEMA = "198T-O-PRE-PLAN-DISPATCH-1"
SAME_CYCLE_DISPATCH_REASON = "SAME_CYCLE_SELL_DISPATCH_REQUIRES_NEXT_CYCLE_RECONCILIATION"
STATUS_UNAVAILABLE_SAME_CYCLE = "UNAVAILABLE_SAME_CYCLE_DISPATCH_RECONCILIATION"
SAME_CYCLE_SUPPRESSION_JSONL = "defensive_exit_same_cycle_suppression.jsonl"
SCHEMA_VERSION = "198I-ZA-SELL-READINESS-1"
PRODUCER_ID = "services.defensive_exit_approval_readiness"
CANONICAL_MODE = "PAPER"
LIVE_EXECUTION_BLOCKED = "BLOCKED"
# Explicit entry-only vocabulary that may appear on a mixed F plan.
# These rows are excluded from the SELL packet; they are never converted
# into SELL actions. Unknown/ambiguous actions still fail closed.
RECOGNIZED_ENTRY_ONLY_ACTIONS = frozenset(set(BUY_ACTIONS) | {"OPEN_NEW"})
CONTRADICTORY_BUY_SIDES = frozenset({"BUY", "B", "BID", "LONG_ENTRY", "ENTRY"})


def default_results_root() -> Path:
    return DEFAULT_RESULTS


def packet_path(results_root: Optional[Path] = None) -> Path:
    return Path(results_root or default_results_root()) / PACKET_NAME


def pre_plan_dispatch_evidence_path(results_root: Optional[Path] = None) -> Path:
    return Path(results_root or default_results_root()) / PRE_PLAN_DISPATCH_EVIDENCE_NAME


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("generated_at_must_be_timezone_aware")
    return dt.astimezone(timezone.utc).isoformat()


def derive_pre_plan_dispatch_performed(pre_plan: Dict[str, Any]) -> bool:
    """True when this invocation crossed a holdings-changing dispatch boundary."""
    attempted = bool(pre_plan.get("attempted"))
    dispatched = bool(pre_plan.get("dispatched"))
    consumed = bool(pre_plan.get("consumed"))
    try:
        placed = int(pre_plan.get("broker_submissions") or 0)
    except (TypeError, ValueError):
        placed = 0
    submit = pre_plan.get("submit") if isinstance(pre_plan.get("submit"), dict) else {}
    try:
        placed = max(placed, int(submit.get("placed") or 0))
    except (TypeError, ValueError):
        pass
    submitted = bool(pre_plan.get("broker_orders_submitted")) or placed > 0 or dispatched
    return bool(attempted and (dispatched or consumed or submitted or placed > 0))


def build_pre_plan_dispatch_evidence(
    pre_plan: Optional[Dict[str, Any]] = None,
    *,
    invocation_id: str,
    now: Optional[datetime] = None,
    session_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Current-invocation dispatch sidecar. Never contacts a broker."""
    from services.manage_positions_cycle_binding import resolve_invocation_id

    pre_plan = dict(pre_plan or {})
    now = now or _utc_now()
    inv = str(invocation_id or resolve_invocation_id() or "").strip()
    submit = pre_plan.get("submit") if isinstance(pre_plan.get("submit"), dict) else {}
    try:
        placed = int(pre_plan.get("broker_submissions") or submit.get("placed") or 0)
    except (TypeError, ValueError):
        placed = 0
    try:
        failed = int(submit.get("failed") or 0)
    except (TypeError, ValueError):
        failed = 0
    dispatched = bool(pre_plan.get("dispatched"))
    consumed = bool(pre_plan.get("consumed"))
    attempted = bool(pre_plan.get("attempted"))
    performed_raw = pre_plan.get("dispatch_performed")
    if performed_raw is None:
        performed = derive_pre_plan_dispatch_performed(pre_plan)
        evidence_status = "OK"
    else:
        if isinstance(performed_raw, bool):
            performed = performed_raw
            evidence_status = "OK"
        else:
            performed = True
            evidence_status = "AMBIGUOUS"
    broker_submitted = bool(dispatched or placed > 0 or pre_plan.get("broker_orders_submitted"))
    terminal = str(
        pre_plan.get("dispatch_terminal_status")
        or ("TERMINAL_SUCCESS" if dispatched and placed > 0 else (pre_plan.get("reason") or "NONE"))
    )
    sell_block_class = str(pre_plan.get("sell_block_class") or "")
    sell_blocked = bool(pre_plan.get("sell_blocked"))
    if sell_block_class == "HARD_CYCLE_FAILURE":
        pipeline_health_class = "SYSTEM_FAILURE"
    elif sell_blocked or sell_block_class == "SELL_PERMISSION_UNUSABLE":
        pipeline_health_class = "SELL_NOT_AUTHORIZED"
    else:
        pipeline_health_class = "OK"
    return {
        "schema_version": PRE_PLAN_DISPATCH_SCHEMA,
        "generated_at": _iso_utc(now),
        "invocation_id": inv,
        "cycle_id": inv,
        "market_open_invocation_id": inv,
        "session_date": session_date,
        "dispatch_id": str(pre_plan.get("dispatch_intent_id") or pre_plan.get("dispatch_id") or ""),
        "authority_id": str(pre_plan.get("authority_id") or ""),
        "attempted": attempted,
        "dispatch_performed": performed,
        "consumed": consumed,
        "dispatched": dispatched,
        "broker_orders_submitted": broker_submitted,
        "placed_count": placed,
        "failed_count": failed,
        "dispatch_terminal_status": terminal,
        "reason": str(pre_plan.get("reason") or ""),
        "evidence_status": evidence_status,
        "creates_no_authority": True,
        "live_execution_permitted": False,
        "sell_blocked": sell_blocked,
        "sell_block_class": sell_block_class,
        "pipeline_health_class": pipeline_health_class,
    }


def write_pre_plan_dispatch_evidence(
    results_root: Path,
    pre_plan: Optional[Dict[str, Any]] = None,
    *,
    invocation_id: str = "",
    now: Optional[datetime] = None,
    session_date: Optional[str] = None,
) -> Dict[str, Any]:
    evidence = build_pre_plan_dispatch_evidence(
        pre_plan,
        invocation_id=invocation_id,
        now=now,
        session_date=session_date,
    )
    path = pre_plan_dispatch_evidence_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, evidence, require_full_durability=False)
    return evidence


def load_pre_plan_dispatch_evidence(results_root: Path) -> Dict[str, Any]:
    return _read_json(pre_plan_dispatch_evidence_path(results_root))


def classify_same_cycle_dispatch(
    *,
    invocation_id: str,
    results_root: Path,
) -> Tuple[str, Dict[str, Any]]:
    """Return ('none'|'suppress'|'ambiguous', evidence) for the current invocation."""
    evidence = load_pre_plan_dispatch_evidence(results_root)
    if not evidence:
        return "none", {}
    ev_inv = str(
        evidence.get("invocation_id")
        or evidence.get("market_open_invocation_id")
        or evidence.get("cycle_id")
        or ""
    ).strip()
    current = str(invocation_id or "").strip()
    if not current or not ev_inv or ev_inv != current:
        return "none", evidence
    status = str(evidence.get("evidence_status") or "").upper()
    performed = evidence.get("dispatch_performed")
    if status == "AMBIGUOUS" or performed not in (True, False):
        return "ambiguous", evidence
    if performed is True:
        return "suppress", evidence
    return "none", evidence


def append_same_cycle_suppression_record(
    results_root: Path,
    *,
    invocation_id: str,
    authority_id: Any = None,
    dispatch_id: Any = None,
    classification: str = STATUS_UNAVAILABLE_SAME_CYCLE,
    reason: str = SAME_CYCLE_DISPATCH_REASON,
    now: Optional[datetime] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append-only same-cycle suppression evidence. Does not block later-cycle rebuild."""
    now = now or _utc_now()
    path = Path(results_root) / SAME_CYCLE_SUPPRESSION_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _iso_utc(now),
        "classification": classification,
        "reason": reason,
        "invocation_id": invocation_id,
        "authority_id": authority_id,
        "dispatch_id": dispatch_id,
        "creates_no_authority": True,
    }
    if extra:
        row.update(extra)
        row["creates_no_authority"] = True
    line = json.dumps(row, default=str) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    return path


def write_same_cycle_dispatch_reconciliation_packet(
    *,
    results_root: Path,
    binding: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Overwrite canonical packet with a non-armable same-cycle reconciliation marker."""
    now = now or _utc_now()
    results_root = Path(results_root)
    dest = packet_path(results_root)
    binding = dict(binding or {})
    evidence = dict(evidence or {})
    inv = str(
        binding.get("invocation_id")
        or evidence.get("invocation_id")
        or os.environ.get(INVOCATION_ENV)
        or ""
    )
    packet = build_canonical_packet(
        now=now,
        actions=[],
        cycle_id=inv or "unbound",
        run_id=str(
            binding.get("plan_generated_at") or evidence.get("generated_at") or _iso_utc(now)
        ),
        batch_id=f"manage-{(inv or 'unbound')[:16]}",
        plan_id=f"manage_positions_plan@{binding.get('plan_generated_at') or 'unbound'}",
        plan_hash=str(binding.get("selected_row_hash") or "same-cycle-dispatch-reconciliation"),
        generation_status="ok",
        extra={
            "invocation_id": inv,
            "market_open_invocation_id": inv,
            "plan_generated_at": binding.get("plan_generated_at"),
            "f_stage_completed_at": binding.get("f_stage_completed_at"),
            "f_stage_status": binding.get("f_stage_status"),
            "source_plan_sha256": binding.get("source_plan_sha256"),
            "selected_row_hash": binding.get("selected_row_hash"),
            "arm_eligibility": "UNAVAILABLE",
            "packet_status": SAME_CYCLE_DISPATCH_REASON,
            "reason": SAME_CYCLE_DISPATCH_REASON,
            "dispatch_id": evidence.get("dispatch_id"),
            "authority_id": evidence.get("authority_id"),
            "dispatch_performed": evidence.get("dispatch_performed"),
            "dispatch_terminal_status": evidence.get("dispatch_terminal_status"),
            "placed_count": evidence.get("placed_count"),
            "session_date": evidence.get("session_date") or binding.get("session_date"),
            "same_cycle_dispatch_evidence": evidence,
        },
    )
    packet["status"] = STATUS_UNAVAILABLE_SAME_CYCLE
    packet["approval_ready_symbols"] = []
    packet["approval_ready_actions"] = []
    packet["total_proposed_notional"] = 0.0
    packet["creates_no_authority"] = True
    packet["orders_submitted"] = bool(evidence.get("broker_orders_submitted"))
    packet["packet_status"] = SAME_CYCLE_DISPATCH_REASON
    packet["reason"] = SAME_CYCLE_DISPATCH_REASON
    packet["arm_eligibility"] = "UNAVAILABLE"
    if write:
        append_same_cycle_suppression_record(
            results_root,
            invocation_id=inv or "unbound",
            authority_id=evidence.get("authority_id"),
            dispatch_id=evidence.get("dispatch_id"),
            classification=STATUS_UNAVAILABLE_SAME_CYCLE,
            reason=SAME_CYCLE_DISPATCH_REASON,
            now=now,
            extra={
                "placed_count": evidence.get("placed_count"),
                "dispatch_performed": evidence.get("dispatch_performed"),
            },
        )
        write_canonical_packet(dest, packet, overwrite=True)
        write_entry_interlock(
            results_root=results_root,
            invocation_id=inv or "unbound",
            now=now,
            pending_sell_symbols=[],
            pending_sell_actions=[],
            affected_sectors=[],
            source_plan_sha256=binding.get("source_plan_sha256"),
            selected_row_hash_value=binding.get("selected_row_hash"),
            packet_status=SAME_CYCLE_DISPATCH_REASON,
            decision=DECISION_BLOCK,
            reason=SAME_CYCLE_DISPATCH_REASON,
            interlock_scope=SCOPE_GLOBAL_SAFETY_BLOCK,
            global_block_reason=SAME_CYCLE_DISPATCH_REASON,
            risk_criticality="UNKNOWN",
        )
    return {
        "ok": True,
        "reason": SAME_CYCLE_DISPATCH_REASON,
        "wrote": bool(write),
        "packet": packet,
        "path": str(dest),
        "previous_preserved": False,
        "broker": 0,
        "network": 0,
        "exit_code": F2_OTHER_GATE,
        "packet_status": SAME_CYCLE_DISPATCH_REASON,
        "classifications": [],
        "invocation_id": inv,
        "interlock_decision": DECISION_BLOCK,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        return list(csv.DictReader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    except Exception:
        return []


def _f(v: Any) -> Optional[float]:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return None
        return float(v)
    except Exception:
        return None


def plan_fingerprint(rows: Sequence[Dict[str, Any]]) -> str:
    blob = "|".join(
        sorted(
            f"{r.get('symbol')}:{r.get('final_action')}:{r.get('qty')}:"
            f"{r.get('selected_for_execution')}:{r.get('reason_code')}:{r.get('skip_reason')}"
            for r in rows
        )
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_recognized_entry_only_action(action: Any) -> bool:
    return str(action or "").upper().strip() in RECOGNIZED_ENTRY_ONLY_ACTIONS


def _contradictory_buy_side(row: Dict[str, Any]) -> bool:
    side = str(row.get("order_side") or row.get("side") or "").upper().strip()
    if not side:
        return False
    if side in CONTRADICTORY_BUY_SIDES or side in BUY_ACTIONS:
        return True
    return False


def _partition_invalid_classifications(
    classified: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split CLASS_INVALID into recognized entry-only vs fail-closed rows."""
    excluded: List[Dict[str, Any]] = []
    fatal: List[Dict[str, Any]] = []
    for rec in classified:
        if rec.get("classification") != CLASS_INVALID:
            continue
        if _is_recognized_entry_only_action(rec.get("action")):
            excluded.append(rec)
        else:
            fatal.append(rec)
    return excluded, fatal


def _normalize_action_row(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(row, dict):
        return None, "action_row_not_object"
    symbol = str(row.get("symbol") or "").upper().strip()
    action = str(row.get("action") or row.get("final_action") or "").upper().strip()
    qty = _f(row.get("qty"))
    if not symbol:
        return None, "action_missing_symbol"
    if action in BUY_ACTIONS:
        return None, "buy_or_add_in_sell_packet"
    if action not in SELL_ACTIONS:
        return None, "action_not_sell"
    if qty is None or qty <= 0:
        return None, "action_qty_invalid"
    price = _f(row.get("est_price") or row.get("current_price"))
    notional = _f(row.get("est_notional"))
    if notional is None and price is not None:
        notional = round(qty * price, 2)
    return {
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "est_price": price,
        "est_notional": notional,
    }, None


def build_canonical_packet(
    *,
    now: datetime,
    actions: Sequence[Dict[str, Any]],
    mode: str = CANONICAL_MODE,
    cycle_id: str,
    run_id: str,
    batch_id: str,
    plan_id: str,
    plan_hash: str,
    generation_status: str = "ok",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generated_at = _iso_utc(now)
    session_close = None
    session_date = None
    try:
        from services.market_calendar import regular_session_state

        _st = regular_session_state(now=now)
        session_date = _st.get("session_date")
        close_raw = _st.get("session_close_et")
        if close_raw:
            session_close = datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))
    except Exception:
        session_close = None
        session_date = None
    valid_until = _iso_utc(readiness_packet_valid_until(now, session_close=session_close))
    normalized: List[Dict[str, Any]] = []
    for row in actions:
        item, err = _normalize_action_row(row)
        if err:
            raise ValueError(f"packet_action_rejected:{err}")
        assert item is not None
        normalized.append(item)
    symbols = [r["symbol"] for r in normalized]
    total = round(sum(float(r.get("est_notional") or 0.0) for r in normalized), 2)
    packet: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": READINESS_STATUS_OK,
        "generation_status": generation_status,
        "producer": PRODUCER_ID,
        "generated_at": generated_at,
        "valid_until": valid_until,
        "freshness_ttl_minutes": READINESS_REFERENCE_TTL_MINUTES,
        "mode": str(mode or "").upper(),
        "live_execution_state": LIVE_EXECUTION_BLOCKED,
        "creates_no_authority": True,
        "orders_submitted": False,
        "approval_created": False,
        "cycle_id": cycle_id,
        "run_id": run_id,
        "batch_id": batch_id,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "approval_ready_symbols": symbols,
        "approval_ready_actions": normalized,
        "total_proposed_notional": total,
        "packet_generation_timestamp": generated_at,
        "session_date": session_date,
    }
    if extra:
        for key, value in extra.items():
            if key in packet:
                continue
            packet[key] = value
    return packet


def write_canonical_packet(
    path: Path, packet: Dict[str, Any], *, overwrite: bool = True
) -> Dict[str, Any]:
    if not overwrite and path.is_file():
        return {
            "ok": False,
            "reason": "refused_overwrite",
            "wrote": False,
            "path": str(path),
        }
    _atomic_write_json(path, packet, require_full_durability=False)
    return {"ok": True, "reason": "ok", "wrote": True, "path": str(path)}


def _selected_sell_rows(
    plan_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    out: List[Dict[str, Any]] = []
    for row in plan_rows:
        selected = str(row.get("selected_for_execution") or "").lower() == "true"
        if not selected:
            continue
        action = str(
            row.get("final_action") or row.get("management_action_candidate") or ""
        ).upper()
        if _is_recognized_entry_only_action(action):
            continue
        mapped = {
            "symbol": row.get("symbol"),
            "action": action,
            "qty": row.get("qty"),
            "est_price": row.get("current_price") or row.get("limit_price"),
            "est_notional": row.get("planned_notional"),
        }
        item, err = _normalize_action_row(mapped)
        if err:
            return [], err
        assert item is not None
        if (
            item.get("est_notional") is not None
            and float(item["est_notional"]) > APPROVAL_CLI_CAP + 1e-9
        ):
            continue
        out.append(item)
    return out, None


def _validate_cycle_binding(
    *,
    results_root: Path,
    plan: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    now: datetime,
    expected_invocation_id: Optional[str],
) -> Dict[str, Any]:
    binding = load_cycle_binding(results_root)
    if not binding:
        return {
            "ok": False,
            "reason": "f_stage_binding_missing",
            "exit_code": F2_SOURCE_UNBOUND,
        }
    env_id = str(os.environ.get(INVOCATION_ENV) or "").strip()
    expected = str(expected_invocation_id or env_id or "").strip()
    inv = str(binding.get("invocation_id") or "")
    if expected and inv != expected:
        return {
            "ok": False,
            "reason": "cycle_id_mismatch",
            "exit_code": F2_SOURCE_UNBOUND,
        }
    plan_ts = _parse_iso(plan.get("timestamp") or plan.get("plan_generated_at"))
    bind_ts = _parse_iso(binding.get("plan_generated_at"))
    if plan_ts is None or bind_ts is None:
        return {
            "ok": False,
            "reason": "plan_generated_at_unbound",
            "exit_code": F2_SOURCE_UNBOUND,
        }
    if plan_ts != bind_ts:
        return {
            "ok": False,
            "reason": "plan_generated_at_mismatch",
            "exit_code": F2_SOURCE_UNBOUND,
        }
    if bind_ts > now:
        return {
            "ok": False,
            "reason": "future_plan_generated_at",
            "exit_code": F2_SOURCE_UNBOUND,
        }
    csv_hash = sha256_file_bytes(results_root / "manage_positions_plan.csv")
    if csv_hash and csv_hash != binding.get("source_plan_sha256"):
        return {
            "ok": False,
            "reason": "source_plan_hash_mismatch",
            "exit_code": F2_SOURCE_UNBOUND,
        }
    row_hash = selected_row_hash(rows)
    if row_hash != binding.get("selected_row_hash"):
        return {
            "ok": False,
            "reason": "selected_row_hash_mismatch",
            "exit_code": F2_SOURCE_UNBOUND,
        }
    if str(binding.get("mode") or "").upper() != "PAPER":
        return {
            "ok": False,
            "reason": "binding_mode_not_paper",
            "exit_code": F2_GENERATION_FAILURE,
        }
    if binding.get("live_execution_blocked") is not True:
        return {
            "ok": False,
            "reason": "live_execution_not_blocked",
            "exit_code": F2_GENERATION_FAILURE,
        }
    if str(binding.get("f_stage_status") or "") != "COMPLETE":
        return {
            "ok": False,
            "reason": "f_stage_not_complete",
            "exit_code": F2_GENERATION_FAILURE,
        }
    completed = _parse_iso(binding.get("f_stage_completed_at"))
    if completed is None or completed > now:
        return {
            "ok": False,
            "reason": "f_stage_completed_at_invalid",
            "exit_code": F2_SOURCE_UNBOUND,
        }
    return {"ok": True, "reason": "ok", "binding": binding, "exit_code": F2_OK}


def generate_from_manage_artifacts(
    *,
    results_root: Path,
    now: Optional[datetime] = None,
    write: bool = True,
    expected_invocation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a current-cycle canonical packet. Failed generation does not restamp."""
    now = now or _utc_now()
    results_root = Path(results_root)
    dest = packet_path(results_root)
    existed = dest.is_file()
    try:
        plan = _read_json(results_root / "manage_positions_plan.json")
        rows = _read_csv(results_root / "manage_positions_plan.csv")
        if not plan:
            raise ValueError("manage_positions_plan_missing")
        checked = _validate_cycle_binding(
            results_root=results_root,
            plan=plan,
            rows=rows,
            now=now,
            expected_invocation_id=expected_invocation_id,
        )
        if not checked.get("ok"):
            return {
                "ok": False,
                "reason": checked.get("reason"),
                "wrote": False,
                "packet": None,
                "path": str(dest),
                "previous_preserved": existed,
                "broker": 0,
                "network": 0,
                "exit_code": checked.get("exit_code") or F2_SOURCE_UNBOUND,
                "packet_status": "SOURCE_UNBOUND",
                "classifications": [],
            }
        binding = checked["binding"]
        inv_bind = str(binding.get("invocation_id") or "")
        dispatch_kind, dispatch_ev = classify_same_cycle_dispatch(
            invocation_id=inv_bind, results_root=results_root
        )
        if dispatch_kind in ("suppress", "ambiguous"):
            rec = write_same_cycle_dispatch_reconciliation_packet(
                results_root=results_root,
                binding=binding,
                evidence=dispatch_ev,
                now=now,
                write=write,
            )
            rec["dispatch_classification"] = dispatch_kind
            return rec
        classified = classify_plan_rows(rows, binding)
        if any(c.get("classification") == CLASS_OUTCOME_UNKNOWN for c in classified):
            return {
                "ok": False,
                "reason": "sell_outcome_unknown",
                "wrote": False,
                "packet": None,
                "path": str(dest),
                "previous_preserved": existed,
                "broker": 0,
                "network": 0,
                "exit_code": F2_OUTCOME_UNKNOWN,
                "packet_status": "OUTCOME_UNKNOWN",
                "classifications": classified,
                "invocation_id": binding.get("invocation_id"),
            }
        ready = [c for c in classified if c.get("classification") == CLASS_AUTHORITY_READY]
        other_gate = [c for c in classified if c.get("classification") == CLASS_BLOCKED_OTHER_GATE]
        excluded_entry_only, fatal_invalid = _partition_invalid_classifications(classified)
        if fatal_invalid:
            raise ValueError("invalid_selected_sell_row")
        actions = []
        for rec in ready:
            row = next(
                (r for r in rows if str(r.get("symbol") or "").upper() == rec["symbol"]),
                {},
            )
            if _contradictory_buy_side(row):
                raise ValueError("contradictory_side_action")
            item, err = _normalize_action_row(
                {
                    "symbol": rec["symbol"],
                    "action": rec["action"],
                    "qty": rec.get("qty") or row.get("qty"),
                    "est_price": row.get("current_price") or row.get("limit_price"),
                    "est_notional": row.get("planned_notional"),
                }
            )
            if err:
                raise ValueError(err)
            actions.append(item)
        inv = str(binding.get("invocation_id"))
        # plan_hash / selected_row_hash remain the whole-plan F-binding identity.
        # Exclusion of recognized BUY/ADD rows does not recompute that hash.
        packet = build_canonical_packet(
            now=now,
            actions=actions,
            cycle_id=inv,
            run_id=str(binding.get("plan_generated_at")),
            batch_id=f"manage-{inv[:16]}",
            plan_id=f"manage_positions_plan@{binding.get('plan_generated_at')}",
            plan_hash=str(binding.get("selected_row_hash")),
            generation_status="ok",
            extra={
                "invocation_id": inv,
                "market_open_invocation_id": inv,
                "plan_generated_at": binding.get("plan_generated_at"),
                "f_stage_completed_at": binding.get("f_stage_completed_at"),
                "f_stage_status": binding.get("f_stage_status"),
                "source_plan_sha256": binding.get("source_plan_sha256"),
                "selected_row_hash": binding.get("selected_row_hash"),
                "positions_snapshot_sha256": binding.get("positions_snapshot_sha256"),
                "open_orders_snapshot_sha256": binding.get("open_orders_snapshot_sha256"),
                "row_classifications": classified,
                "excluded_entry_only_count": len(excluded_entry_only),
                "excluded_entry_only_symbols": [
                    str(c.get("symbol") or "") for c in excluded_entry_only
                ],
                "excluded_entry_only_reason": (
                    "recognized_non_sell_row" if excluded_entry_only else None
                ),
                "arm_eligibility": (CLASS_AUTHORITY_READY if actions else "EMPTY"),
                "remaining_blocker": binding.get("remaining_blocker"),
                "packet_status": (
                    CLASS_AUTHORITY_READY
                    if actions
                    else ("BLOCKED_OTHER_GATE" if other_gate else "VALID_EMPTY")
                ),
            },
        )
        if write:
            write_canonical_packet(dest, packet)
        pending_actions = [
            {
                "symbol": c.get("symbol"),
                "action": c.get("action"),
                "qty": c.get("qty"),
                "reason_code": c.get("reason_code"),
                "rotation_id": c.get("rotation_id"),
                "risk_objective": c.get("risk_objective"),
                "risk_criticality": ("CRITICAL" if is_critical_pending_sell(c) else "ORDINARY"),
            }
            for c in ready
        ]
        affected = sorted(
            {
                pending_sell_sector(str(a.get("symbol") or ""))
                for a in pending_actions
                if a.get("symbol")
            }
        )
        critical = any(is_critical_pending_sell(c) for c in ready)
        bind_kwargs = {
            "results_root": results_root,
            "invocation_id": inv,
            "now": now,
            "pending_sell_symbols": [str(a["symbol"]) for a in pending_actions],
            "pending_sell_actions": pending_actions,
            "affected_sectors": affected,
            "source_plan_sha256": binding.get("source_plan_sha256"),
            "selected_row_hash_value": binding.get("selected_row_hash"),
        }
        if actions and critical:
            status = "AUTHORITY_READY"
            exit_code = F2_PENDING_SELL_ARM
            decision = DECISION_BLOCK
            reason = REASON_CRITICAL_EXIT
            scope = SCOPE_GLOBAL_SAFETY_BLOCK
            risk_crit = "CRITICAL"
            global_reason = REASON_CRITICAL_EXIT
        elif actions:
            status = "AUTHORITY_READY"
            exit_code = F2_PENDING_SELL_ARM
            decision = DECISION_CANDIDATE_AWARE
            reason = REASON_PENDING_SELL_ARM
            scope = SCOPE_CANDIDATE_FILTER
            risk_crit = "ORDINARY"
            global_reason = None
        elif other_gate:
            status = "BLOCKED_OTHER_GATE"
            exit_code = F2_OTHER_GATE
            decision = DECISION_BLOCK
            reason = REASON_SELL_PRECONDITION
            scope = SCOPE_GLOBAL_SAFETY_BLOCK
            risk_crit = "UNKNOWN"
            global_reason = REASON_SELL_PRECONDITION
        else:
            status = "VALID_EMPTY"
            exit_code = F2_OK
            decision = DECISION_MAY_CONTINUE
            reason = "ok"
            scope = SCOPE_NONE
            risk_crit = "NONE"
            global_reason = None
        if write:
            write_entry_interlock(
                packet_status=status,
                decision=decision,
                reason=reason,
                interlock_scope=scope,
                global_block_reason=global_reason,
                risk_criticality=risk_crit,
                **bind_kwargs,
            )
        return {
            "ok": True,
            "reason": "ok",
            "wrote": bool(write),
            "packet": packet,
            "path": str(dest),
            "previous_preserved": False,
            "broker": 0,
            "network": 0,
            "exit_code": exit_code,
            "packet_status": status,
            "classifications": classified,
            "invocation_id": inv,
            "interlock_decision": decision,
        }
    except Exception as exc:
        return {
            "ok": False,
            "reason": str(exc),
            "wrote": False,
            "packet": None,
            "path": str(dest),
            "previous_preserved": existed,
            "broker": 0,
            "network": 0,
            "exit_code": F2_GENERATION_FAILURE,
            "packet_status": "GENERATION_FAILURE",
            "classifications": [],
        }


def run_market_open_readiness_stage(
    *,
    results_root: Optional[Path] = None,
    now: Optional[datetime] = None,
    expected_invocation_id: Optional[str] = None,
) -> int:
    """F2 hook. Writes interlock. Never dispatches or arms."""
    results_root = results_root or default_results_root()
    now = now or _utc_now()
    result = generate_from_manage_artifacts(
        results_root=results_root,
        now=now,
        expected_invocation_id=expected_invocation_id,
    )
    if not result.get("ok"):
        write_entry_interlock(
            results_root=results_root,
            invocation_id=str(
                result.get("invocation_id") or os.environ.get(INVOCATION_ENV) or "unbound"
            ),
            now=now,
            decision=DECISION_BLOCK,
            reason=(
                REASON_SELL_OUTCOME_UNKNOWN
                if result.get("exit_code") == F2_OUTCOME_UNKNOWN
                else (
                    REASON_SELL_READINESS_UNKNOWN
                    if result.get("exit_code") == F2_SOURCE_UNBOUND
                    else REASON_SELL_PRECONDITION
                )
            ),
            packet_status=str(result.get("packet_status") or "GENERATION_FAILURE"),
            interlock_scope=SCOPE_GLOBAL_SAFETY_BLOCK,
            global_block_reason=(
                REASON_SELL_OUTCOME_UNKNOWN
                if result.get("exit_code") == F2_OUTCOME_UNKNOWN
                else REASON_SELL_READINESS_UNKNOWN
            ),
            risk_criticality="UNKNOWN",
        )
    printable = {
        k: result.get(k)
        for k in (
            "ok",
            "reason",
            "wrote",
            "previous_preserved",
            "broker",
            "network",
            "exit_code",
            "packet_status",
        )
    }
    print(json.dumps(printable, default=str))
    return int(
        result.get("exit_code") if result.get("exit_code") is not None else F2_GENERATION_FAILURE
    )


def main() -> int:
    return run_market_open_readiness_stage()


if __name__ == "__main__":
    raise SystemExit(main())
