"""
defensive_exit_governance_sync.py
Phase 197I — Sync capped defensive exit preview with governance/readiness/operator approval.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
READINESS_JSON = RESULTS / "execution_readiness.json"
GOVERNANCE_AUTH_JSON = RESULTS / "governance_authorization.json"
HUMAN_QUEUE_JSON = RESULTS / "human_approval_queue.json"
OPEN_ORDERS_JSON = RESULTS / "open_orders_snapshot.csv"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"

DEFENSIVE_APPROVAL_ACTIONS = frozenset(
    {
        "REDUCE_CONCENTRATION",
        "REVIEW_DRAWDOWN_POSITIONS",
        "DEFENSIVE_EXIT",
        "POSITION_EXIT",
        "FORCE_EXIT",
    }
)
APPROVED_QUEUE_STATUSES = frozenset({"APPROVED", "EXECUTION_PERMITTED", "AUTHORIZED"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _load_preview() -> Dict[str, Any]:
    return _read_json(PREVIEW_JSON)


def _runtime_capped_preview(doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        doc.get("capped_preview_runtime_cap")
        or doc.get("capped_preview")
        or doc.get("capped_preview_cli_cap")
        or {}
    )


def _open_sell_symbols() -> Set[str]:
    if not OPEN_ORDERS_JSON.is_file():
        return set()
    try:
        import pandas as pd

        df = pd.read_csv(OPEN_ORDERS_JSON, low_memory=False)
        sym_col = (
            "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
        )
        side_col = "side" if "side" in df.columns else None
        if not sym_col:
            return set()
        if side_col:
            df = df[df[side_col].astype(str).str.lower() == "sell"]
        return {
            str(x).strip().upper()
            for x in df[sym_col].dropna().astype(str)
            if str(x).strip() and str(x).strip().upper() not in ("NAN", "NONE")
        }
    except Exception:
        return set()


def _operator_approval_for_symbols(symbols: List[str]) -> Tuple[bool, List[str], Dict[str, Any]]:
    queue = _read_json(HUMAN_QUEUE_JSON)
    requests = list(queue.get("requests") or [])
    sym_set = {s.upper() for s in symbols if s}
    approved: List[str] = []
    pending: List[str] = []
    details: List[Dict[str, Any]] = []
    for req in requests:
        action = str(req.get("candidate_action") or "").strip().upper()
        subject = str(req.get("subject") or "").strip().upper()
        status = str(req.get("status") or "").strip().upper()
        if subject not in sym_set:
            continue
        if action not in DEFENSIVE_APPROVAL_ACTIONS:
            continue
        entry = {
            "request_id": req.get("request_id"),
            "subject": subject,
            "action": action,
            "status": status,
            "execution_permitted": bool(req.get("execution_permitted")),
        }
        details.append(entry)
        if status in APPROVED_QUEUE_STATUSES or bool(req.get("execution_permitted")):
            approved.append(subject)
        else:
            pending.append(subject)
    covered = sym_set and sym_set.issubset(set(approved))
    return covered, pending, {"matches": details, "approved_subjects": sorted(set(approved))}


def build_defensive_exit_approval_sync(*, cap_context: str = "runtime") -> Dict[str, Any]:
    """
    Classify capped defensive batch against governance/readiness/operator gates.
    Read-only — never approves or submits orders.
    """
    preview_doc = _load_preview()
    capped_block = _runtime_capped_preview(preview_doc)
    if cap_context == "cli":
        capped_block = dict(
            preview_doc.get("capped_preview_cli_cap") or preview_doc.get("capped_preview") or {}
        )

    capped_symbols = list(capped_block.get("capped_preview_symbols") or [])
    deferred_symbols = list(capped_block.get("deferred_symbols") or [])
    capped_notional = float(capped_block.get("batch_notional_capped_preview") or 0.0)
    configured_cap = float(
        preview_doc.get("configured_batch_cap_manage_positions")
        or preview_doc.get("configured_batch_cap_effective")
        or preview_doc.get("configured_batch_cap_cli_default")
        or 0.0
    )
    partial_authorized = bool(preview_doc.get("partial_batch_authorized"))

    readiness = _read_json(READINESS_JSON)
    auth = _read_json(GOVERNANCE_AUTH_JSON)
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    governance_authorized = bool(auth.get("governance_authorized"))
    operator_authorized_global = bool(auth.get("operator_authorized"))
    execution_authorized_policy = bool(auth.get("execution_authorized"))
    paper_permitted = bool(readiness.get("paper_execution_permitted", False))
    live_permitted = bool(readiness.get("live_execution_permitted", True))
    failed_checks = list(readiness.get("failed_checks") or [])

    open_sells = _open_sell_symbols()
    capped_set = {s.upper() for s in capped_symbols}
    in_flight_overlap = sorted(capped_set & open_sells)

    op_ok, op_pending, op_details = _operator_approval_for_symbols(capped_symbols)

    approval_sync_status = "CAPPED_DEFENSIVE_BATCH_AVAILABLE"
    approval_block_reason = ""
    approval_source = "none"

    preview_exists = bool(preview_doc) and bool(capped_symbols or deferred_symbols)

    if not preview_exists:
        approval_sync_status = "CAPPED_DEFENSIVE_BATCH_AVAILABLE"
        approval_block_reason = "no_capped_preview_artifact"
    elif not partial_authorized:
        approval_sync_status = "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_GOVERNANCE"
        approval_block_reason = (
            preview_doc.get("partial_batch_status") or "partial_defensive_batch_not_authorized"
        )
    elif not capped_symbols:
        approval_sync_status = "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_NOTIONAL_CAP"
        approval_block_reason = (
            f"no_symbols_fit_cap cap={configured_cap} full_batch="
            f"{preview_doc.get('batch_notional_estimate')}"
        )
    elif in_flight_overlap:
        approval_sync_status = "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_IN_FLIGHT_ORDERS"
        approval_block_reason = f"in_flight_sell_orders symbols={','.join(in_flight_overlap)}"
    elif not governance_authorized or "governance" in failed_checks:
        approval_sync_status = "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_GOVERNANCE"
        approval_block_reason = str(
            auth.get("gate_reasons", {}).get("governance")
            or readiness.get("check_details", {}).get("governance")
            or "governance_authorized=false"
        )
        approval_source = "governance_authorization.json"
    elif readiness_status != "READY":
        approval_sync_status = "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_READINESS"
        approval_block_reason = f"execution_readiness={readiness_status}"
        if failed_checks:
            approval_block_reason += f" failed_checks={','.join(failed_checks)}"
        approval_source = "execution_readiness.json"
    elif op_pending and not op_ok:
        approval_sync_status = "CAPPED_DEFENSIVE_BATCH_AWAITING_OPERATOR_APPROVAL"
        approval_block_reason = (
            f"pending_operator_review subjects={','.join(sorted(set(op_pending)))}"
        )
        approval_source = "human_approval_queue.json"
    else:
        approval_sync_status = "CAPPED_DEFENSIVE_BATCH_APPROVED"
        approval_block_reason = ""
        approval_source = "governance_authorization.json+execution_readiness.json"

    eligible_for_paper_execution = (
        approval_sync_status == "CAPPED_DEFENSIVE_BATCH_APPROVED"
        and paper_permitted
        and not live_permitted
        and bool(capped_symbols)
    )

    return {
        "timestamp": _utc_now_iso(),
        "phase": "197I",
        "preview_exists": preview_exists,
        "partial_batch_authorized": partial_authorized,
        "capped_preview_symbols": capped_symbols,
        "deferred_symbols": deferred_symbols,
        "capped_preview_notional": round(capped_notional, 2),
        "configured_cap": configured_cap,
        "cap_context": cap_context,
        "batch_notional_full": preview_doc.get("batch_notional_estimate"),
        "cap_excess_amount": preview_doc.get("cap_excess_amount"),
        "readiness_status": readiness_status,
        "readiness_failed_checks": failed_checks,
        "governance_authorized": governance_authorized,
        "operator_authorized_global": operator_authorized_global,
        "operator_authorized_for_capped_batch": op_ok,
        "operator_approval_details": op_details,
        "execution_authorized_policy": execution_authorized_policy,
        "paper_execution_permitted": paper_permitted,
        "live_execution_permitted": live_permitted,
        "approval_sync_status": approval_sync_status,
        "approval_block_reason": approval_block_reason,
        "approval_source": approval_source,
        "eligible_for_paper_execution": eligible_for_paper_execution,
        "execution_eligibility_note": (
            "eligible_for_paper_execution requires READY readiness, governance authorized, "
            "operator approval for capped symbols, paper permitted, live blocked"
        ),
        "in_flight_sell_overlap": in_flight_overlap,
        "open_orders_count_assumed_from_snapshot": OPEN_ORDERS_JSON.is_file(),
        "preview_path": str(PREVIEW_JSON.relative_to(ROOT)).replace("\\", "/"),
        "handoff_status_197h": preview_doc.get("handoff_status"),
    }


def build_operator_approval_packet(sync: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    sync = sync or build_defensive_exit_approval_sync()
    capped = list(sync.get("capped_preview_symbols") or [])
    deferred = list(sync.get("deferred_symbols") or [])
    queue = _read_json(HUMAN_QUEUE_JSON)
    related_requests = []
    capped_set = {s.upper() for s in capped}
    for req in queue.get("requests") or []:
        subject = str(req.get("subject") or "").strip().upper()
        if subject in capped_set or subject in {d.upper() for d in deferred}:
            related_requests.append(
                {
                    "request_id": req.get("request_id"),
                    "subject": subject,
                    "candidate_action": req.get("candidate_action"),
                    "status": req.get("status"),
                    "execution_permitted": req.get("execution_permitted"),
                    "reason": req.get("reason"),
                }
            )
    packet = {
        "generated_at": _utc_now_iso(),
        "phase": "197I",
        "disclaimer": ("Read-only operator approval packet. Does not approve or submit orders."),
        "approval_sync_status": sync.get("approval_sync_status"),
        "approval_block_reason": sync.get("approval_block_reason"),
        "capped_preview_symbols": capped,
        "deferred_symbols": deferred,
        "capped_preview_notional": sync.get("capped_preview_notional"),
        "configured_cap": sync.get("configured_cap"),
        "would_require": [
            "governance_authorized=true",
            "execution_readiness=READY",
            "operator approval for capped defensive exit symbols",
            "paper_execution_permitted=true",
            "live_execution_permitted=false",
            "no in-flight duplicate sell orders",
            "batch within configured notional cap",
        ],
        "current_blockers": (
            [sync.get("approval_block_reason")] if sync.get("approval_block_reason") else []
        ),
        "related_human_approval_requests": related_requests,
        "operator_action_required": sync.get("approval_sync_status")
        == "CAPPED_DEFENSIVE_BATCH_AWAITING_OPERATOR_APPROVAL",
    }
    return packet


def write_operator_approval_packet(packet: Dict[str, Any]) -> Path:
    OPERATOR_PACKET_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OPERATOR_PACKET_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(packet, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, OPERATOR_PACKET_JSON)
    return OPERATOR_PACKET_JSON
