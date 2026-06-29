"""
defensive_exit_operator_paper_approval.py
Phase 197J — Formal operator/governance review packet and paper-only approval contract.
Read-only validation; never auto-approves or submits broker orders.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
READINESS_JSON = RESULTS / "execution_readiness.json"
GOVERNANCE_AUTH_JSON = RESULTS / "governance_authorization.json"
OPEN_ORDERS_JSON = RESULTS / "open_orders_snapshot.csv"
POSITIONS_JSON = RESULTS / "positions_snapshot.csv"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
APPROVAL_CONTRACT_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"

PACKET_TTL_HOURS = 24

OPERATOR_DECISION_APPROVE = "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH"
OPERATOR_DECISION_REJECT = "REJECT_CAPPED_PAPER_DEFENSIVE_BATCH"
OPERATOR_DECISION_DEFER = "DEFER_REVIEW"
OPERATOR_DECISION_REVISE = "REQUEST_REVISED_BATCH"

OPERATOR_DECISION_OPTIONS = [
    OPERATOR_DECISION_APPROVE,
    OPERATOR_DECISION_REJECT,
    OPERATOR_DECISION_DEFER,
    OPERATOR_DECISION_REVISE,
]

APPROVAL_STATUS_REQUIRED = "OPERATOR_APPROVAL_REQUIRED"
APPROVAL_STATUS_INVALID = "OPERATOR_APPROVAL_INVALID"
APPROVAL_STATUS_VALID = "OPERATOR_APPROVAL_VALID_PAPER_ONLY"

REQUIRED_CONTRACT_FIELDS = frozenset(
    {
        "approval_id",
        "approved_at",
        "expires_at",
        "approved_by",
        "approval_mode",
        "approved_symbols",
        "max_notional",
        "source_packet_id",
        "source_preview_hash",
        "acknowledgement_live_blocked",
        "acknowledgement_no_live_execution",
        "acknowledgement_paper_only",
        "decision",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _runtime_capped_preview(doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        doc.get("capped_preview_runtime_cap")
        or doc.get("capped_preview")
        or doc.get("capped_preview_cli_cap")
        or {}
    )


def compute_capped_preview_hash(
    preview_doc: Optional[Dict[str, Any]] = None,
    *,
    capped_symbols: Optional[List[str]] = None,
    deferred_symbols: Optional[List[str]] = None,
    capped_notional: Optional[float] = None,
    configured_cap: Optional[float] = None,
) -> str:
    preview_doc = preview_doc if preview_doc is not None else _read_json(PREVIEW_JSON)
    block = _runtime_capped_preview(preview_doc)
    symbols = (
        capped_symbols
        if capped_symbols is not None
        else list(block.get("capped_preview_symbols") or [])
    )
    deferred = (
        deferred_symbols
        if deferred_symbols is not None
        else list(block.get("deferred_symbols") or [])
    )
    notional = capped_notional
    if notional is None:
        notional = float(block.get("batch_notional_capped_preview") or 0.0)
    cap = configured_cap
    if cap is None:
        cap = float(
            preview_doc.get("configured_batch_cap_manage_positions")
            or preview_doc.get("configured_batch_cap_effective")
            or preview_doc.get("configured_batch_cap_cli_default")
            or 0.0
        )
    canonical = {
        "capped_preview_symbols": sorted({s.upper() for s in symbols if s}),
        "deferred_symbols": sorted({s.upper() for s in deferred if s}),
        "batch_notional_capped_preview": round(float(notional), 2),
        "configured_cap": round(float(cap), 2),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _risk_reduction_intent(
    preview_doc: Dict[str, Any], capped_symbols: List[str]
) -> Dict[str, Any]:
    handoff_status = str(preview_doc.get("handoff_status") or "")
    partial_status = str(preview_doc.get("partial_batch_status") or "")
    return {
        "summary": (
            "Reduce portfolio risk via governed partial defensive exits under configured "
            "batch notional cap; highest-priority symbols first; deferred symbols remain "
            "until cap or separate approval permits."
        ),
        "handoff_status": handoff_status,
        "partial_batch_status": partial_status,
        "planned_exit_count": preview_doc.get("planned_exit_count"),
        "capped_symbol_count": len(capped_symbols),
        "objectives": [
            "reduce_concentration_and_drawdown_exposure",
            "respect_batch_notional_cap",
            "paper_only_no_live_execution",
            "preserve_deferred_symbols_until_explicitly_approved",
        ],
    }


def build_formal_operator_approval_packet(
    *,
    sync: Optional[Dict[str, Any]] = None,
    broker_snapshot_ok: bool = True,
    broker_orders_submitted: bool = False,
) -> Dict[str, Any]:
    from services.defensive_exit_governance_sync import build_defensive_exit_approval_sync

    sync = sync or build_defensive_exit_approval_sync(cap_context="runtime")
    preview_doc = _read_json(PREVIEW_JSON)
    readiness = _read_json(READINESS_JSON)
    auth = _read_json(GOVERNANCE_AUTH_JSON)
    from services.live_block_classifier import evaluate_live_block_status

    live_block = evaluate_live_block_status(paper_mode=True)

    capped = list(sync.get("capped_preview_symbols") or [])
    deferred = list(sync.get("deferred_symbols") or [])
    capped_notional = float(sync.get("capped_preview_notional") or 0.0)
    configured_cap = float(sync.get("configured_cap") or 0.0)
    preview_hash = compute_capped_preview_hash(
        preview_doc,
        capped_symbols=capped,
        deferred_symbols=deferred,
        capped_notional=capped_notional,
        configured_cap=configured_cap,
    )
    generated_at = _utc_now()
    expires_at = generated_at + timedelta(hours=PACKET_TTL_HOURS)
    packet_id = f"def-exit-paper-{preview_hash[:12]}-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"

    contract_exists = APPROVAL_CONTRACT_PATH.is_file()
    contract_doc = _read_json(APPROVAL_CONTRACT_PATH) if contract_exists else {}
    approval_status, approval_reason, _ = validate_operator_approval_contract(
        packet_id=packet_id,
        preview_hash=preview_hash,
        capped_symbols=capped,
        configured_cap=configured_cap,
        broker_snapshot_ok=broker_snapshot_ok,
    )

    return {
        "packet_id": packet_id,
        "generated_at": generated_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "phase": "197J",
        "disclaimer": (
            "Read-only operator approval packet. Does not approve or submit orders. "
            "Operator must intentionally create data/governance/operator_paper_defensive_exit_approval.json."
        ),
        "paper_only": True,
        "live_blocked": bool(live_block.get("live_blocked")),
        "live_block_reason": live_block.get("live_block_reason"),
        "current_governance_status": {
            "governance_authorized": bool(sync.get("governance_authorized")),
            "approval_sync_status": sync.get("approval_sync_status"),
            "approval_block_reason": sync.get("approval_block_reason"),
        },
        "current_readiness_status": {
            "readiness_status": sync.get("readiness_status"),
            "failed_checks": sync.get("readiness_failed_checks") or [],
            "paper_execution_permitted": bool(sync.get("paper_execution_permitted")),
            "live_execution_permitted": bool(sync.get("live_execution_permitted")),
        },
        "approval_required": True,
        "approval_status": approval_status,
        "approval_block_reason": (
            approval_reason if approval_status != APPROVAL_STATUS_VALID else ""
        ),
        "approval_contract_path": str(APPROVAL_CONTRACT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "approval_contract_exists": contract_exists,
        "source_preview_hash": preview_hash,
        "capped_symbols": capped,
        "deferred_symbols": deferred,
        "capped_notional": round(capped_notional, 2),
        "configured_cap": configured_cap,
        "expected_position_count_before": _count_csv_rows(POSITIONS_JSON),
        "open_orders_count_before": _count_csv_rows(OPEN_ORDERS_JSON),
        "broker_snapshot_ok": broker_snapshot_ok,
        "broker_orders_submitted": broker_orders_submitted,
        "risk_reduction_intent": _risk_reduction_intent(preview_doc, capped),
        "operator_decision_options": [
            {
                "decision": OPERATOR_DECISION_APPROVE,
                "description": (
                    "Approve capped defensive batch for paper execution only; requires "
                    "explicit governance artifact with matching packet_id and preview_hash."
                ),
            },
            {
                "decision": OPERATOR_DECISION_REJECT,
                "description": "Reject capped defensive batch; no paper defensive exits from this packet.",
            },
            {
                "decision": OPERATOR_DECISION_DEFER,
                "description": "Defer review; no approval artifact; batch remains blocked.",
            },
            {
                "decision": OPERATOR_DECISION_REVISE,
                "description": "Request revised capped batch preview before approval consideration.",
            },
        ],
        "approval_contract_required_fields": sorted(REQUIRED_CONTRACT_FIELDS),
        "execution_prerequisites": [
            "operator_paper_defensive_exit_approval.json exists with decision=APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH",
            "approval_mode=PAPER_ONLY",
            "source_packet_id matches latest packet_id",
            "source_preview_hash matches latest capped preview hash",
            "approval not expired",
            "governance_authorized=true",
            "execution_readiness=READY",
            "live_execution_permitted=false",
            "no in-flight sell orders on capped symbols",
            "max_notional <= configured_cap",
        ],
        "current_blockers": list(
            filter(
                None,
                [
                    sync.get("approval_block_reason"),
                    approval_reason if approval_status != APPROVAL_STATUS_VALID else None,
                ],
            )
        ),
        "governance_gate_reason": str(
            auth.get("gate_reasons", {}).get("governance")
            or readiness.get("check_details", {}).get("governance")
            or ""
        ),
        "related_contract_decision": contract_doc.get("decision") if contract_exists else None,
    }


def write_operator_approval_packet(packet: Dict[str, Any]) -> Path:
    OPERATOR_PACKET_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OPERATOR_PACKET_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(packet, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, OPERATOR_PACKET_JSON)
    return OPERATOR_PACKET_JSON


def validate_operator_approval_contract(
    *,
    packet_id: Optional[str] = None,
    preview_hash: Optional[str] = None,
    capped_symbols: Optional[List[str]] = None,
    configured_cap: Optional[float] = None,
    broker_snapshot_ok: bool = True,
    live_execution_blocked: Optional[bool] = None,
    contract_path: Optional[Path] = None,
    open_sell_symbols_override: Optional[Set[str]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Validate operator paper defensive exit approval contract.
    Returns (approval_status, reason, details).

    contract_path: optional override for dry-run fixture validation (197K).
    open_sell_symbols_override: optional in-flight sell set for simulation (197K).
    """
    path = contract_path or APPROVAL_CONTRACT_PATH
    try:
        contract_path_display = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        contract_path_display = str(path)
    details: Dict[str, Any] = {
        "contract_path": contract_path_display,
        "contract_exists": path.is_file(),
        "checks": {},
    }

    if not path.is_file():
        return APPROVAL_STATUS_REQUIRED, "approval_contract_missing", details

    contract = _read_json(path)
    if bool(contract.get("template_only")):
        return APPROVAL_STATUS_INVALID, "template_only_not_runtime_authority", details

    missing = sorted(REQUIRED_CONTRACT_FIELDS - set(contract.keys()))
    details["checks"]["required_fields_present"] = not missing
    if missing:
        return APPROVAL_STATUS_INVALID, f"missing_required_fields={','.join(missing)}", details

    decision = str(contract.get("decision") or "").strip().upper()
    details["decision"] = decision

    if decision == OPERATOR_DECISION_REJECT:
        return (
            APPROVAL_STATUS_INVALID,
            "operator_rejected_batch",
            {**details, "operator_rejected": True},
        )
    if decision in (OPERATOR_DECISION_DEFER, OPERATOR_DECISION_REVISE):
        return APPROVAL_STATUS_INVALID, f"decision_not_approval={decision}", details
    if decision != OPERATOR_DECISION_APPROVE:
        return APPROVAL_STATUS_INVALID, f"unknown_decision={decision}", details

    if str(contract.get("approval_mode") or "").upper() != "PAPER_ONLY":
        return APPROVAL_STATUS_INVALID, "approval_mode_not_paper_only", details

    for ack_field in (
        "acknowledgement_live_blocked",
        "acknowledgement_no_live_execution",
        "acknowledgement_paper_only",
    ):
        if not bool(contract.get(ack_field)):
            return APPROVAL_STATUS_INVALID, f"{ack_field}_not_true", details

    expires = _parse_iso(str(contract.get("expires_at") or ""))
    if not expires or expires <= _utc_now():
        return APPROVAL_STATUS_INVALID, "approval_expired", details

    preview_doc = _read_json(PREVIEW_JSON)
    block = _runtime_capped_preview(preview_doc)
    current_capped = (
        capped_symbols
        if capped_symbols is not None
        else list(block.get("capped_preview_symbols") or [])
    )
    current_cap = configured_cap
    if current_cap is None:
        current_cap = float(
            preview_doc.get("configured_batch_cap_manage_positions")
            or preview_doc.get("configured_batch_cap_effective")
            or 0.0
        )
    current_hash = preview_hash or compute_capped_preview_hash(preview_doc)
    current_packet_id = packet_id
    if not current_packet_id and OPERATOR_PACKET_JSON.is_file():
        current_packet_id = str(_read_json(OPERATOR_PACKET_JSON).get("packet_id") or "")

    approved_symbols = [
        str(s).strip().upper() for s in (contract.get("approved_symbols") or []) if str(s).strip()
    ]
    capped_set = {s.upper() for s in current_capped if s}
    approved_set = set(approved_symbols)
    if not approved_set:
        return APPROVAL_STATUS_INVALID, "approved_symbols_empty", details
    if not approved_set.issubset(capped_set):
        extra = sorted(approved_set - capped_set)
        return (
            APPROVAL_STATUS_INVALID,
            f"approved_symbols_not_subset extra={','.join(extra)}",
            details,
        )

    max_notional = float(contract.get("max_notional") or 0.0)
    if max_notional <= 0 or max_notional > float(current_cap):
        return (
            APPROVAL_STATUS_INVALID,
            f"max_notional_exceeds_cap max={max_notional} cap={current_cap}",
            details,
        )

    contract_packet_id = str(contract.get("source_packet_id") or "")
    if not current_packet_id or contract_packet_id != current_packet_id:
        return (
            APPROVAL_STATUS_INVALID,
            f"source_packet_id_mismatch expected={current_packet_id} got={contract_packet_id}",
            details,
        )

    contract_hash = str(contract.get("source_preview_hash") or "")
    if contract_hash != current_hash:
        return (
            APPROVAL_STATUS_INVALID,
            f"source_preview_hash_mismatch expected={current_hash[:16]} got={contract_hash[:16]}",
            details,
        )

    overlap = sorted(approved_set & (open_sell_symbols_override or _open_sell_symbols()))
    if overlap:
        return (
            APPROVAL_STATUS_INVALID,
            f"in_flight_sell_orders symbols={','.join(overlap)}",
            details,
        )

    if not broker_snapshot_ok:
        return APPROVAL_STATUS_INVALID, "broker_snapshot_not_ok", details

    if live_execution_blocked is None:
        from services.live_block_classifier import evaluate_live_block_status

        live_execution_blocked = bool(
            evaluate_live_block_status(paper_mode=True).get("live_blocked")
        )

    if not live_execution_blocked:
        return APPROVAL_STATUS_INVALID, "live_execution_not_blocked", details

    details["checks"] = {
        "decision_approve": True,
        "paper_only": True,
        "acknowledgements": True,
        "not_expired": True,
        "symbols_subset": True,
        "notional_within_cap": True,
        "packet_id_match": True,
        "preview_hash_match": True,
        "no_in_flight_conflict": True,
        "broker_snapshot_ok": broker_snapshot_ok,
        "live_blocked": live_execution_blocked,
    }
    return APPROVAL_STATUS_VALID, "", details


def classify_operator_paper_approval_for_sync(
    sync: Dict[str, Any],
    *,
    broker_snapshot_ok: bool = True,
) -> Dict[str, Any]:
    """Map operator contract validation into governance sync extension fields."""
    packet = build_formal_operator_approval_packet(
        sync=sync,
        broker_snapshot_ok=broker_snapshot_ok,
    )
    approval_status = str(packet.get("approval_status") or APPROVAL_STATUS_REQUIRED)
    contract_exists = bool(packet.get("approval_contract_exists"))

    operator_rejected = False
    if contract_exists:
        decision = str(_read_json(APPROVAL_CONTRACT_PATH).get("decision") or "").upper()
        operator_rejected = decision == OPERATOR_DECISION_REJECT

    extended_status = sync.get("approval_sync_status")
    if operator_rejected:
        extended_status = "CAPPED_DEFENSIVE_BATCH_REJECTED_BY_OPERATOR"
    elif approval_status == APPROVAL_STATUS_VALID and sync.get("approval_sync_status") in (
        "CAPPED_DEFENSIVE_BATCH_APPROVED",
        None,
    ):
        extended_status = "CAPPED_DEFENSIVE_BATCH_APPROVED_FOR_PAPER"
    elif approval_status == APPROVAL_STATUS_VALID:
        extended_status = sync.get("approval_sync_status")
    elif approval_status == APPROVAL_STATUS_REQUIRED and sync.get("approval_sync_status") == (
        "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_GOVERNANCE"
    ):
        extended_status = "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_GOVERNANCE"

    return {
        "operator_paper_approval_status": approval_status,
        "operator_paper_approval_reason": packet.get("approval_block_reason") or "",
        "operator_paper_approval_packet_id": packet.get("packet_id"),
        "operator_paper_preview_hash": packet.get("source_preview_hash"),
        "approval_contract_path": packet.get("approval_contract_path"),
        "approval_contract_exists": contract_exists,
        "extended_approval_sync_status": extended_status,
        "operator_rejected": operator_rejected,
        "packet": packet,
    }


def approval_contract_schema_example() -> Dict[str, Any]:
    """Document-only example; not written to disk by TRITON."""
    return {
        "approval_id": "example-do-not-use",
        "approved_at": "2026-06-29T12:00:00+00:00",
        "expires_at": "2026-06-30T12:00:00+00:00",
        "approved_by": "operator_name",
        "approval_mode": "PAPER_ONLY",
        "approved_symbols": ["SLV", "CVX"],
        "max_notional": 17360.3,
        "source_packet_id": "def-exit-paper-xxxxxxxxxxxx-20260629T120000Z",
        "source_preview_hash": "sha256_hex_from_latest_preview",
        "acknowledgement_live_blocked": True,
        "acknowledgement_no_live_execution": True,
        "acknowledgement_paper_only": True,
        "decision": OPERATOR_DECISION_APPROVE,
    }
