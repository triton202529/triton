"""
paper_operator_approval_artifact_state.py
Phase 197W-FIX — Classify paper-only operator defensive-exit approval artifact lifecycle.
Read-only; does not submit orders or mutate governance unless cleanup module invoked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"

PAPER_APPROVAL_ABSENT = "PAPER_APPROVAL_ABSENT"
PAPER_APPROVAL_VALID_CURRENT = "PAPER_APPROVAL_VALID_CURRENT"
PAPER_APPROVAL_EXPIRED = "PAPER_APPROVAL_EXPIRED"
PAPER_APPROVAL_MALFORMED = "PAPER_APPROVAL_MALFORMED"
PAPER_APPROVAL_SCOPE_INVALID = "PAPER_APPROVAL_SCOPE_INVALID"
PAPER_APPROVAL_PACKET_MISMATCH = "PAPER_APPROVAL_PACKET_MISMATCH"
PAPER_APPROVAL_LIVE_AUTH_INVALID = "PAPER_APPROVAL_LIVE_AUTH_INVALID"
PAPER_APPROVAL_EXECUTION_AUTH_INVALID = "PAPER_APPROVAL_EXECUTION_AUTH_INVALID"

APPROVE_ACTION = "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH"
APPROVED_STATUS = "APPROVED"
PAPER_SCOPE = "PAPER_ONLY"

INVALID_STATES = frozenset(
    {
        PAPER_APPROVAL_MALFORMED,
        PAPER_APPROVAL_SCOPE_INVALID,
        PAPER_APPROVAL_PACKET_MISMATCH,
        PAPER_APPROVAL_LIVE_AUTH_INVALID,
        PAPER_APPROVAL_EXECUTION_AUTH_INVALID,
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _norm_symbols(symbols: Any) -> List[str]:
    return sorted({str(s).strip().upper() for s in (symbols or []) if str(s).strip()})


def _runtime_capped_preview(doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        doc.get("capped_preview_runtime_cap")
        or doc.get("capped_preview")
        or doc.get("capped_preview_cli_cap")
        or {}
    )


def _current_packet_context() -> Dict[str, Any]:
    packet = _read_json(OPERATOR_PACKET_JSON)
    preview = _read_json(PREVIEW_JSON)
    capped_block = _runtime_capped_preview(preview)
    deferred = _norm_symbols(
        packet.get("deferred_symbols") or capped_block.get("deferred_symbols") or []
    )
    capped = _norm_symbols(
        packet.get("capped_symbols") or capped_block.get("capped_preview_symbols") or []
    )
    return {
        "packet": packet,
        "preview": preview,
        "packet_id": str(packet.get("packet_id") or ""),
        "source_preview_hash": str(packet.get("source_preview_hash") or ""),
        "capped_symbols": capped,
        "deferred_symbols": deferred,
    }


def classify_paper_operator_approval_artifact(
    *,
    contract_path: Optional[Path] = None,
    broker_snapshot_ok: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify paper-only operator approval artifact lifecycle state."""
    path = contract_path or APPROVAL_PATH
    details: Dict[str, Any] = {
        "artifact_path": (
            str(path.relative_to(ROOT)).replace("\\", "/") if path.is_file() else None
        ),
        "artifact_exists": path.is_file(),
        "broker_snapshot_ok": broker_snapshot_ok,
    }

    if not path.is_file():
        return PAPER_APPROVAL_ABSENT, "approval_artifact_absent", details

    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            return PAPER_APPROVAL_MALFORMED, "json_not_object", details
    except Exception as exc:
        return PAPER_APPROVAL_MALFORMED, f"json_parse_error={exc}", details

    details["contract"] = contract
    details["expires_at"] = contract.get("expires_at")
    details["created_at"] = contract.get("created_at")
    details["packet_id"] = contract.get("packet_id") or contract.get("source_packet_id")
    details["source_preview_hash"] = contract.get("source_preview_hash")

    if bool(contract.get("template_only")):
        return PAPER_APPROVAL_MALFORMED, "template_only_not_runtime_authority", details

    for field in (
        "approval_status",
        "approval_action",
        "approval_scope",
        "created_at",
        "expires_at",
        "source_preview_hash",
        "approved_symbols",
    ):
        if field == "approved_symbols":
            if not contract.get("approved_symbols"):
                return PAPER_APPROVAL_MALFORMED, "approved_symbols_empty", details
            continue
        if not str(contract.get(field) or "").strip():
            return PAPER_APPROVAL_MALFORMED, f"missing_{field}", details

    packet_id_field = str(contract.get("packet_id") or contract.get("source_packet_id") or "")
    if not packet_id_field:
        return PAPER_APPROVAL_MALFORMED, "missing_packet_id", details

    if str(contract.get("approval_status") or "").upper() != APPROVED_STATUS:
        return (
            PAPER_APPROVAL_MALFORMED,
            f"approval_status={contract.get('approval_status')}",
            details,
        )

    action = str(contract.get("approval_action") or contract.get("decision") or "").upper()
    if action != APPROVE_ACTION:
        return PAPER_APPROVAL_MALFORMED, f"approval_action={action}", details

    scope = str(contract.get("approval_scope") or contract.get("approval_mode") or "").upper()
    if scope != PAPER_SCOPE:
        return PAPER_APPROVAL_SCOPE_INVALID, f"approval_scope={scope}", details

    if contract.get("live_authorized") is True:
        return PAPER_APPROVAL_LIVE_AUTH_INVALID, "live_authorized=true", details
    if contract.get("broker_execution_authorized") is True:
        return PAPER_APPROVAL_EXECUTION_AUTH_INVALID, "broker_execution_authorized=true", details
    if contract.get("raw_execution_authorized") is True:
        return PAPER_APPROVAL_EXECUTION_AUTH_INVALID, "raw_execution_authorized=true", details

    for ack in (
        "acknowledgement_live_blocked",
        "acknowledgement_no_live_execution",
        "acknowledgement_paper_only",
    ):
        if ack in contract and not bool(contract.get(ack)):
            return PAPER_APPROVAL_MALFORMED, f"{ack}=false", details

    expires = _parse_iso(str(contract.get("expires_at") or ""))
    if not expires:
        return PAPER_APPROVAL_MALFORMED, "expires_at_missing_or_unparseable", details
    details["expires_at_parsed"] = expires.isoformat()
    details["expired"] = expires <= _utc_now()
    if expires <= _utc_now():
        return PAPER_APPROVAL_EXPIRED, "approval_expired", details

    created = _parse_iso(str(contract.get("created_at") or contract.get("approved_at") or ""))
    if not created:
        return PAPER_APPROVAL_MALFORMED, "created_at_missing_or_unparseable", details

    ctx = _current_packet_context()
    if packet_id_field != ctx["packet_id"]:
        return (
            PAPER_APPROVAL_PACKET_MISMATCH,
            f"packet_id_mismatch artifact={packet_id_field} current={ctx['packet_id']}",
            details,
        )
    if str(contract.get("source_preview_hash") or "") != ctx["source_preview_hash"]:
        return PAPER_APPROVAL_PACKET_MISMATCH, "source_preview_hash_mismatch", details

    approved = _norm_symbols(contract.get("approved_symbols"))
    deferred = _norm_symbols(contract.get("deferred_symbols") or ctx["deferred_symbols"])
    capped = ctx["capped_symbols"]

    # Partial defensive-exit approvals authorize one symbol/qty (e.g. GLD 48),
    # not the full capped multi-symbol batch list.
    is_partial_contract = False
    try:
        from services.partial_defensive_exit_preview import (
            PREVIEW_TYPE,
            extract_partial_approval_contract_fields,
        )

        partial_fields = extract_partial_approval_contract_fields(ctx["preview"])
        contract_preview_type = str(contract.get("preview_type") or "").strip().upper()
        approved_quantities_raw = contract.get("approved_quantities") or {}
        is_partial_contract = contract_preview_type == str(PREVIEW_TYPE).upper() or (
            isinstance(approved_quantities_raw, dict) and bool(approved_quantities_raw)
        )
        if is_partial_contract:
            if not partial_fields.get("partial_preview_available"):
                return PAPER_APPROVAL_PACKET_MISMATCH, "partial_preview_unavailable", details
            expected_symbol = str(partial_fields.get("symbol") or "").upper()
            expected_qty = int(partial_fields.get("partial_qty") or 0)
            if approved != [expected_symbol]:
                return (
                    PAPER_APPROVAL_PACKET_MISMATCH,
                    f"partial_approved_symbols_mismatch approved={approved} expected={[expected_symbol]}",
                    details,
                )
            qty_map: Dict[str, int] = {}
            for sym, qty in (approved_quantities_raw or {}).items():
                try:
                    qty_map[str(sym).strip().upper()] = int(float(qty))
                except (TypeError, ValueError):
                    return (
                        PAPER_APPROVAL_MALFORMED,
                        f"invalid_approved_quantity symbol={sym}",
                        details,
                    )
            if qty_map.get(expected_symbol) != expected_qty:
                return (
                    PAPER_APPROVAL_PACKET_MISMATCH,
                    f"partial_qty_mismatch expected={expected_qty} got={qty_map.get(expected_symbol)}",
                    details,
                )
            auth_qty = int(contract.get("authorized_qty") or contract.get("partial_qty") or 0)
            if auth_qty and auth_qty != expected_qty:
                return (
                    PAPER_APPROVAL_PACKET_MISMATCH,
                    f"authorized_qty_mismatch expected={expected_qty} got={auth_qty}",
                    details,
                )
            details["partial_contract"] = True
            details["authorized_qty"] = expected_qty
            details["approved_quantities"] = qty_map
    except Exception as exc:
        if (
            str(contract.get("preview_type") or "").strip().upper()
            == "PARTIAL_DEFENSIVE_EXIT_PREVIEW"
        ):
            return PAPER_APPROVAL_MALFORMED, f"partial_classify_error={exc}", details

    if not is_partial_contract:
        if approved != capped:
            return (
                PAPER_APPROVAL_PACKET_MISMATCH,
                f"approved_symbols_mismatch approved={approved} capped={capped}",
                details,
            )
    if set(approved) & set(deferred) and not is_partial_contract:
        overlap = sorted(set(approved) & set(deferred))
        return (
            PAPER_APPROVAL_PACKET_MISMATCH,
            f"deferred_symbols_in_approved={overlap}",
            details,
        )
    # Partial path may list the approved symbol in deferred full-exit set; that is expected.
    if is_partial_contract and len(approved) != 1:
        return PAPER_APPROVAL_PACKET_MISMATCH, "partial_approved_symbols_must_be_single", details

    batch_notional = float(contract.get("batch_notional") or contract.get("max_notional") or 0.0)
    batch_cap = float(
        contract.get("batch_notional_cap") or ctx["packet"].get("configured_cap") or 0.0
    )
    if batch_cap > 0 and batch_notional > batch_cap:
        return (
            PAPER_APPROVAL_PACKET_MISMATCH,
            f"batch_notional_exceeds_cap notional={batch_notional} cap={batch_cap}",
            details,
        )

    if contract.get("live_blocked_at_approval") is False:
        return PAPER_APPROVAL_LIVE_AUTH_INVALID, "live_blocked_at_approval=false", details

    open_at_approval = contract.get("open_orders_count_at_approval")
    if open_at_approval is not None and int(open_at_approval) != 0:
        return (
            PAPER_APPROVAL_MALFORMED,
            f"open_orders_count_at_approval={open_at_approval}",
            details,
        )

    try:
        from services.live_block_classifier import scan_live_confirm_artifacts

        confirm = scan_live_confirm_artifacts()
        if int(confirm.get("valid_live_confirm_count") or 0) > 0:
            return PAPER_APPROVAL_LIVE_AUTH_INVALID, "valid_live_confirm_present", details
    except Exception:
        pass

    details.update(
        {
            "approved_symbols": approved,
            "deferred_symbols": deferred,
            "approval_valid": True,
            "approval_expired": False,
        }
    )
    return PAPER_APPROVAL_VALID_CURRENT, "paper_approval_valid_current", details


def is_paper_approval_valid_current(**kwargs: Any) -> bool:
    state, _, _ = classify_paper_operator_approval_artifact(**kwargs)
    return state == PAPER_APPROVAL_VALID_CURRENT


def is_paper_approval_expired(**kwargs: Any) -> bool:
    state, _, _ = classify_paper_operator_approval_artifact(**kwargs)
    return state == PAPER_APPROVAL_EXPIRED


def paper_approval_cleanup_required(**kwargs: Any) -> bool:
    state, _, _ = classify_paper_operator_approval_artifact(**kwargs)
    return state in {PAPER_APPROVAL_EXPIRED} | INVALID_STATES
