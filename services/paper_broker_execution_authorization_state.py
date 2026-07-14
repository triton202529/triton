"""
paper_broker_execution_authorization_state.py
Phase 197Y — Classify paper broker execution authorization artifact lifecycle.
Read-only; does not submit broker orders or create authorization artifacts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
EXEC_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
DRY_RUN_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
PROJECTED_ORDERS_CSV = RESULTS / "phase_197x_projected_paper_orders.csv"

PAPER_BROKER_EXECUTION_AUTH_ABSENT = "PAPER_BROKER_EXECUTION_AUTH_ABSENT"
PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT = "PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT"
PAPER_BROKER_EXECUTION_AUTH_EXPIRED = "PAPER_BROKER_EXECUTION_AUTH_EXPIRED"
PAPER_BROKER_EXECUTION_AUTH_MALFORMED = "PAPER_BROKER_EXECUTION_AUTH_MALFORMED"
PAPER_BROKER_EXECUTION_AUTH_SCOPE_INVALID = "PAPER_BROKER_EXECUTION_AUTH_SCOPE_INVALID"
PAPER_BROKER_EXECUTION_AUTH_LIVE_INVALID = "PAPER_BROKER_EXECUTION_AUTH_LIVE_INVALID"
PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH = "PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH"
PAPER_BROKER_EXECUTION_AUTH_197W_APPROVAL_MISSING = (
    "PAPER_BROKER_EXECUTION_AUTH_197W_APPROVAL_MISSING"
)
PAPER_BROKER_EXECUTION_AUTH_197X_DRY_RUN_MISSING = (
    "PAPER_BROKER_EXECUTION_AUTH_197X_DRY_RUN_MISSING"
)
PAPER_BROKER_EXECUTION_AUTH_OPEN_ORDERS_PRESENT = "PAPER_BROKER_EXECUTION_AUTH_OPEN_ORDERS_PRESENT"

AUTH_ACTION = "EXECUTE_APPROVED_197X_CAPPED_DEFENSIVE_BATCH"
AUTH_ACTION_PARTIAL = "EXECUTE_APPROVED_197X_PARTIAL_DEFENSIVE_EXIT"
AUTH_ACTIONS_ALLOWED = frozenset({AUTH_ACTION, AUTH_ACTION_PARTIAL})
AUTH_STATUS_APPROVED = "AUTHORIZED"
PAPER_SCOPE = "PAPER_ONLY"
DEFAULT_TTL_MINUTES = 10
PARTIAL_AUTHORIZED_QTY_REQUIRED = 48

INVALID_AUTH_STATES = frozenset(
    {
        PAPER_BROKER_EXECUTION_AUTH_MALFORMED,
        PAPER_BROKER_EXECUTION_AUTH_SCOPE_INVALID,
        PAPER_BROKER_EXECUTION_AUTH_LIVE_INVALID,
        PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH,
        PAPER_BROKER_EXECUTION_AUTH_197W_APPROVAL_MISSING,
        PAPER_BROKER_EXECUTION_AUTH_197X_DRY_RUN_MISSING,
        PAPER_BROKER_EXECUTION_AUTH_OPEN_ORDERS_PRESENT,
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


def compute_artifact_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def load_197x_projected_orders() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load 197X projected order rows and symbol list."""
    if not PROJECTED_ORDERS_CSV.is_file():
        return [], []
    try:
        import csv

        rows: List[Dict[str, Any]] = []
        with PROJECTED_ORDERS_CSV.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                rows.append(dict(row))
        symbols = sorted({str(r.get("symbol") or "").upper() for r in rows})
        return rows, symbols
    except Exception:
        return [], []


def load_197x_dry_run_authority() -> Dict[str, Any]:
    doc = _read_json(DRY_RUN_JSON)
    rows, symbols = load_197x_projected_orders()
    return {
        "dry_run_json_path": (
            str(DRY_RUN_JSON.relative_to(ROOT)).replace("\\", "/")
            if DRY_RUN_JSON.is_file()
            else None
        ),
        "dry_run_status": doc.get("status"),
        "dry_run_final_state": doc.get("final_state"),
        "dry_run_artifact_hash": compute_artifact_hash(DRY_RUN_JSON),
        "projected_orders_csv_path": (
            str(PROJECTED_ORDERS_CSV.relative_to(ROOT)).replace("\\", "/")
            if PROJECTED_ORDERS_CSV.is_file()
            else None
        ),
        "projected_orders_csv_hash": compute_artifact_hash(PROJECTED_ORDERS_CSV),
        "projected_order_count": len(rows),
        "projected_symbols": symbols,
        "projected_orders": rows,
        "dry_run_valid": doc.get("status") in ("PASS", "PASS_SAFE_BLOCKED")
        and doc.get("final_state")
        == "PAPER_BROKER_EXECUTION_DRY_RUN_READY_OPERATOR_EXECUTION_REQUIRED"
        and len(rows) > 0,
    }


def classify_paper_broker_execution_authorization(
    *,
    auth_path: Optional[Path] = None,
    open_orders_count: int = 0,
    broker_snapshot_ok: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify paper broker execution authorization artifact."""
    path = auth_path or EXEC_AUTH_PATH
    details: Dict[str, Any] = {
        "artifact_path": (
            str(path.relative_to(ROOT)).replace("\\", "/") if path.is_file() else None
        ),
        "artifact_exists": path.is_file(),
        "open_orders_count": open_orders_count,
        "broker_snapshot_ok": broker_snapshot_ok,
    }

    if open_orders_count > 0:
        return (
            PAPER_BROKER_EXECUTION_AUTH_OPEN_ORDERS_PRESENT,
            f"open_orders_count={open_orders_count}",
            details,
        )

    if not path.is_file():
        return PAPER_BROKER_EXECUTION_AUTH_ABSENT, "execution_authorization_absent", details

    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            return PAPER_BROKER_EXECUTION_AUTH_MALFORMED, "json_not_object", details
    except Exception as exc:
        return PAPER_BROKER_EXECUTION_AUTH_MALFORMED, f"json_parse_error={exc}", details

    details["contract"] = contract

    if bool(contract.get("template_only")):
        return PAPER_BROKER_EXECUTION_AUTH_MALFORMED, "template_only_not_runtime_authority", details

    for field in (
        "authorization_status",
        "authorization_scope",
        "authorization_action",
        "created_at",
        "expires_at",
        "approval_artifact_path",
        "dry_run_artifact_path",
        "projected_orders_csv_path",
        "approved_symbols",
    ):
        if field == "approved_symbols":
            if not contract.get("approved_symbols"):
                return PAPER_BROKER_EXECUTION_AUTH_MALFORMED, "approved_symbols_empty", details
            continue
        if not str(contract.get(field) or "").strip():
            return PAPER_BROKER_EXECUTION_AUTH_MALFORMED, f"missing_{field}", details

    if str(contract.get("authorization_status") or "").upper() != AUTH_STATUS_APPROVED:
        return (
            PAPER_BROKER_EXECUTION_AUTH_MALFORMED,
            f"authorization_status={contract.get('authorization_status')}",
            details,
        )

    scope = str(contract.get("authorization_scope") or "").upper()
    if scope != PAPER_SCOPE:
        return PAPER_BROKER_EXECUTION_AUTH_SCOPE_INVALID, f"authorization_scope={scope}", details

    action = str(contract.get("authorization_action") or "").upper()
    if action not in AUTH_ACTIONS_ALLOWED:
        return PAPER_BROKER_EXECUTION_AUTH_MALFORMED, f"authorization_action={action}", details
    is_partial_auth = action == AUTH_ACTION_PARTIAL
    details["partial_authorization"] = is_partial_auth

    if contract.get("live_authorized") is True:
        return PAPER_BROKER_EXECUTION_AUTH_LIVE_INVALID, "live_authorized=true", details
    if contract.get("broker_execution_authorized") is not True:
        return (
            PAPER_BROKER_EXECUTION_AUTH_MALFORMED,
            "broker_execution_authorized_not_true",
            details,
        )
    if contract.get("raw_execution_authorized") is True:
        return PAPER_BROKER_EXECUTION_AUTH_MALFORMED, "raw_execution_authorized=true", details

    expires = _parse_iso(str(contract.get("expires_at") or ""))
    if not expires:
        return PAPER_BROKER_EXECUTION_AUTH_MALFORMED, "expires_at_missing_or_unparseable", details
    if expires <= _utc_now():
        return PAPER_BROKER_EXECUTION_AUTH_EXPIRED, "execution_authorization_expired", details

    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )

    ap_state, ap_reason, ap_details = classify_paper_operator_approval_artifact(
        broker_snapshot_ok=broker_snapshot_ok
    )
    details["paper_approval_state"] = ap_state
    if ap_state != PAPER_APPROVAL_VALID_CURRENT:
        return PAPER_BROKER_EXECUTION_AUTH_197W_APPROVAL_MISSING, ap_reason, details

    x197 = load_197x_dry_run_authority()
    details["197x_authority"] = x197
    if not x197.get("dry_run_valid"):
        return (
            PAPER_BROKER_EXECUTION_AUTH_197X_DRY_RUN_MISSING,
            "197x_dry_run_invalid_or_missing",
            details,
        )

    auth_symbols = _norm_symbols(contract.get("approved_symbols"))
    projected_symbols = _norm_symbols(x197.get("projected_symbols"))
    if auth_symbols != projected_symbols:
        return (
            PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH,
            f"auth_symbols={auth_symbols} projected={projected_symbols}",
            details,
        )

    if is_partial_auth:
        authorized_qty = int(
            contract.get("authorized_qty") or contract.get("partial_authorization_qty") or 0
        )
        qty_map = contract.get("authorized_quantities") or {}
        if isinstance(qty_map, dict) and auth_symbols:
            try:
                authorized_qty = int(qty_map.get(auth_symbols[0]) or authorized_qty or 0)
            except (TypeError, ValueError):
                return (
                    PAPER_BROKER_EXECUTION_AUTH_MALFORMED,
                    "invalid_authorized_quantities",
                    details,
                )
        projected_qty_map = x197.get("projected_quantities") or {}
        projected_qty = 0
        if isinstance(projected_qty_map, dict) and auth_symbols:
            try:
                projected_qty = int(float(projected_qty_map.get(auth_symbols[0]) or 0))
            except (TypeError, ValueError):
                projected_qty = 0
        if authorized_qty != PARTIAL_AUTHORIZED_QTY_REQUIRED:
            return (
                PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH,
                f"authorized_qty_not_48 qty={authorized_qty}",
                details,
            )
        if projected_qty and projected_qty != authorized_qty:
            return (
                PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH,
                f"authorized_qty_mismatch auth={authorized_qty} projected={projected_qty}",
                details,
            )
        if contract.get("full_exit_authorized") is True:
            return PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH, "full_exit_authorized=true", details
        details["authorized_qty"] = authorized_qty

    approval_hash = compute_artifact_hash(APPROVAL_PATH)
    if (
        str(contract.get("approval_artifact_hash") or "")
        and str(contract.get("approval_artifact_hash")) != approval_hash
    ):
        return (
            PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH,
            "approval_artifact_hash_mismatch",
            details,
        )

    dry_hash = compute_artifact_hash(DRY_RUN_JSON)
    if (
        str(contract.get("dry_run_artifact_hash") or "")
        and str(contract.get("dry_run_artifact_hash")) != dry_hash
    ):
        return PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH, "dry_run_artifact_hash_mismatch", details

    csv_hash = compute_artifact_hash(PROJECTED_ORDERS_CSV)
    if (
        str(contract.get("projected_orders_csv_hash") or "")
        and str(contract.get("projected_orders_csv_hash")) != csv_hash
    ):
        return (
            PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH,
            "projected_orders_csv_hash_mismatch",
            details,
        )

    for row in x197.get("projected_orders") or []:
        if str(row.get("broker_order_submitted") or "").lower() in ("true", "1", "yes"):
            return (
                PAPER_BROKER_EXECUTION_AUTH_BATCH_MISMATCH,
                "projected_order_already_submitted",
                details,
            )

    details.update(
        {
            "approved_symbols": auth_symbols,
            "approved_order_count": len(auth_symbols),
            "authorization_valid": True,
        }
    )
    return (
        PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
        "execution_authorization_valid_current",
        details,
    )
