"""
phase_197z_paper_broker_execution_authorization_creator.py
Phase 197Z — Create/validate paper broker execution authorization artifact tooling.
Default: dry-run only. Does not submit broker orders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197z_paper_broker_execution_authorization_creator.json"
EXEC_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
DRY_RUN_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
PROJECTED_ORDERS_CSV = RESULTS / "phase_197x_projected_paper_orders.csv"
PREVIOUS_PHASE = "197Y"
PREVIOUS_COMMIT = "e362576e"

TTL_MINUTES = 10
OPERATOR_CONFIRMATION_PHRASE = (
    "I authorize the paper broker execution phase for the approved capped defensive batch. "
    "Paper only. No live trading."
)

BLOCKED_197W = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_BLOCKED_197W_APPROVAL_REQUIRED"
BLOCKED_197X = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_BLOCKED_197X_DRY_RUN_REQUIRED"
BLOCKED_OPEN_ORDERS = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_BLOCKED_OPEN_ORDERS_PRESENT"
BLOCKED_LIVE = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_BLOCKED_LIVE_NOT_BLOCKED"
BLOCKED_CONFIRMATION = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_BLOCKED_OPERATOR_CONFIRMATION_REQUIRED"
BLOCKED_EXISTING_AUTH = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_BLOCKED_EXISTING_AUTHORIZATION"
READY_WOULD_CREATE = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_READY_WOULD_CREATE"
CREATED = "PAPER_BROKER_EXECUTION_AUTH_CREATOR_CREATED"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def evaluate_creator_preconditions(
    *,
    broker_snapshot_ok: bool = True,
    open_orders_count: Optional[int] = None,
    live_blocked: Optional[bool] = None,
    market_open: Optional[bool] = None,
    require_confirmation: bool = False,
    operator_confirmation: Optional[str] = None,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Evaluate prerequisites for paper broker execution authorization creation."""
    from services.defensive_exit_governance_review_criteria import (
        collect_governance_review_evidence,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.paper_broker_execution_authorization_state import (
        PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
        PAPER_SCOPE,
        compute_artifact_hash,
        classify_paper_broker_execution_authorization,
        load_197x_dry_run_authority,
        load_197x_projected_orders,
    )
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )

    reasons: List[str] = []
    context: Dict[str, Any] = {}

    if live_blocked is None:
        live_blocked = bool(evaluate_live_block_status(paper_mode=True).get("live_blocked"))
    evidence = collect_governance_review_evidence(broker_snapshot_ok=broker_snapshot_ok)
    if open_orders_count is None:
        open_orders_count = int(evidence.get("open_orders_count") or 0)
    if market_open is None:
        market_open = bool(evidence.get("market_open"))

    context.update(
        {
            "live_blocked": bool(live_blocked),
            "open_orders_count": open_orders_count,
            "market_open": bool(market_open),
        }
    )

    if not live_blocked:
        return BLOCKED_LIVE, ["live_blocked=false"], context

    if open_orders_count > 0:
        return BLOCKED_OPEN_ORDERS, [f"open_orders_count={open_orders_count}"], context

    ap_state, ap_reason, ap_details = classify_paper_operator_approval_artifact(
        broker_snapshot_ok=broker_snapshot_ok
    )
    context["paper_approval_state"] = ap_state
    context["paper_approval_reason"] = ap_reason

    if ap_state != PAPER_APPROVAL_VALID_CURRENT:
        return BLOCKED_197W, [ap_reason or f"paper_approval_state={ap_state}"], context

    contract = ap_details.get("contract") or {}
    scope = str(contract.get("approval_scope") or contract.get("approval_mode") or "").upper()
    if scope != PAPER_SCOPE:
        return BLOCKED_197W, [f"approval_scope={scope}"], context
    if contract.get("broker_execution_authorized") is True:
        return BLOCKED_197W, ["197w_broker_execution_authorized=true"], context
    if contract.get("live_authorized") is True:
        return BLOCKED_197W, ["197w_live_authorized=true"], context

    x197 = load_197x_dry_run_authority()
    context["197x_authority"] = x197
    if not x197.get("dry_run_valid"):
        return BLOCKED_197X, ["197x_dry_run_invalid_or_missing"], context

    rows, symbols = load_197x_projected_orders()
    if not PROJECTED_ORDERS_CSV.is_file() or len(rows) == 0:
        return BLOCKED_197X, ["projected_orders_csv_missing_or_empty"], context

    for row in rows:
        if str(row.get("broker_order_submitted") or "").lower() in ("true", "1", "yes"):
            return BLOCKED_197X, ["projected_order_already_submitted"], context

    if not market_open:
        return BLOCKED_197X, ["market_closed"], context

    auth_state, auth_reason, auth_details = classify_paper_broker_execution_authorization(
        open_orders_count=0,
        broker_snapshot_ok=broker_snapshot_ok,
    )
    context["existing_execution_auth_state"] = auth_state
    if auth_state == PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT:
        return BLOCKED_EXISTING_AUTH, [auth_reason], context

    if require_confirmation:
        if not operator_confirmation or operator_confirmation != OPERATOR_CONFIRMATION_PHRASE:
            return BLOCKED_CONFIRMATION, ["operator_confirmation_phrase_mismatch"], context

    notional = round(
        sum(float(r.get("notional_estimate") or 0) for r in rows),
        2,
    )
    context.update(
        {
            "approved_symbols": symbols,
            "approved_order_count": len(symbols),
            "approved_notional_estimate": notional,
            "approval_artifact_hash": compute_artifact_hash(APPROVAL_PATH),
            "dry_run_artifact_hash": compute_artifact_hash(DRY_RUN_JSON),
            "projected_orders_csv_hash": compute_artifact_hash(PROJECTED_ORDERS_CSV),
        }
    )
    return READY_WOULD_CREATE, reasons, context


def build_execution_authorization_artifact(
    *,
    context: Dict[str, Any],
    operator_confirmation_text: str,
) -> Dict[str, Any]:
    now = _utc_now()
    expires = now + timedelta(minutes=TTL_MINUTES)
    symbols = list(context.get("approved_symbols") or [])

    return {
        "authorization_id": str(uuid.uuid4()),
        "authorization_status": "AUTHORIZED",
        "authorization_scope": "PAPER_ONLY",
        "authorization_action": "EXECUTE_APPROVED_197X_CAPPED_DEFENSIVE_BATCH",
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "ttl_minutes": TTL_MINUTES,
        "operator_confirmation_text": operator_confirmation_text,
        "approval_artifact_path": _rel(APPROVAL_PATH),
        "approval_artifact_hash": str(context.get("approval_artifact_hash") or ""),
        "dry_run_artifact_path": _rel(DRY_RUN_JSON),
        "dry_run_artifact_hash": str(context.get("dry_run_artifact_hash") or ""),
        "projected_orders_csv_path": _rel(PROJECTED_ORDERS_CSV),
        "projected_orders_csv_hash": str(context.get("projected_orders_csv_hash") or ""),
        "approved_symbols": symbols,
        "approved_order_count": len(symbols),
        "approved_notional_estimate": float(context.get("approved_notional_estimate") or 0.0),
        "broker_execution_authorized": True,
        "live_authorized": False,
        "raw_execution_authorized": False,
        "eligible_order_source": "phase_197x_projected_paper_orders.csv",
        "execution_must_use_loop": True,
        "execution_must_use_approved_scope_only": True,
        "phase": "197Z",
        "warning": (
            "Paper broker execution authorization only. Loop-mediated execution required. "
            "No live trading."
        ),
    }


def write_execution_authorization_artifact(contract: Dict[str, Any]) -> None:
    GOVERNANCE.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(EXEC_AUTH_PATH, contract)


def run_creator(
    *,
    create: bool = False,
    operator_confirmation: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    final_state, block_reasons, context = evaluate_creator_preconditions(
        require_confirmation=create,
        operator_confirmation=operator_confirmation,
    )

    would_create = final_state == READY_WOULD_CREATE
    real_created = False
    contract: Dict[str, Any] = {}

    if create and would_create:
        contract = build_execution_authorization_artifact(
            context=context,
            operator_confirmation_text=OPERATOR_CONFIRMATION_PHRASE,
        )
        write_execution_authorization_artifact(contract)
        real_created = True
        final_state = CREATED

    if real_created:
        status = "PASS"
    elif would_create:
        status = "PASS_DRY_RUN"
    else:
        status = "PASS_SAFE_BLOCKED"

    payload: Dict[str, Any] = {
        "phase": "197Z",
        "generated_at": _utc_now_iso(),
        "mode": "create" if create else "dry_run",
        "status": status,
        "final_state": final_state,
        "would_create_authorization": would_create and not real_created,
        "real_authorization_created": real_created,
        "authorization_artifact_path": _rel(EXEC_AUTH_PATH),
        "authorization_artifact_exists": EXEC_AUTH_PATH.is_file(),
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "live_blocked": bool(context.get("live_blocked")),
        "open_orders_count": int(context.get("open_orders_count") or 0),
        "market_open": bool(context.get("market_open")),
        "paper_approval_state": context.get("paper_approval_state"),
        "197x_dry_run_valid": bool((context.get("197x_authority") or {}).get("dry_run_valid")),
        "approved_symbols": context.get("approved_symbols") or [],
        "approved_order_count": int(context.get("approved_order_count") or 0),
        "approved_notional_estimate": float(context.get("approved_notional_estimate") or 0.0),
        "operator_confirmation_required_phrase": OPERATOR_CONFIRMATION_PHRASE,
        "block_reasons": block_reasons,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "projected_authorization_preview": (
            build_execution_authorization_artifact(
                context=context,
                operator_confirmation_text=OPERATOR_CONFIRMATION_PHRASE,
            )
            if would_create
            else None
        ),
        "authorization_contract": contract if real_created else None,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197Z-CREATOR] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Z-CREATOR] status={status} final_state={final_state} "
            f"would_create={payload['would_create_authorization']} "
            f"real_created={real_created}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 197Z paper broker execution authorization creator (dry-run default)"
    )
    ap.add_argument(
        "--create",
        action="store_true",
        help="Create real authorization artifact (requires exact operator confirmation)",
    )
    ap.add_argument(
        "--operator-confirmation",
        default="",
        help="Exact operator confirmation phrase (required with --create)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    result = run_creator(
        create=bool(args.create),
        operator_confirmation=args.operator_confirmation or None,
        verbose=not args.quiet,
    )
    if result.get("status") == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
