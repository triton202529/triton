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
W197_JSON = RESULTS / "phase_197w_create_paper_operator_approval_artifact.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _extract_partial_auth_context() -> Dict[str, Any]:
    """Normalize partial defensive-exit fields for broker-auth dry-run reporting."""
    from services.partial_defensive_exit_preview import (
        PREVIEW_TYPE,
        extract_partial_approval_contract_fields,
    )

    preview_doc = _read_json(PREVIEW_JSON)
    x197 = _read_json(DRY_RUN_JSON)
    partial = extract_partial_approval_contract_fields(preview_doc)
    qty_map = x197.get("projected_quantities") or partial.get("approved_quantities") or {}
    symbol = str(
        partial.get("symbol") or (x197.get("projected_symbols") or [None])[0] or ""
    ).upper()
    authorized_qty = int(qty_map.get(symbol) or partial.get("partial_qty") or 0)
    full_qty = int(partial.get("full_qty") or 0)
    remaining = int(partial.get("remaining_qty_after_preview") or max(0, full_qty - authorized_qty))
    notional = float(x197.get("projected_notional") or partial.get("estimated_notional") or 0.0)
    cap = float(x197.get("batch_notional_cap") or 50000.0)
    available = bool(partial.get("partial_preview_available")) and authorized_qty > 0
    return {
        "partial_broker_authorization_supported": available,
        "preview_type": PREVIEW_TYPE if available else None,
        "partial_authorization_symbol": symbol if available else None,
        "symbol": symbol if available else None,
        "authorized_action": "PARTIAL_EXIT" if available else None,
        "authorized_qty": authorized_qty if available else 0,
        "partial_authorization_qty": authorized_qty if available else 0,
        "full_qty": full_qty if available else 0,
        "remaining_qty_after_authorized_preview": remaining if available else 0,
        "estimated_notional": round(notional, 2) if available else 0.0,
        "partial_authorization_notional": round(notional, 2) if available else 0.0,
        "estimated_notional_under_cap": bool(available and notional <= cap),
        "full_exit_deferred": bool(available and (partial.get("full_exit_deferred") or True)),
        "full_exit_deferred_reason": (
            partial.get("full_exit_deferred_reason") if available else None
        ),
        "full_exit_authorized": False,
    }


def _partial_dry_run_compat_available(
    *,
    x197_authority: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """True when 197X/197W dry-run chain proves partial projection without real approval."""
    x197 = _read_json(DRY_RUN_JSON)
    w197 = _read_json(W197_JSON)
    partial = _extract_partial_auth_context()
    if not partial.get("partial_broker_authorization_supported"):
        return False, "partial_preview_unavailable"
    if not x197_authority.get("dry_run_valid"):
        return False, "197x_dry_run_invalid"
    if not rows:
        return False, "projected_orders_empty"
    qty = int(partial.get("authorized_qty") or 0)
    full_qty = int(partial.get("full_qty") or 0)
    projected_qty = 0
    for row in rows:
        if str(row.get("symbol") or "").upper() == str(partial.get("symbol") or "").upper():
            try:
                projected_qty = int(float(row.get("quantity") or 0))
            except (TypeError, ValueError):
                projected_qty = 0
            break
    if projected_qty != qty or qty <= 0:
        return False, f"projected_qty_mismatch projected={projected_qty} expected={qty}"
    if full_qty > 0 and projected_qty >= full_qty:
        return False, "projected_full_exit_not_partial"
    if not bool(w197.get("dry_run_validated")) and not bool(x197.get("using_dry_run_contract")):
        return False, "197w_dry_run_not_validated"
    return True, ""


def evaluate_creator_preconditions(
    *,
    broker_snapshot_ok: bool = True,
    open_orders_count: Optional[int] = None,
    live_blocked: Optional[bool] = None,
    market_open: Optional[bool] = None,
    require_confirmation: bool = False,
    operator_confirmation: Optional[str] = None,
    create: bool = False,
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
        PAPER_APPROVAL_ABSENT,
        PAPER_APPROVAL_VALID_CURRENT,
        classify_paper_operator_approval_artifact,
    )
    from services.portfolio_history_sync import load_live_execution_flag

    reasons: List[str] = []
    context: Dict[str, Any] = {}
    partial_ctx = _extract_partial_auth_context()
    context.update(partial_ctx)
    context["live_execution_permitted"] = bool(load_live_execution_flag())
    context["broker_authorization_created"] = EXEC_AUTH_PATH.is_file()
    context["broker_orders_submitted"] = False

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

    x197 = load_197x_dry_run_authority()
    context["197x_authority"] = x197
    rows, symbols = load_197x_projected_orders()

    partial_compat = False
    partial_compat_reason = ""
    if not create and ap_state == PAPER_APPROVAL_ABSENT:
        partial_compat, partial_compat_reason = _partial_dry_run_compat_available(
            x197_authority=x197, rows=rows
        )
        context["partial_dry_run_compat"] = partial_compat
        context["partial_dry_run_compat_reason"] = partial_compat_reason

    if ap_state != PAPER_APPROVAL_VALID_CURRENT:
        if not (not create and partial_compat):
            return BLOCKED_197W, [ap_reason or f"paper_approval_state={ap_state}"], context
        context["using_partial_dry_run_compat"] = True
    else:
        contract = ap_details.get("contract") or {}
        scope = str(contract.get("approval_scope") or contract.get("approval_mode") or "").upper()
        if scope != PAPER_SCOPE:
            return BLOCKED_197W, [f"approval_scope={scope}"], context
        if contract.get("broker_execution_authorized") is True:
            return BLOCKED_197W, ["197w_broker_execution_authorized=true"], context
        if contract.get("live_authorized") is True:
            return BLOCKED_197W, ["197w_live_authorized=true"], context
        if contract.get("live_execution_allowed") is True:
            return BLOCKED_197W, ["197w_live_execution_allowed=true"], context
        # Real create: approval scope must match partial projected qty (GLD 48).
        if create and partial_ctx.get("partial_broker_authorization_supported"):
            expected_qty = int(partial_ctx.get("authorized_qty") or 0)
            approval_qty = int(
                contract.get("authorized_qty")
                or contract.get("partial_qty")
                or (
                    (contract.get("approved_quantities") or {}).get(
                        str(partial_ctx.get("symbol") or "").upper()
                    )
                    if isinstance(contract.get("approved_quantities"), dict)
                    else 0
                )
                or 0
            )
            if expected_qty != 48:
                return (
                    BLOCKED_197W,
                    [f"projected_authorized_qty_not_48 qty={expected_qty}"],
                    context,
                )
            if approval_qty != 48:
                return BLOCKED_197W, [f"197w_approval_qty_not_48 qty={approval_qty}"], context
            if str(contract.get("symbol") or "").upper() not in ("", "GLD"):
                if "GLD" not in [str(s).upper() for s in (contract.get("approved_symbols") or [])]:
                    return BLOCKED_197W, ["197w_approval_symbol_not_gld"], context
        context["using_partial_dry_run_compat"] = False
        context["approval_artifact_path"] = _rel(APPROVAL_PATH)

    if not x197.get("dry_run_valid"):
        return BLOCKED_197X, ["197x_dry_run_invalid_or_missing"], context

    if not PROJECTED_ORDERS_CSV.is_file() or len(rows) == 0:
        return BLOCKED_197X, ["projected_orders_csv_missing_or_empty"], context

    for row in rows:
        if str(row.get("broker_order_submitted") or "").lower() in ("true", "1", "yes"):
            return BLOCKED_197X, ["projected_order_already_submitted"], context

    # Real creation still requires market open; dry-run partial compat may run after hours.
    if not market_open and create:
        return BLOCKED_197X, ["market_closed"], context
    if not market_open and not context.get("using_partial_dry_run_compat"):
        return BLOCKED_197X, ["market_closed"], context

    auth_state, auth_reason, _auth_details = classify_paper_broker_execution_authorization(
        open_orders_count=0,
        broker_snapshot_ok=broker_snapshot_ok,
    )
    context["existing_execution_auth_state"] = auth_state
    if auth_state == PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT:
        return BLOCKED_EXISTING_AUTH, [auth_reason], context

    if require_confirmation or create:
        if not operator_confirmation or operator_confirmation != OPERATOR_CONFIRMATION_PHRASE:
            return BLOCKED_CONFIRMATION, ["operator_confirmation_phrase_mismatch"], context

    notional = round(
        sum(float(r.get("notional_estimate") or 0) for r in rows),
        2,
    )
    # Prefer partial projected quantities when present.
    if partial_ctx.get("partial_broker_authorization_supported"):
        symbols = [str(partial_ctx.get("symbol") or "").upper()]
        notional = float(partial_ctx.get("estimated_notional") or notional)

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
    partial = bool(context.get("partial_broker_authorization_supported"))

    artifact: Dict[str, Any] = {
        "authorization_id": str(uuid.uuid4()),
        "authorization_status": "AUTHORIZED",
        "authorization_scope": "PAPER_ONLY",
        "authorization_action": (
            "EXECUTE_APPROVED_197X_PARTIAL_DEFENSIVE_EXIT"
            if partial
            else "EXECUTE_APPROVED_197X_CAPPED_DEFENSIVE_BATCH"
        ),
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
    if partial:
        artifact.update(
            {
                "preview_type": context.get("preview_type"),
                "authorized_action": "PARTIAL_EXIT",
                "authorized_qty": int(context.get("authorized_qty") or 0),
                "authorized_quantities": {
                    str(context.get("symbol") or "").upper(): int(
                        context.get("authorized_qty") or 0
                    )
                },
                "full_qty": int(context.get("full_qty") or 0),
                "remaining_qty_after_authorized_preview": int(
                    context.get("remaining_qty_after_authorized_preview") or 0
                ),
                "full_exit_deferred": True,
                "full_exit_authorized": False,
                "partial_authorization_symbol": context.get("partial_authorization_symbol"),
                "partial_authorization_qty": int(context.get("partial_authorization_qty") or 0),
                "partial_authorization_notional": float(
                    context.get("partial_authorization_notional") or 0.0
                ),
            }
        )
    return artifact


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
        create=create,
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

    # Hard safety: dry-run must never leave a real authorization artifact.
    if not create and EXEC_AUTH_PATH.is_file():
        status = "FAIL"
        block_reasons = list(block_reasons) + ["dry_run_left_broker_authorization_artifact"]

    payload: Dict[str, Any] = {
        "phase": "197Z",
        "generated_at": _utc_now_iso(),
        "mode": "create" if create else "dry_run",
        "status": status,
        "final_state": final_state,
        "would_create_authorization": would_create and not real_created,
        "broker_authorization_would_be_created": would_create and not real_created,
        "real_authorization_created": real_created,
        "broker_authorization_created": real_created,
        "authorization_artifact_path": _rel(EXEC_AUTH_PATH),
        "authorization_artifact_exists": EXEC_AUTH_PATH.is_file(),
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "live_blocked": bool(context.get("live_blocked")),
        "live_execution_permitted": bool(context.get("live_execution_permitted")),
        "open_orders_count": int(context.get("open_orders_count") or 0),
        "market_open": bool(context.get("market_open")),
        "paper_approval_state": context.get("paper_approval_state"),
        "197x_dry_run_valid": bool((context.get("197x_authority") or {}).get("dry_run_valid")),
        "approved_symbols": context.get("approved_symbols") or [],
        "approved_order_count": int(context.get("approved_order_count") or 0),
        "approved_notional_estimate": float(context.get("approved_notional_estimate") or 0.0),
        "partial_broker_authorization_supported": bool(
            context.get("partial_broker_authorization_supported")
        ),
        "preview_type": context.get("preview_type"),
        "symbol": context.get("symbol"),
        "authorized_action": context.get("authorized_action"),
        "authorized_qty": context.get("authorized_qty"),
        "partial_authorization_symbol": context.get("partial_authorization_symbol"),
        "partial_authorization_qty": context.get("partial_authorization_qty"),
        "partial_authorization_notional": context.get("partial_authorization_notional"),
        "full_qty": context.get("full_qty"),
        "remaining_qty_after_authorized_preview": context.get(
            "remaining_qty_after_authorized_preview"
        ),
        "estimated_notional": context.get("estimated_notional"),
        "estimated_notional_under_cap": context.get("estimated_notional_under_cap"),
        "full_exit_deferred": bool(context.get("full_exit_deferred")),
        "full_exit_authorized": False,
        "full_exit_deferred_reason": context.get("full_exit_deferred_reason"),
        "using_partial_dry_run_compat": bool(context.get("using_partial_dry_run_compat")),
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
            f"real_created={real_created} "
            f"partial={payload.get('partial_broker_authorization_supported')} "
            f"authorized_qty={payload.get('authorized_qty')}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 197Z paper broker execution authorization creator. "
            "Default is dry-run. Real creation requires valid "
            "data/governance/operator_paper_defensive_exit_approval.json "
            "from 197W, plus --create and the exact operator confirmation phrase."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Safe default: evaluate only; do not write broker auth\n"
            "  python -m services.phase_197z_paper_broker_execution_authorization_creator\n"
            "\n"
            "  # Real create (only after valid 197W approval artifact exists):\n"
            "  python -m services.phase_197z_paper_broker_execution_authorization_creator "
            "--create --operator-confirmation "
            '"I authorize the paper broker execution phase for the approved capped '
            'defensive batch. Paper only. No live trading."\n'
            "\n"
            "Requires approval path:\n"
            "  data/governance/operator_paper_defensive_exit_approval.json\n"
            "Writes (on --create only):\n"
            "  data/governance/operator_paper_broker_execution_authorization.json\n"
            "Does NOT submit broker orders."
        ),
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
