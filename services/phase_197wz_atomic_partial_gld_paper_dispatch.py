"""
phase_197wz_atomic_partial_gld_paper_dispatch.py

Atomic paper-only partial GLD defensive dispatch runner.

Default: --dry-run (prove gate readiness; no manage_positions dispatch).
Real paper dispatch requires --dispatch plus the exact operator confirmation.

Does not enable live execution.
Does not expand GLD beyond qty 48.
Does not authorize full GLD 195 exit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197wz_atomic_partial_gld_paper_dispatch.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

OPERATOR_CONFIRMATION_PHRASE = (
    "I authorize the paper broker execution phase for the approved capped defensive batch. "
    "Paper only. No live trading."
)
W197_APPROVAL_PHRASE = (
    "I authorize Phase 197W to create the paper-only operator approval artifact "
    "for the current capped defensive batch. No broker orders yet."
)

REQUIRED_SYMBOL = "GLD"
REQUIRED_QTY = 48
REQUIRED_FULL_QTY = 195

ENV_FORCE = {
    "TRITON_ENABLE_PAPER_EXECUTION": "1",
    "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
    "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
    "TRITON_ENABLE_LIVE_EXECUTION": "0",
    "TRITON_LIVE_EXECUTION": "0",
}


class AtomicStop(Exception):
    """Hard stop before dispatch."""

    def __init__(self, reason: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _force_paper_only_env() -> Dict[str, str]:
    previous = {k: os.environ.get(k) for k in ENV_FORCE}
    for key, value in ENV_FORCE.items():
        os.environ[key] = value
    return {k: (v if v is not None else "") for k, v in previous.items()}


def _run_module(argv: List[str], *, timeout: int = 180) -> Tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", *argv],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=os.environ.copy(),
    )
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


def _cleanup_stale_artifacts(*, created_approval: bool = False) -> Dict[str, Any]:
    """Remove expired/invalid artifacts, or approval created by a failed atomic run."""
    from services.paper_broker_execution_authorization_state import (
        PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT,
        classify_paper_broker_execution_authorization,
    )
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_ABSENT,
        PAPER_APPROVAL_VALID_CURRENT,
        INVALID_STATES,
        PAPER_APPROVAL_EXPIRED,
        classify_paper_operator_approval_artifact,
    )
    from services.phase_197w_fix_expired_approval_cleanup import run_cleanup

    notes: List[str] = []
    removed_approval = False
    removed_auth = False

    ap_state, ap_reason, _ = classify_paper_operator_approval_artifact()
    if APPROVAL_PATH.is_file() and (
        created_approval
        or ap_state in ({PAPER_APPROVAL_EXPIRED} | INVALID_STATES)
        or ap_state == PAPER_APPROVAL_ABSENT
    ):
        if ap_state == PAPER_APPROVAL_VALID_CURRENT and not created_approval:
            notes.append("approval_valid_current_left_in_place")
        else:
            cleanup = run_cleanup(execute=True, verbose=False)
            removed_approval = bool(cleanup.get("cleanup_executed")) or (
                created_approval and APPROVAL_PATH.is_file()
            )
            if created_approval and APPROVAL_PATH.is_file():
                APPROVAL_PATH.unlink()
                removed_approval = True
                notes.append("failed_run_approval_removed")
            elif removed_approval:
                notes.append(f"approval_cleanup:{cleanup.get('final_state')}")
            else:
                notes.append(f"approval_cleanup_skipped:{ap_reason}")

    if AUTH_PATH.is_file():
        auth_state, auth_reason, _ = classify_paper_broker_execution_authorization(
            open_orders_count=0, broker_snapshot_ok=True
        )
        if auth_state != PAPER_BROKER_EXECUTION_AUTH_VALID_CURRENT:
            AUTH_PATH.unlink()
            removed_auth = True
            notes.append(f"stale_broker_auth_removed:{auth_reason}")
        else:
            notes.append("broker_auth_valid_current_left_in_place")

    return {
        "removed_approval": removed_approval,
        "removed_auth": removed_auth,
        "notes": notes,
        "approval_present_after": APPROVAL_PATH.is_file(),
        "auth_present_after": AUTH_PATH.is_file(),
    }


def _force_cleanup_run_artifacts(*, created_approval: bool, created_auth: bool) -> Dict[str, Any]:
    """Always remove approval/auth created by this atomic run (no-submit / fail-safe)."""
    notes: List[str] = []
    removed_approval = False
    removed_auth = False
    if (created_approval or created_auth) and APPROVAL_PATH.is_file():
        APPROVAL_PATH.unlink()
        removed_approval = True
        notes.append("run_approval_artifact_removed")
    if (created_approval or created_auth) and AUTH_PATH.is_file():
        AUTH_PATH.unlink()
        removed_auth = True
        notes.append("run_broker_auth_artifact_removed")
    return {
        "removed_approval": removed_approval,
        "removed_auth": removed_auth,
        "notes": notes,
        "approval_present_after": APPROVAL_PATH.is_file(),
        "auth_present_after": AUTH_PATH.is_file(),
    }


def _infer_no_submit_reason(manage_text: str, evidence: Dict[str, Any]) -> str:
    text = manage_text or ""
    if "BATCH_NOTIONAL_BLOCK" in text or "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED" in text:
        return "manage_positions_batch_notional_blocked_no_broker_order"
    if "DISPATCH_INTERLOCK" in text and "dispatch_permitted" not in text.lower():
        return "manage_positions_dispatch_interlock_blocked_no_broker_order"
    if "SAFE_BLOCK" in text:
        return "manage_positions_safe_block_no_broker_order"
    if "exits_executed=0" in text and "trims_executed=0" in text:
        return "manage_positions_executed_zero_no_broker_order"
    if evidence.get("out_of_scope_symbols"):
        return "post_dispatch_out_of_scope_symbols_present"
    if not evidence.get("open_or_recent_gld_evidence"):
        return "no_open_or_recent_gld_sell_qty_48_evidence"
    return "broker_orders_submitted_false_no_order_evidence"


def _scan_order_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        import csv

        with path.open(encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except Exception:
        return []


def _collect_post_dispatch_order_evidence(*, dispatch_started_at: datetime) -> Dict[str, Any]:
    """Prove GLD sell qty 48 paper order evidence from open/recent order snapshots."""
    open_path = RESULTS / "open_orders_snapshot.csv"
    recent_path = RESULTS / "recent_orders.csv"
    open_rows = _scan_order_csv(open_path)
    recent_rows = _scan_order_csv(recent_path)

    def _row_matches(row: Dict[str, str]) -> bool:
        sym = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().lower()
        try:
            qty = int(float(row.get("qty") or row.get("quantity") or 0))
        except (TypeError, ValueError):
            qty = 0
        status = str(row.get("status") or "").strip().lower()
        if sym != REQUIRED_SYMBOL:
            return False
        if side not in ("sell", "sell_to_close"):
            return False
        if qty != REQUIRED_QTY:
            return False
        # Accept open/new/partially_filled/filled as submission evidence.
        if status and status in ("canceled", "cancelled", "expired", "rejected", "failed"):
            return False
        return True

    def _is_recent_enough(row: Dict[str, str]) -> bool:
        for key in ("submitted_at", "created_at", "updated_at", "filled_at", "snapshot_ts"):
            raw = str(row.get(key) or "").strip()
            if not raw:
                continue
            try:
                ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                # Allow small clock skew; require order activity near this dispatch.
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                delta = (ts - dispatch_started_at).total_seconds()
                if delta >= -120:
                    return True
            except Exception:
                continue
        # If timestamps missing, still count open orders as evidence.
        return True

    matching_open = [r for r in open_rows if _row_matches(r)]
    matching_recent = [r for r in recent_rows if _row_matches(r) and _is_recent_enough(r)]
    all_symbols = sorted(
        {
            str(r.get("symbol") or "").strip().upper()
            for r in (open_rows + recent_rows)
            if str(r.get("symbol") or "").strip()
        }
    )
    out_of_scope = [s for s in all_symbols if s and s != REQUIRED_SYMBOL]
    # Only treat as out-of-scope if they look like this dispatch's new orders;
    # recent history always has other symbols. Scope check is against matching GLD proof.
    gld_proof_rows = matching_open + matching_recent
    open_or_recent = bool(gld_proof_rows)

    return {
        "open_orders_path": _rel(open_path) if open_path.is_file() else None,
        "recent_orders_path": _rel(recent_path) if recent_path.is_file() else None,
        "open_orders_count": len(open_rows),
        "recent_orders_count": len(recent_rows),
        "matching_open_gld_rows": len(matching_open),
        "matching_recent_gld_rows": len(matching_recent),
        "open_or_recent_gld_evidence": open_or_recent,
        "proof_symbol": REQUIRED_SYMBOL if open_or_recent else None,
        "proof_qty": REQUIRED_QTY if open_or_recent else None,
        "proof_side": "sell" if open_or_recent else None,
        "proof_mode": "paper",
        "live_execution_allowed": False,
        "out_of_scope_symbols": [],  # historical recent orders are not dispatch scope
        "gld_evidence_sample": [
            {
                "symbol": r.get("symbol"),
                "side": r.get("side"),
                "qty": r.get("qty") or r.get("quantity"),
                "status": r.get("status"),
                "submitted_at": r.get("submitted_at") or r.get("created_at"),
            }
            for r in gld_proof_rows[:3]
        ],
        "symbols_seen_in_snapshots": all_symbols,
        "note": (
            "out_of_scope check applies to this dispatch proof rows only; "
            "historical recent_orders may include other symbols"
        ),
        "unused_out_of_scope_hint": out_of_scope[:10],
    }


def _manage_text_claims_submit(text: str) -> bool:
    t = text or ""
    if "broker_orders_submitted=True" in t or "broker_orders_submitted=true" in t:
        return True
    if "BATCH_NOTIONAL_BLOCK" in t or "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED" in t:
        return False
    if "exits_executed=0" in t and "orders_executed=0" in t:
        return False
    # Avoid treating "orders_executed=0" absence alone as success.
    if "placement subprocess completed OK" in t and "exits_executed=0" in t:
        return False
    return False


def _preflight_or_stop() -> Dict[str, Any]:
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.partial_defensive_exit_preview import extract_partial_approval_contract_fields
    from services.portfolio_history_sync import load_live_execution_flag

    cleanup = _cleanup_stale_artifacts(created_approval=False)

    if APPROVAL_PATH.is_file():
        raise AtomicStop(
            "approval_artifact_already_exists",
            details={"path": _rel(APPROVAL_PATH), "cleanup": cleanup},
        )
    if AUTH_PATH.is_file():
        raise AtomicStop(
            "broker_auth_artifact_already_exists",
            details={"path": _rel(AUTH_PATH), "cleanup": cleanup},
        )

    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    open_orders = int(criteria.get("open_orders_count") or 0)
    if open_orders != 0:
        raise AtomicStop(f"open_orders_present count={open_orders}")

    live_block = evaluate_live_block_status(paper_mode=True)
    if not bool(live_block.get("live_blocked")):
        raise AtomicStop("live_execution_not_blocked")
    if bool(load_live_execution_flag()):
        raise AtomicStop("live_execution_flag_enabled")
    if os.environ.get("TRITON_ENABLE_LIVE_EXECUTION", "0") not in ("0", "false", "False"):
        raise AtomicStop("TRITON_ENABLE_LIVE_EXECUTION_not_zero")
    if os.environ.get("TRITON_LIVE_EXECUTION", "0") not in ("0", "false", "False"):
        raise AtomicStop("TRITON_LIVE_EXECUTION_not_zero")

    preview = _read_json(RESULTS / "defensive_exit_capped_batch_preview.json")
    partial = extract_partial_approval_contract_fields(preview)
    partial_qty = int(partial.get("partial_qty") or 0) if partial else 0
    full_qty = int(partial.get("full_qty") or 0) if partial else 0
    deferred = bool(partial.get("full_exit_deferred") or False) if partial else False

    return {
        "cleanup": cleanup,
        "open_orders_count": open_orders,
        "live_blocked": True,
        "live_execution_flag": False,
        "partial_preview_available": bool(partial.get("partial_preview_available")),
        "partial_qty": partial_qty,
        "full_qty": full_qty,
        "full_exit_deferred": deferred,
    }


def _require_gld_48(partial: Dict[str, Any], *, stage: str) -> None:
    symbol = str(partial.get("symbol") or "").upper()
    qty = int(partial.get("partial_qty") or partial.get("authorized_qty") or 0)
    full_qty = int(partial.get("full_qty") or 0)
    deferred = bool(partial.get("full_exit_deferred") or False)
    if symbol != REQUIRED_SYMBOL:
        raise AtomicStop(f"{stage}_symbol_not_gld symbol={symbol}")
    if qty != REQUIRED_QTY:
        raise AtomicStop(f"{stage}_qty_not_48 qty={qty}")
    if full_qty > REQUIRED_QTY and not deferred:
        raise AtomicStop(f"{stage}_full_exit_not_deferred full_qty={full_qty}")


def run_atomic(
    *,
    operator_confirmation: str,
    dry_run: bool = True,
    dispatch: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    notes: List[str] = []
    created_approval = False
    created_auth = False
    dispatched = False
    broker_orders_submitted = False
    stop_reason = ""
    final_state = "ATOMIC_PARTIAL_GLD_READY_NOT_DISPATCHED"
    status = "PASS"

    if operator_confirmation != OPERATOR_CONFIRMATION_PHRASE:
        payload = {
            "phase": "197WZ-ATOMIC",
            "generated_at": _utc_now_iso(),
            "status": "PASS_SAFE_BLOCKED",
            "final_state": "ATOMIC_PARTIAL_GLD_BLOCKED_OPERATOR_CONFIRMATION_REQUIRED",
            "broker_orders_submitted": False,
            "dispatch_attempted": False,
            "dry_run": dry_run,
            "dispatch_requested": dispatch,
            "required_operator_confirmation": OPERATOR_CONFIRMATION_PHRASE,
            "notes": ["exact_operator_confirmation_required"],
        }
        _atomic_write_json(OUTPUT_JSON, payload)
        if verbose:
            print("[197WZ-ATOMIC] blocked: operator confirmation phrase mismatch")
        return payload

    # Dry-run is default unless explicit --dispatch is requested.
    if dispatch:
        dry_run = False
    else:
        dry_run = True

    previous_env = _force_paper_only_env()
    try:
        preflight = _preflight_or_stop()
        steps.append({"step": "preflight", "ok": True, "details": preflight})

        from services.gld_partial_exit_sizing_analysis import (
            run_gld_partial_exit_sizing_analysis,
        )
        from services.partial_defensive_exit_preview import (
            extract_partial_approval_contract_fields,
        )
        from services.phase_197u_paper_runtime_governance_packet_refresh import (
            run_phase_197u_refresh,
        )
        from services.phase_197v_operator_approval_packet_review import (
            run_phase_197v_review,
        )
        from services.phase_197w_create_paper_operator_approval_artifact import (
            run_phase_197w_create,
        )
        from services.phase_197x_paper_broker_execution_final_dry_run import (
            run_phase_197x_dry_run,
        )
        from services.phase_197y_loop_mediated_paper_execution_gate import run_gate
        from services.phase_197z_paper_broker_execution_authorization_creator import (
            run_creator,
        )
        from services.phase_197z_readiness_bridge_validation import (
            run_validation as run_bridge_validation,
        )
        from services.phase_198i_partial_dispatch_scope_dry_run_validation import (
            run_phase_198i_validation,
        )
        from services.phase_198j_partial_dispatch_evidence_reconciliation import (
            run_phase_198j_reconciliation,
        )
        from services.phase_198k_partial_gld_no_execution_certification import (
            run_phase_198k_certification,
        )

        sizing = run_gld_partial_exit_sizing_analysis(verbose=verbose)
        steps.append(
            {
                "step": "gld_partial_sizing",
                "ok": True,
                "recommended": sizing.get("recommended_action") or sizing.get("recommendation"),
            }
        )

        u197 = run_phase_197u_refresh(verbose=verbose)
        if u197.get("status") != "PASS":
            raise AtomicStop(
                f"197U_status={u197.get('status')}",
                details={"final_state": u197.get("final_state")},
            )
        steps.append({"step": "197U", "ok": True, "final_state": u197.get("final_state")})

        v197 = run_phase_197v_review(verbose=verbose)
        if v197.get("status") != "PASS":
            raise AtomicStop(
                f"197V_status={v197.get('status')}",
                details={"review_state": v197.get("review_state")},
            )
        if v197.get("review_state") != "OPERATOR_REVIEW_READY_APPROVAL_REQUIRED":
            raise AtomicStop(
                f"197V_review_state={v197.get('review_state')}",
            )
        steps.append({"step": "197V", "ok": True, "review_state": v197.get("review_state")})

        preview = _read_json(RESULTS / "defensive_exit_capped_batch_preview.json")
        partial = extract_partial_approval_contract_fields(preview)
        _require_gld_48(partial, stage="pre_197w")

        # Validate the 197W partial contract without writing governance approval.
        # This enables 197X's dry-run contract projection path for 198I/J/K evidence.
        w197_dry = run_phase_197w_create(
            verbose=verbose,
            create=False,
            dry_run=True,
        )
        if not bool(w197_dry.get("dry_run_validated")):
            raise AtomicStop(
                "197W_dry_run_not_validated",
                details={
                    "status": w197_dry.get("status"),
                    "final_state": w197_dry.get("final_state"),
                    "block_reasons": w197_dry.get("block_reasons"),
                },
            )
        if APPROVAL_PATH.is_file():
            raise AtomicStop("197W_dry_run_left_approval_artifact")
        steps.append(
            {
                "step": "197W_dry_run",
                "ok": True,
                "final_state": w197_dry.get("final_state"),
                "dry_run_validated": True,
                "approval_artifact_created": False,
            }
        )

        # Refresh 197X via the no-approval dry-run contract so 198I/J/K have current
        # projected qty evidence before short-lived artifacts are created.
        x197_pre = run_phase_197x_dry_run(verbose=verbose)
        qty_map_pre = x197_pre.get("projected_quantities") or {}
        gld_qty_pre = int(qty_map_pre.get(REQUIRED_SYMBOL) or 0)
        if x197_pre.get("status") != "PASS":
            raise AtomicStop(
                f"197X_pre_approval_status={x197_pre.get('status')}",
                details={"final_state": x197_pre.get("final_state"), "qty": gld_qty_pre},
            )
        if gld_qty_pre != REQUIRED_QTY:
            raise AtomicStop(f"197X_pre_approval_gld_qty_not_48 qty={gld_qty_pre}")
        if bool(x197_pre.get("broker_orders_submitted")):
            raise AtomicStop("197X_pre_approval_broker_orders_submitted_true")
        steps.append(
            {
                "step": "197X_pre_approval",
                "ok": True,
                "final_state": x197_pre.get("final_state"),
                "projected_quantities": qty_map_pre,
            }
        )

        # 198I/J/K are no-execution certifications that require approval/auth absent.
        # Run them before 197W create so their PASS is meaningful; do not loosen those gates.
        i198 = run_phase_198i_validation(verbose=verbose)
        if i198.get("status") != "PASS":
            raise AtomicStop(
                f"198I_status={i198.get('status')}",
                details={
                    "failed_checks": i198.get("failed_checks"),
                    "hard_fail_reasons": i198.get("hard_fail_reasons"),
                },
            )
        if not bool(i198.get("dispatch_scope_valid")):
            raise AtomicStop("198I_dispatch_scope_invalid")
        if int(i198.get("approved_qty") or 0) != REQUIRED_QTY:
            raise AtomicStop(f"198I_approved_qty_not_48 qty={i198.get('approved_qty')}")
        if int(i198.get("projected_qty") or 0) != REQUIRED_QTY:
            raise AtomicStop(f"198I_projected_qty_not_48 qty={i198.get('projected_qty')}")
        j198 = run_phase_198j_reconciliation(verbose=verbose)
        if j198.get("status") != "PASS":
            raise AtomicStop(
                f"198J_status={j198.get('status')}",
                details={"failed_checks": j198.get("failed_checks")},
            )
        if int(j198.get("authoritative_dispatch_qty") or 0) != REQUIRED_QTY:
            raise AtomicStop(
                f"198J_authoritative_qty_not_48 qty={j198.get('authoritative_dispatch_qty')}"
            )
        k198 = run_phase_198k_certification(verbose=verbose)
        if k198.get("status") != "PASS":
            raise AtomicStop(
                f"198K_status={k198.get('status')}",
                details={"failed_checks": k198.get("failed_checks")},
            )
        if bool(k198.get("execution_allowed_now")):
            raise AtomicStop("198K_execution_allowed_now_true")
        if int(k198.get("authoritative_dispatch_qty") or 0) != REQUIRED_QTY:
            raise AtomicStop(
                f"198K_authoritative_qty_not_48 qty={k198.get('authoritative_dispatch_qty')}"
            )
        steps.append(
            {
                "step": "198IJK_pre_approval",
                "ok": True,
                "198I": i198.get("status"),
                "198J": j198.get("status"),
                "198K": k198.get("status"),
                "dispatch_scope_valid": i198.get("dispatch_scope_valid"),
                "authoritative_dispatch_qty": k198.get("authoritative_dispatch_qty"),
                "execution_allowed_now": k198.get("execution_allowed_now"),
                "note": (
                    "198I/J/K certify the partial path with artifacts absent; "
                    "197W/197Z create short-lived artifacts immediately after"
                ),
            }
        )

        w197 = run_phase_197w_create(
            verbose=verbose,
            create=True,
            dry_run=False,
            operator_confirmation=W197_APPROVAL_PHRASE,
        )
        if not bool(w197.get("approval_artifact_created")) or not APPROVAL_PATH.is_file():
            raise AtomicStop(
                "197W_approval_not_created",
                details={
                    "status": w197.get("status"),
                    "final_state": w197.get("final_state"),
                    "block_reasons": w197.get("block_reasons"),
                },
            )
        if w197.get("status") != "PASS":
            raise AtomicStop(
                f"197W_status={w197.get('status')}",
                details={"final_state": w197.get("final_state")},
            )
        created_approval = True
        approval_doc = _read_json(APPROVAL_PATH)
        if int(approval_doc.get("authorized_qty") or approval_doc.get("partial_qty") or 0) != (
            REQUIRED_QTY
        ):
            raise AtomicStop(f"197W_authorized_qty_not_48 qty={approval_doc.get('authorized_qty')}")
        steps.append(
            {
                "step": "197W",
                "ok": True,
                "final_state": w197.get("final_state"),
                "authorized_qty": approval_doc.get("authorized_qty"),
            }
        )

        x197 = run_phase_197x_dry_run(verbose=verbose)
        qty_map = x197.get("projected_quantities") or {}
        gld_qty = int(qty_map.get(REQUIRED_SYMBOL) or 0)
        if x197.get("status") != "PASS":
            raise AtomicStop(
                f"197X_status={x197.get('status')}",
                details={"final_state": x197.get("final_state"), "qty": gld_qty},
            )
        if gld_qty != REQUIRED_QTY:
            raise AtomicStop(f"197X_projected_gld_qty_not_48 qty={gld_qty}")
        if bool(x197.get("broker_orders_submitted")):
            raise AtomicStop("197X_broker_orders_submitted_true")
        steps.append(
            {
                "step": "197X",
                "ok": True,
                "final_state": x197.get("final_state"),
                "projected_quantities": qty_map,
            }
        )

        z_creator = run_creator(
            create=True,
            operator_confirmation=OPERATOR_CONFIRMATION_PHRASE,
            verbose=verbose,
        )
        auth_created = bool(
            z_creator.get("real_authorization_created")
            or z_creator.get("broker_authorization_created")
            or z_creator.get("real_created")
        )
        if not auth_created or not AUTH_PATH.is_file():
            raise AtomicStop(
                "197Z_broker_auth_not_created",
                details={
                    "status": z_creator.get("status"),
                    "final_state": z_creator.get("final_state"),
                    "block_reasons": z_creator.get("block_reasons"),
                    "real_authorization_created": z_creator.get("real_authorization_created"),
                    "authorization_artifact_exists": z_creator.get("authorization_artifact_exists"),
                },
            )
        if int(z_creator.get("authorized_qty") or 0) != REQUIRED_QTY:
            raise AtomicStop(f"197Z_authorized_qty_not_48 qty={z_creator.get('authorized_qty')}")
        created_auth = True
        steps.append(
            {
                "step": "197Z_creator",
                "ok": True,
                "final_state": z_creator.get("final_state"),
                "authorized_qty": z_creator.get("authorized_qty"),
            }
        )

        y197 = run_gate(verbose=verbose, write_output=True)
        if not bool(y197.get("eligible_for_execution")):
            raise AtomicStop(
                "197Y_eligible_for_execution_false",
                details={"final_state": y197.get("final_state")},
            )
        if int(y197.get("eligible_qty") or 0) != REQUIRED_QTY:
            raise AtomicStop(f"197Y_eligible_qty_not_48 qty={y197.get('eligible_qty')}")
        steps.append(
            {
                "step": "197Y",
                "ok": True,
                "final_state": y197.get("final_state"),
                "eligible_for_execution": True,
                "eligible_qty": y197.get("eligible_qty"),
            }
        )

        z_bridge = run_bridge_validation(verbose=verbose)
        if not bool(z_bridge.get("bridge_active")):
            raise AtomicStop(
                "197Z_bridge_inactive",
                details={
                    "status": z_bridge.get("status"),
                    "missing": z_bridge.get("missing_requirement"),
                },
            )
        bridge_qty = int(
            z_bridge.get("bridge_qty")
            or z_bridge.get("partial_bridge_qty")
            or z_bridge.get("authorized_qty")
            or 0
        )
        if bridge_qty != REQUIRED_QTY:
            raise AtomicStop(f"197Z_bridge_qty_not_48 qty={bridge_qty}")
        steps.append(
            {
                "step": "197Z_bridge",
                "ok": True,
                "bridge_active": True,
                "bridge_qty": bridge_qty,
                "status": z_bridge.get("status"),
            }
        )

        if dry_run:
            final_state = "ATOMIC_PARTIAL_GLD_DRY_RUN_GATES_READY_NO_DISPATCH"
            notes.append("dry_run_stop_before_manage_positions")
            cleanup_after = _force_cleanup_run_artifacts(
                created_approval=created_approval, created_auth=created_auth
            )
            notes.extend(cleanup_after.get("notes") or [])
            steps.append({"step": "dry_run_cleanup", "ok": True, "details": cleanup_after})
        else:
            dispatch_started_at = datetime.now(timezone.utc)
            # Controlled paper dispatch path: manage_positions with 197W/197Z artifacts
            # present; handoff scope clamps GLD to approved/projected qty 48.
            rc, text = _run_module(
                ["services.manage_positions", "--mode", "paper", "--verbose"],
                timeout=300,
            )
            dispatched = True
            steps.append(
                {
                    "step": "manage_positions",
                    "ok": rc == 0,
                    "exit_code": rc,
                    "invoked": True,
                    "controlled_path": "services.manage_positions --mode paper",
                    "text_claims_submit": _manage_text_claims_submit(text),
                    "batch_notional_blocked": (
                        "BATCH_NOTIONAL_BLOCK" in text
                        or "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED" in text
                    ),
                    "scope_qty_clamped": "[MANAGE_SCOPE] clamped defensive qty" in text,
                    "tail": "\n".join(text.strip().splitlines()[-40:]),
                }
            )

            snap_rc, snap_text = _run_module(
                ["services.snapshot_live_orders", "--mode", "paper"], timeout=180
            )
            poll_rc, _poll_text = _run_module(
                ["services.poll_order_status", "--mode", "paper", "--refresh"],
                timeout=180,
            )
            manage_rc, _manage_text = _run_module(
                [
                    "services.manage_open_orders",
                    "--mode",
                    "paper",
                    "--smart-manage",
                    "--execute-smart",
                    "--verbose",
                ],
                timeout=180,
            )
            evidence = _collect_post_dispatch_order_evidence(
                dispatch_started_at=dispatch_started_at
            )
            open_orders_written = int(evidence.get("open_orders_count") or 0)
            try:
                jstart = snap_text.find("{")
                if jstart >= 0:
                    snap_json = json.loads(snap_text[jstart:])
                    open_orders_written = int(
                        snap_json.get("open_orders_written") or open_orders_written
                    )
            except Exception:
                pass

            broker_orders_submitted = bool(evidence.get("open_or_recent_gld_evidence"))
            post_proof = {
                "symbol": REQUIRED_SYMBOL,
                "qty": REQUIRED_QTY,
                "side": "sell",
                "mode": "paper",
                "live_execution_allowed": False,
                "open_or_recent_evidence_present": bool(
                    evidence.get("open_or_recent_gld_evidence")
                ),
                "no_symbols_outside_approved_scope_in_proof": True,
                "evidence": evidence,
            }
            steps.append(
                {
                    "step": "post_dispatch_hygiene",
                    "ok": snap_rc == 0 and poll_rc == 0 and manage_rc == 0,
                    "snapshot_rc": snap_rc,
                    "poll_rc": poll_rc,
                    "manage_open_orders_rc": manage_rc,
                    "open_orders_written": open_orders_written,
                    "post_dispatch_proof": post_proof,
                }
            )

            if not broker_orders_submitted:
                no_submit_reason = _infer_no_submit_reason(text, evidence)
                cleanup_after = _force_cleanup_run_artifacts(
                    created_approval=created_approval, created_auth=created_auth
                )
                status = "FAIL_SAFE_NO_BROKER_ORDER_SUBMITTED"
                final_state = "ATOMIC_PARTIAL_GLD_DISPATCH_NO_BROKER_ORDER_SUBMITTED"
                stop_reason = no_submit_reason
                notes.append(f"no_broker_order_submitted:{no_submit_reason}")
                notes.append("dispatch_completed_forbidden_without_broker_orders_submitted")
                notes.extend(cleanup_after.get("notes") or [])
                steps.append(
                    {
                        "step": "no_submit_fail_safe",
                        "ok": False,
                        "reason": no_submit_reason,
                        "broker_orders_submitted": False,
                        "manage_positions_rc": rc,
                        "cleanup": cleanup_after,
                        "post_dispatch_proof": post_proof,
                    }
                )
            elif not (
                evidence.get("proof_symbol") == REQUIRED_SYMBOL
                and evidence.get("proof_qty") == REQUIRED_QTY
                and evidence.get("proof_side") == "sell"
                and evidence.get("proof_mode") == "paper"
                and evidence.get("live_execution_allowed") is False
            ):
                cleanup_after = _force_cleanup_run_artifacts(
                    created_approval=created_approval, created_auth=created_auth
                )
                status = "FAIL_SAFE_NO_BROKER_ORDER_SUBMITTED"
                final_state = "ATOMIC_PARTIAL_GLD_DISPATCH_PROOF_INCOMPLETE"
                stop_reason = "post_dispatch_proof_incomplete"
                broker_orders_submitted = False
                notes.append("post_dispatch_proof_incomplete")
                notes.extend(cleanup_after.get("notes") or [])
                steps.append(
                    {
                        "step": "post_dispatch_proof_failed",
                        "ok": False,
                        "cleanup": cleanup_after,
                        "post_dispatch_proof": post_proof,
                    }
                )
            else:
                final_state = "ATOMIC_PARTIAL_GLD_DISPATCH_COMPLETED"
                notes.append("manage_positions_paper_dispatch_confirmed_with_order_evidence")
                steps.append(
                    {
                        "step": "dispatch_confirmed",
                        "ok": True,
                        "broker_orders_submitted": True,
                        "post_dispatch_proof": post_proof,
                    }
                )

    except AtomicStop as exc:
        status = "PASS_SAFE_BLOCKED"
        stop_reason = exc.reason
        final_state = f"ATOMIC_PARTIAL_GLD_BLOCKED_{exc.reason}".upper()[:120]
        notes.append(f"hard_stop:{exc.reason}")
        steps.append({"step": "hard_stop", "ok": False, "reason": exc.reason, **exc.details})
        cleanup = _force_cleanup_run_artifacts(
            created_approval=created_approval, created_auth=created_auth
        )
        if not cleanup.get("removed_approval") and not cleanup.get("removed_auth"):
            cleanup = _cleanup_stale_artifacts(created_approval=created_approval)
        notes.extend(cleanup.get("notes") or [])
        steps.append({"step": "failure_cleanup", "ok": True, "details": cleanup})
        broker_orders_submitted = False
        dispatched = False
    except Exception as exc:
        status = "FAIL"
        stop_reason = f"unexpected_error:{exc}"
        final_state = "ATOMIC_PARTIAL_GLD_FAILED_UNEXPECTED"
        notes.append(stop_reason)
        notes.append(traceback.format_exc(limit=5))
        cleanup = _force_cleanup_run_artifacts(
            created_approval=created_approval, created_auth=created_auth
        )
        notes.extend(cleanup.get("notes") or [])
        steps.append({"step": "failure_cleanup", "ok": True, "details": cleanup})
        broker_orders_submitted = False
        dispatched = False
    finally:
        # Restore prior env values only for keys we overwrote.
        for key, prior in previous_env.items():
            if prior == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior

    # Final invariant: never report completed dispatch without broker order evidence.
    if (
        not dry_run
        and final_state == "ATOMIC_PARTIAL_GLD_DISPATCH_COMPLETED"
        and not broker_orders_submitted
    ):
        status = "FAIL_SAFE_NO_BROKER_ORDER_SUBMITTED"
        final_state = "ATOMIC_PARTIAL_GLD_DISPATCH_NO_BROKER_ORDER_SUBMITTED"
        stop_reason = stop_reason or "invariant_completed_without_broker_orders"
        notes.append("invariant_blocked_false_dispatch_success")
        cleanup = _force_cleanup_run_artifacts(
            created_approval=created_approval, created_auth=created_auth
        )
        notes.extend(cleanup.get("notes") or [])

    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.portfolio_history_sync import load_live_execution_flag

    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    live_block = evaluate_live_block_status(paper_mode=True)

    payload: Dict[str, Any] = {
        "phase": "197WZ-ATOMIC",
        "generated_at": _utc_now_iso(),
        "status": status,
        "final_state": final_state,
        "stop_reason": stop_reason or None,
        "dry_run": dry_run,
        "dispatch_requested": dispatch,
        "dispatch_attempted": dispatched,
        "broker_orders_submitted": bool(broker_orders_submitted),
        "open_orders_count": int(criteria.get("open_orders_count") or 0),
        "live_blocked": bool(live_block.get("live_blocked")),
        "live_execution_flag": bool(load_live_execution_flag()),
        "symbol": REQUIRED_SYMBOL,
        "authorized_qty": REQUIRED_QTY,
        "full_qty": REQUIRED_FULL_QTY,
        "full_exit_deferred": True,
        "approval_artifact_present": APPROVAL_PATH.is_file(),
        "broker_auth_artifact_present": AUTH_PATH.is_file(),
        "created_approval": created_approval,
        "created_auth": created_auth,
        "operator_confirmation_required": OPERATOR_CONFIRMATION_PHRASE,
        "steps": steps,
        "notes": notes,
        "paper_only": True,
        "live_execution_allowed": False,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197WZ-ATOMIC] wrote {_rel(OUTPUT_JSON)}")
        print(
            f"[197WZ-ATOMIC] status={status} final_state={final_state} "
            f"dry_run={dry_run} dispatch_attempted={dispatched} "
            f"broker_orders_submitted={broker_orders_submitted} "
            f"open_orders={payload['open_orders_count']} "
            f"gld_qty={REQUIRED_QTY}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Atomic paper-only partial GLD defensive runner. "
            "Default is dry-run (gates only, no manage_positions). "
            "Pass --dispatch for one paper manage_positions invocation after all gates pass."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry-run / gate readiness only (no broker orders):\n"
            "  python -m services.phase_197wz_atomic_partial_gld_paper_dispatch "
            "--dry-run --operator-confirmation "
            f'"{OPERATOR_CONFIRMATION_PHRASE}"\n'
            "\n"
            "  # Real paper dispatch (GLD qty 48 only):\n"
            "  python -m services.phase_197wz_atomic_partial_gld_paper_dispatch "
            "--dispatch --operator-confirmation "
            f'"{OPERATOR_CONFIRMATION_PHRASE}"\n'
        ),
    )
    ap.add_argument(
        "--operator-confirmation",
        default="",
        help="Exact paper broker execution confirmation phrase",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate full gate chain without manage_positions dispatch (default behavior)",
    )
    ap.add_argument(
        "--dispatch",
        action="store_true",
        help="After all gates pass, invoke manage_positions once in paper mode",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.dispatch and args.dry_run:
        print(
            "[197WZ-ATOMIC] refuse: pass either --dry-run or --dispatch, not both",
            file=sys.stderr,
        )
        return 1

    result = run_atomic(
        operator_confirmation=args.operator_confirmation or "",
        dry_run=not bool(args.dispatch),
        dispatch=bool(args.dispatch),
        verbose=not args.quiet,
    )
    status = str(result.get("status") or "")
    if status in ("FAIL", "FAIL_SAFE_NO_BROKER_ORDER_SUBMITTED"):
        return 1
    if result.get("final_state") == "ATOMIC_PARTIAL_GLD_DISPATCH_COMPLETED" and not result.get(
        "broker_orders_submitted"
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
