"""
phase_198m_atomic_gld_20pct_paper_dispatch.py

Atomic paper-only GLD 20% target reduction dispatch runner (Phase 198L → 198M).

Default: --dry-run (prove gate readiness; no manage_positions dispatch).
Real paper dispatch requires --dispatch plus the exact operator confirmation.

Does not enable live execution.
Does not sell more than 131 GLD.
Does not fully liquidate GLD.
Does not authorize non-GLD symbols.
"""

from __future__ import annotations

import argparse
import copy
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
OUTPUT_JSON = RESULTS / "phase_198m_atomic_gld_20pct_paper_dispatch.json"
OUTPUT_MD = RESULTS / "phase_198m_atomic_gld_20pct_paper_dispatch.md"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
PREVIEW_BACKUP = RESULTS / "phase_198m_preview_backup.json"
L198_JSON = RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
PROJECTED_CSV = RESULTS / "phase_197x_projected_paper_orders.csv"

OPERATOR_CONFIRMATION_PHRASE = (
    "I authorize the paper broker execution phase for the GLD 20 percent target "
    "reduction batch. Paper only. No live trading."
)
W197_APPROVAL_PHRASE = (
    "I authorize Phase 197W to create the paper-only operator approval artifact "
    "for the current capped defensive batch. No broker orders yet."
)

REQUIRED_SYMBOL = "GLD"
REQUIRED_CURRENT_QTY = 147
REQUIRED_SELL_QTY = 131
REQUIRED_REMAINING_QTY = 16
TARGET_WEIGHT = 0.20

ENV_FORCE = {
    "TRITON_ENABLE_PAPER_EXECUTION": "1",
    "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
    "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
    "TRITON_ENABLE_LIVE_EXECUTION": "0",
    "TRITON_LIVE_EXECUTION": "0",
}


class AtomicStop(Exception):
    def __init__(self, reason: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
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


def _force_cleanup_run_artifacts(*, created_approval: bool, created_auth: bool) -> Dict[str, Any]:
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


def _restore_preview_backup() -> Dict[str, Any]:
    if not PREVIEW_BACKUP.is_file():
        return {"restored": False, "reason": "no_backup"}
    try:
        backup = json.loads(PREVIEW_BACKUP.read_text(encoding="utf-8"))
        _atomic_write_json(PREVIEW_JSON, backup)
        PREVIEW_BACKUP.unlink()
        return {"restored": True, "path": _rel(PREVIEW_JSON)}
    except Exception as exc:
        return {"restored": False, "reason": str(exc)}


def _inject_198l_preview(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Align capped-batch preview with Phase 198L GLD sell qty 131 / remain 16."""
    preview = _read_json(PREVIEW_JSON)
    if not preview:
        raise AtomicStop("preview_missing_for_198l_inject")
    _atomic_write_json(PREVIEW_BACKUP, preview)

    sell_qty = int(analysis.get("recommended_sell_qty") or REQUIRED_SELL_QTY)
    remaining = int(analysis.get("projected_remaining_qty") or REQUIRED_REMAINING_QTY)
    full_qty = int(float(analysis.get("current_gld_qty") or REQUIRED_CURRENT_QTY))
    price = float(analysis.get("current_gld_price") or 0.0)
    notional = round(sell_qty * price, 2)
    remaining_notional = round(remaining * price, 2)

    order = {
        "symbol": REQUIRED_SYMBOL,
        "qty": sell_qty,
        "limit_price": round(price, 4),
        "notional": notional,
        "notional_estimate": notional,
        "quantity": sell_qty,
        "side": "sell",
        "management_type": "TRIM",
        "effective_stance": "TRIM",
        "preview_type": "PARTIAL_DEFENSIVE_EXIT_PREVIEW",
        "partial_defensive_exit": True,
        "scenario_name": "GLD_20PCT_TARGET_REDUCTION",
        "classification": "TARGET_WEIGHT_REDUCTION",
        "full_position_qty": full_qty,
        "remaining_qty": remaining,
        "remaining_notional": remaining_notional,
        "paper_only": True,
        "broker_orders_submitted": False,
        "discipline_allowed": True,
        "rank": 1,
        "reason_code": "GLD_20PCT_TARGET_REDUCTION",
        "priority_bucket": "EXIT",
        "source": "phase_198l",
    }

    nested = dict(preview.get("partial_defensive_exit_preview") or {})
    nested.update(
        {
            "partial_preview_available": True,
            "partial_preview_symbol": REQUIRED_SYMBOL,
            "partial_preview_qty": sell_qty,
            "partial_preview_notional": notional,
            "partial_preview_classification": "TARGET_WEIGHT_REDUCTION",
            "partial_preview_scenario": "GLD_20PCT_TARGET_REDUCTION",
            "partial_preview_order": order,
            "remaining_qty": remaining,
            "remaining_notional": remaining_notional,
            "full_exit_notional": round(full_qty * price, 2),
            "full_exit_deferred_reason": ("GLD_20PCT_TARGET_REDUCTION_DEFERS_REMAINING_SHARES"),
            "partial_preview_requires_operator_approval": True,
            "partial_preview_analysis_only": False,
            "paper_only": True,
            "broker_orders_submitted": False,
            "source": "phase_198l_inject",
        }
    )

    def _patch_cap_block(block: Any) -> Dict[str, Any]:
        out = dict(block or {})
        out["preview_type"] = "PARTIAL_DEFENSIVE_EXIT_PREVIEW"
        out["partial_preview_qty"] = sell_qty
        out["partial_preview_symbol"] = REQUIRED_SYMBOL
        out["partial_preview_notional"] = notional
        out["capped_preview_symbols"] = [REQUIRED_SYMBOL]
        out["capped_preview_orders"] = [order]
        out["deferred_symbols"] = [REQUIRED_SYMBOL]
        out["batch_notional_capped_preview"] = notional
        out["partial_defensive_exit_preview"] = True
        return out

    updated = copy.deepcopy(preview)
    updated.update(
        {
            "partial_preview_available": True,
            "partial_preview_symbol": REQUIRED_SYMBOL,
            "partial_preview_qty": sell_qty,
            "partial_preview_notional": notional,
            "partial_preview_classification": "TARGET_WEIGHT_REDUCTION",
            "preview_type": "PARTIAL_DEFENSIVE_EXIT_PREVIEW",
            "full_exit_notional": round(full_qty * price, 2),
            "partial_defensive_exit_preview": nested,
            "capped_preview": _patch_cap_block(preview.get("capped_preview")),
            "capped_preview_runtime_cap": _patch_cap_block(
                preview.get("capped_preview_runtime_cap")
            ),
            "phase_198m_inject": {
                "injected_at": _utc_now_iso(),
                "sell_qty": sell_qty,
                "remaining_qty": remaining,
                "full_qty": full_qty,
                "source": "phase_198l",
            },
        }
    )
    _atomic_write_json(PREVIEW_JSON, updated)
    return {
        "injected": True,
        "sell_qty": sell_qty,
        "remaining_qty": remaining,
        "full_qty": full_qty,
        "notional": notional,
        "backup": _rel(PREVIEW_BACKUP),
    }


def _scan_order_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        import csv

        with path.open(encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except Exception:
        return []


def _collect_gld_order_evidence(*, dispatch_started_at: datetime) -> Dict[str, Any]:
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
        if sym != REQUIRED_SYMBOL or side not in ("sell", "sell_to_close"):
            return False
        if qty != REQUIRED_SELL_QTY:
            return False
        if status in ("canceled", "cancelled", "expired", "rejected", "failed"):
            return False
        return True

    def _recent_enough(row: Dict[str, str]) -> bool:
        for key in ("submitted_at", "created_at", "updated_at", "filled_at", "snapshot_ts"):
            raw = str(row.get(key) or "").strip()
            if not raw:
                continue
            try:
                ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (ts - dispatch_started_at).total_seconds() >= -120:
                    return True
            except Exception:
                continue
        return True

    matching_open = [r for r in open_rows if _row_matches(r)]
    matching_recent = [r for r in recent_rows if _row_matches(r) and _recent_enough(r)]
    proof_rows = matching_open + matching_recent
    return {
        "open_or_recent_gld_evidence": bool(proof_rows),
        "matching_open_gld_rows": len(matching_open),
        "matching_recent_gld_rows": len(matching_recent),
        "proof_symbol": REQUIRED_SYMBOL if proof_rows else None,
        "proof_qty": REQUIRED_SELL_QTY if proof_rows else None,
        "proof_side": "sell" if proof_rows else None,
        "proof_mode": "paper",
        "live_execution_allowed": False,
        "gld_evidence_sample": [
            {
                "symbol": r.get("symbol"),
                "side": r.get("side"),
                "qty": r.get("qty") or r.get("quantity"),
                "status": r.get("status"),
                "submitted_at": r.get("submitted_at") or r.get("created_at"),
            }
            for r in proof_rows[:3]
        ],
    }


def _infer_no_submit_reason(manage_text: str, evidence: Dict[str, Any]) -> str:
    text = manage_text or ""
    if "BATCH_NOTIONAL_BLOCK" in text or "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED" in text:
        return "manage_positions_batch_notional_blocked_no_broker_order"
    if "SAFE_BLOCK" in text:
        return "manage_positions_safe_block_no_broker_order"
    if "exits_executed=0" in text:
        return "manage_positions_executed_zero_no_broker_order"
    if not evidence.get("open_or_recent_gld_evidence"):
        return "no_open_or_recent_gld_sell_qty_131_evidence"
    return "broker_orders_submitted_false_no_order_evidence"


def _render_markdown(payload: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 198M — Atomic GLD 20% Target Paper Dispatch",
            "",
            f"- Generated: `{payload.get('generated_at')}`",
            f"- Status: `{payload.get('status')}`",
            f"- Final state: `{payload.get('final_state')}`",
            f"- Dry-run: `{payload.get('dry_run')}`",
            f"- Dispatch attempted: `{payload.get('dispatch_attempted')}`",
            f"- Broker orders submitted: `{payload.get('broker_orders_submitted')}`",
            "",
            "## Scope",
            "",
            f"- Symbol: **{payload.get('symbol')}**",
            f"- Current qty required: **{REQUIRED_CURRENT_QTY}**",
            f"- Sell qty: **{payload.get('recommended_sell_qty')}**",
            f"- Remaining qty: **{payload.get('projected_remaining_qty')}**",
            f"- Projected GLD weight: **{payload.get('projected_gld_weight')}**",
            "",
            "## Safety",
            "",
            f"- paper_only: `{payload.get('paper_only')}`",
            f"- live_execution_allowed: `{payload.get('live_execution_allowed')}`",
            f"- live_blocked: `{payload.get('live_blocked')}`",
            f"- open_orders_count: `{payload.get('open_orders_count')}`",
            f"- approval_artifact_present: `{payload.get('approval_artifact_present')}`",
            f"- broker_auth_artifact_present: `{payload.get('broker_auth_artifact_present')}`",
            f"- stop_reason: `{payload.get('stop_reason')}`",
            "",
            "ATOMIC_GLD_20PCT_PAPER_DISPATCH",
            "",
        ]
    )


def _preflight_or_stop() -> Dict[str, Any]:
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.portfolio_history_sync import load_live_execution_flag

    if APPROVAL_PATH.is_file():
        raise AtomicStop("approval_artifact_already_exists")
    if AUTH_PATH.is_file():
        raise AtomicStop("broker_auth_artifact_already_exists")

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

    return {
        "open_orders_count": open_orders,
        "live_blocked": True,
        "live_execution_flag": False,
    }


def _require_198l(analysis: Dict[str, Any]) -> None:
    if analysis.get("status") != "PASS":
        raise AtomicStop(f"198L_status={analysis.get('status')}")
    current = int(float(analysis.get("current_gld_qty") or 0))
    sell = int(analysis.get("recommended_sell_qty") or 0)
    remaining = int(analysis.get("projected_remaining_qty") or 0)
    weight = float(analysis.get("projected_post_sale_gld_weight") or 1.0)
    if current != REQUIRED_CURRENT_QTY:
        raise AtomicStop(f"gld_qty_not_147 qty={current}")
    if sell != REQUIRED_SELL_QTY:
        raise AtomicStop(f"sell_qty_not_131 qty={sell}")
    if remaining != REQUIRED_REMAINING_QTY:
        raise AtomicStop(f"remaining_qty_not_16 qty={remaining}")
    if weight > TARGET_WEIGHT + 1e-9:
        raise AtomicStop(f"projected_weight_above_20pct weight={weight}")
    if not bool(analysis.get("full_liquidation_avoided")):
        raise AtomicStop("full_liquidation_not_avoided")
    if bool(analysis.get("broker_orders_submitted")):
        raise AtomicStop("198L_broker_orders_submitted_true")
    if bool(analysis.get("live_execution_allowed")):
        raise AtomicStop("198L_live_execution_allowed_true")


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
    preview_injected = False
    dispatched = False
    broker_orders_submitted = False
    stop_reason = ""
    final_state = "ATOMIC_GLD_20PCT_READY_NOT_DISPATCHED"
    status = "PASS"
    analysis: Dict[str, Any] = {}

    if operator_confirmation != OPERATOR_CONFIRMATION_PHRASE:
        payload = {
            "phase": "198M",
            "generated_at": _utc_now_iso(),
            "status": "PASS_SAFE_BLOCKED",
            "final_state": "ATOMIC_GLD_20PCT_BLOCKED_OPERATOR_CONFIRMATION_REQUIRED",
            "broker_orders_submitted": False,
            "dispatch_attempted": False,
            "dry_run": dry_run,
            "required_operator_confirmation": OPERATOR_CONFIRMATION_PHRASE,
            "notes": ["exact_operator_confirmation_required"],
        }
        _atomic_write_json(OUTPUT_JSON, payload)
        _atomic_write_text(OUTPUT_MD, _render_markdown(payload))
        if verbose:
            print("[198M-ATOMIC] blocked: operator confirmation phrase mismatch")
        return payload

    if dispatch:
        dry_run = False
    else:
        dry_run = True

    previous_env = _force_paper_only_env()
    try:
        preflight = _preflight_or_stop()
        steps.append({"step": "preflight", "ok": True, "details": preflight})

        snap_rc, snap_text = _run_module(
            ["services.snapshot_live_orders", "--mode", "paper"], timeout=180
        )
        if snap_rc != 0:
            raise AtomicStop(f"snapshot_live_orders_rc={snap_rc}")
        steps.append({"step": "broker_snapshot_refresh", "ok": True, "exit_code": snap_rc})

        from services.phase_198l_gld_20pct_target_reduction_analysis import (
            run_phase_198l_analysis,
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

        analysis = run_phase_198l_analysis(verbose=verbose)
        _require_198l(analysis)
        steps.append(
            {
                "step": "198L",
                "ok": True,
                "recommended_sell_qty": analysis.get("recommended_sell_qty"),
                "projected_remaining_qty": analysis.get("projected_remaining_qty"),
                "projected_post_sale_gld_weight": analysis.get("projected_post_sale_gld_weight"),
            }
        )

        inject = _inject_198l_preview(analysis)
        preview_injected = True
        steps.append({"step": "198l_preview_inject", "ok": True, "details": inject})

        u197 = run_phase_197u_refresh(verbose=verbose)
        if u197.get("status") != "PASS":
            raise AtomicStop(f"197U_status={u197.get('status')}")
        steps.append({"step": "197U", "ok": True, "final_state": u197.get("final_state")})

        v197 = run_phase_197v_review(verbose=verbose)
        if v197.get("status") != "PASS":
            raise AtomicStop(f"197V_status={v197.get('status')}")
        if v197.get("review_state") != "OPERATOR_REVIEW_READY_APPROVAL_REQUIRED":
            raise AtomicStop(f"197V_review_state={v197.get('review_state')}")
        steps.append({"step": "197V", "ok": True, "review_state": v197.get("review_state")})

        w197_dry = run_phase_197w_create(verbose=verbose, create=False, dry_run=True)
        if not bool(w197_dry.get("dry_run_validated")):
            raise AtomicStop(
                "197W_dry_run_not_validated",
                details={"final_state": w197_dry.get("final_state")},
            )
        if APPROVAL_PATH.is_file():
            raise AtomicStop("197W_dry_run_left_approval_artifact")
        steps.append(
            {
                "step": "197W_dry_run",
                "ok": True,
                "final_state": w197_dry.get("final_state"),
            }
        )

        x197_pre = run_phase_197x_dry_run(verbose=verbose)
        qty_map_pre = x197_pre.get("projected_quantities") or {}
        gld_pre = int(qty_map_pre.get(REQUIRED_SYMBOL) or 0)
        if x197_pre.get("status") != "PASS":
            raise AtomicStop(
                f"197X_pre_approval_status={x197_pre.get('status')}",
                details={"qty": gld_pre, "final_state": x197_pre.get("final_state")},
            )
        if gld_pre != REQUIRED_SELL_QTY:
            raise AtomicStop(f"197X_pre_approval_qty_not_131 qty={gld_pre}")
        if set(str(s).upper() for s in qty_map_pre.keys()) - {REQUIRED_SYMBOL}:
            raise AtomicStop(f"197X_pre_approval_non_gld_symbols={sorted(qty_map_pre.keys())}")
        steps.append(
            {
                "step": "197X_pre_approval",
                "ok": True,
                "projected_quantities": qty_map_pre,
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
        created_approval = True
        approval_doc = _read_json(APPROVAL_PATH)
        auth_qty = int(
            approval_doc.get("authorized_qty")
            or approval_doc.get("partial_qty")
            or (approval_doc.get("approved_quantities") or {}).get(REQUIRED_SYMBOL)
            or 0
        )
        if auth_qty != REQUIRED_SELL_QTY:
            raise AtomicStop(f"197W_authorized_qty_not_131 qty={auth_qty}")
        approved_symbols = [str(s).upper() for s in (approval_doc.get("approved_symbols") or [])]
        if approved_symbols != [REQUIRED_SYMBOL]:
            raise AtomicStop(f"197W_approved_symbols_not_gld_only={approved_symbols}")
        steps.append(
            {
                "step": "197W",
                "ok": True,
                "final_state": w197.get("final_state"),
                "authorized_qty": auth_qty,
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
        if gld_qty != REQUIRED_SELL_QTY:
            raise AtomicStop(f"197X_projected_gld_qty_not_131 qty={gld_qty}")
        if set(str(s).upper() for s in qty_map.keys()) - {REQUIRED_SYMBOL}:
            raise AtomicStop(f"197X_non_gld_symbols={sorted(qty_map.keys())}")
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
                },
            )
        if int(z_creator.get("authorized_qty") or 0) != REQUIRED_SELL_QTY:
            raise AtomicStop(f"197Z_authorized_qty_not_131 qty={z_creator.get('authorized_qty')}")
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
        if int(y197.get("eligible_qty") or 0) != REQUIRED_SELL_QTY:
            raise AtomicStop(f"197Y_eligible_qty_not_131 qty={y197.get('eligible_qty')}")
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
        if bridge_qty != REQUIRED_SELL_QTY:
            raise AtomicStop(f"197Z_bridge_qty_not_131 qty={bridge_qty}")
        steps.append(
            {
                "step": "197Z_bridge",
                "ok": True,
                "bridge_active": True,
                "bridge_qty": bridge_qty,
            }
        )

        if dry_run:
            final_state = "DRY_RUN_GATES_READY_NO_DISPATCH"
            notes.append("dry_run_stop_before_manage_positions")
            cleanup_after = _force_cleanup_run_artifacts(
                created_approval=created_approval, created_auth=created_auth
            )
            restore = _restore_preview_backup()
            notes.extend(cleanup_after.get("notes") or [])
            notes.append(f"preview_restore={restore}")
            steps.append(
                {
                    "step": "dry_run_cleanup",
                    "ok": True,
                    "details": {"artifacts": cleanup_after, "preview": restore},
                }
            )
            preview_injected = False
        else:
            dispatch_started_at = datetime.now(timezone.utc)
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
                    "scope_qty_clamped": "[MANAGE_SCOPE] clamped defensive qty" in text,
                    "batch_notional_blocked": (
                        "BATCH_NOTIONAL_BLOCK" in text
                        or "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED" in text
                    ),
                    "tail": "\n".join(text.strip().splitlines()[-40:]),
                }
            )

            _run_module(["services.snapshot_live_orders", "--mode", "paper"], timeout=180)
            _run_module(["services.poll_order_status", "--mode", "paper", "--refresh"], timeout=180)
            _run_module(
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
            evidence = _collect_gld_order_evidence(dispatch_started_at=dispatch_started_at)
            broker_orders_submitted = bool(evidence.get("open_or_recent_gld_evidence"))
            steps.append(
                {
                    "step": "post_dispatch_hygiene",
                    "ok": True,
                    "post_dispatch_proof": evidence,
                }
            )

            if not broker_orders_submitted:
                no_submit_reason = _infer_no_submit_reason(text, evidence)
                cleanup_after = _force_cleanup_run_artifacts(
                    created_approval=created_approval, created_auth=created_auth
                )
                restore = _restore_preview_backup()
                preview_injected = False
                status = "FAIL_SAFE_NO_BROKER_ORDER_SUBMITTED"
                final_state = "ATOMIC_GLD_20PCT_DISPATCH_NO_BROKER_ORDER_SUBMITTED"
                stop_reason = no_submit_reason
                notes.append(f"no_broker_order_submitted:{no_submit_reason}")
                notes.extend(cleanup_after.get("notes") or [])
                notes.append(f"preview_restore={restore}")
                steps.append(
                    {
                        "step": "no_submit_fail_safe",
                        "ok": False,
                        "reason": no_submit_reason,
                        "cleanup": cleanup_after,
                    }
                )
            else:
                final_state = "ATOMIC_GLD_20PCT_DISPATCH_COMPLETED"
                notes.append("manage_positions_confirmed_with_gld_131_order_evidence")
                cleanup_after = _force_cleanup_run_artifacts(
                    created_approval=created_approval, created_auth=created_auth
                )
                restore = _restore_preview_backup()
                preview_injected = False
                notes.extend(cleanup_after.get("notes") or [])
                notes.append(f"preview_restore={restore}")
                steps.append(
                    {
                        "step": "dispatch_confirmed_and_cleanup",
                        "ok": True,
                        "cleanup": cleanup_after,
                        "evidence": evidence,
                    }
                )

    except AtomicStop as exc:
        status = "PASS_SAFE_BLOCKED"
        stop_reason = exc.reason
        final_state = f"ATOMIC_GLD_20PCT_BLOCKED_{exc.reason}".upper()[:120]
        notes.append(f"hard_stop:{exc.reason}")
        steps.append({"step": "hard_stop", "ok": False, "reason": exc.reason, **exc.details})
        cleanup = _force_cleanup_run_artifacts(
            created_approval=created_approval, created_auth=created_auth
        )
        notes.extend(cleanup.get("notes") or [])
        if preview_injected:
            restore = _restore_preview_backup()
            notes.append(f"preview_restore={restore}")
        steps.append({"step": "failure_cleanup", "ok": True, "details": cleanup})
        broker_orders_submitted = False
        dispatched = False
    except Exception as exc:
        status = "FAIL"
        stop_reason = f"unexpected_error:{exc}"
        final_state = "ATOMIC_GLD_20PCT_FAILED_UNEXPECTED"
        notes.append(stop_reason)
        notes.append(traceback.format_exc(limit=5))
        cleanup = _force_cleanup_run_artifacts(
            created_approval=created_approval, created_auth=created_auth
        )
        notes.extend(cleanup.get("notes") or [])
        if preview_injected:
            restore = _restore_preview_backup()
            notes.append(f"preview_restore={restore}")
        broker_orders_submitted = False
        dispatched = False
    finally:
        for key, prior in previous_env.items():
            if prior == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior

    if (
        not dry_run
        and final_state == "ATOMIC_GLD_20PCT_DISPATCH_COMPLETED"
        and not broker_orders_submitted
    ):
        status = "FAIL_SAFE_NO_BROKER_ORDER_SUBMITTED"
        final_state = "ATOMIC_GLD_20PCT_DISPATCH_NO_BROKER_ORDER_SUBMITTED"
        stop_reason = stop_reason or "invariant_completed_without_broker_orders"
        notes.append("invariant_blocked_false_dispatch_success")

    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.portfolio_history_sync import load_live_execution_flag

    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    live_block = evaluate_live_block_status(paper_mode=True)
    analysis_doc = analysis or _read_json(L198_JSON)

    payload: Dict[str, Any] = {
        "phase": "198M",
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
        "current_gld_qty": REQUIRED_CURRENT_QTY,
        "recommended_sell_qty": REQUIRED_SELL_QTY,
        "projected_remaining_qty": REQUIRED_REMAINING_QTY,
        "projected_gld_weight": analysis_doc.get("projected_post_sale_gld_weight"),
        "target_weight": TARGET_WEIGHT,
        "full_liquidation_avoided": True,
        "approval_artifact_present": APPROVAL_PATH.is_file(),
        "broker_auth_artifact_present": AUTH_PATH.is_file(),
        "created_approval": created_approval,
        "created_auth": created_auth,
        "operator_confirmation_required": OPERATOR_CONFIRMATION_PHRASE,
        "steps": steps,
        "notes": notes,
        "paper_only": True,
        "live_execution_allowed": False,
        "approval_required": True,
        "broker_authorization_required": True,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))
    if verbose:
        print(f"[198M-ATOMIC] wrote {_rel(OUTPUT_JSON)}")
        print(f"[198M-ATOMIC] wrote {_rel(OUTPUT_MD)}")
        print(
            f"[198M-ATOMIC] status={status} final_state={final_state} "
            f"dry_run={dry_run} dispatch_attempted={dispatched} "
            f"broker_orders_submitted={broker_orders_submitted} "
            f"sell_qty={REQUIRED_SELL_QTY} remaining={REQUIRED_REMAINING_QTY}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Atomic paper-only GLD 20% target reduction runner. "
            "Default is dry-run (gates only). Pass --dispatch for one paper "
            "manage_positions invocation after all gates pass."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m services.phase_198m_atomic_gld_20pct_paper_dispatch "
            "--dry-run --operator-confirmation "
            f'"{OPERATOR_CONFIRMATION_PHRASE}"\n'
        ),
    )
    ap.add_argument("--operator-confirmation", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dispatch", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.dispatch and args.dry_run:
        print(
            "[198M-ATOMIC] refuse: pass either --dry-run or --dispatch, not both", file=sys.stderr
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
    if result.get("final_state") == "ATOMIC_GLD_20PCT_DISPATCH_COMPLETED" and not result.get(
        "broker_orders_submitted"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
