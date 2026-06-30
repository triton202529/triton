"""
phase_197v_operator_approval_packet_review.py
Phase 197V — Operator-facing defensive-exit approval review checkpoint (no execution).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197v_operator_approval_packet_review.json"
OUTPUT_MD = RESULTS / "phase_197v_operator_approval_packet_review.md"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
U197_JSON = RESULTS / "phase_197u_paper_runtime_governance_packet_refresh.json"
SOAK_JSON = RESULTS / "phase_197t_safe_continuous_observation_soak.json"
S197_JSON = RESULTS / "phase_197s_post_interlock_runtime_verification.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "f4c34dc9"
PREVIOUS_PHASE = "197U"

OPERATOR_REVIEW_READY_APPROVAL_REQUIRED = "OPERATOR_REVIEW_READY_APPROVAL_REQUIRED"
OPERATOR_REVIEW_BLOCKED_PACKET_STALE = "OPERATOR_REVIEW_BLOCKED_PACKET_STALE"
OPERATOR_REVIEW_BLOCKED_SAFETY_EVIDENCE_MISSING = "OPERATOR_REVIEW_BLOCKED_SAFETY_EVIDENCE_MISSING"
OPERATOR_REVIEW_BLOCKED_OPEN_ORDERS = "OPERATOR_REVIEW_BLOCKED_OPEN_ORDERS"
OPERATOR_REVIEW_BLOCKED_LIVE_NOT_BLOCKED = "OPERATOR_REVIEW_BLOCKED_LIVE_NOT_BLOCKED"
OPERATOR_REVIEW_BLOCKED_APPROVAL_ARTIFACT_PRESENT = (
    "OPERATOR_REVIEW_BLOCKED_APPROVAL_ARTIFACT_PRESENT"
)
OPERATOR_REVIEW_BLOCKED_EXECUTION_DETECTED = "OPERATOR_REVIEW_BLOCKED_EXECUTION_DETECTED"

FINAL_OPERATOR_INSTRUCTION = (
    "No execution is approved by this review. Creating the real operator approval artifact "
    "requires explicit operator authorization in a separate phase."
)

BLOCKED_COMMAND_NOTE = (
    "manage_positions / place_live_orders defensive-exit dispatch remains blocked until "
    "data/governance/operator_paper_defensive_exit_approval.json exists with a valid "
    "APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH decision matching the current packet_id and "
    "source_preview_hash."
)

FORBIDDEN_ARTIFACT = "data/governance/operator_paper_defensive_exit_approval.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _git_branch() -> str:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()


def _git_clean() -> bool:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return not (proc.stdout or "").strip()


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


def _extract_symbol_analytics(
    preview_doc: Dict[str, Any],
    packet: Dict[str, Any],
) -> Dict[str, Any]:
    capped_block = _runtime_capped_preview(preview_doc)
    orders = list(capped_block.get("capped_preview_orders") or [])
    all_orders = list(capped_block.get("capped_preview_orders") or [])
    for extra in preview_doc.get("capped_preview", {}).get("capped_preview_orders") or []:
        if isinstance(extra, dict):
            all_orders.append(extra)

    symbols_to_exit = list(
        packet.get("capped_symbols") or capped_block.get("capped_preview_symbols") or []
    )
    deferred = list(packet.get("deferred_symbols") or capped_block.get("deferred_symbols") or [])

    trim_symbols: List[str] = []
    forced_symbols: List[str] = []
    for order in orders:
        sym = str(order.get("symbol") or "").upper()
        if not sym:
            continue
        opp = str(order.get("opportunity_type") or order.get("effective_stance") or "").upper()
        bucket = str(order.get("priority_bucket") or "").upper()
        reason = str(order.get("reason_code") or "").upper()
        if "TRIM" in opp or bucket == "TRIM":
            trim_symbols.append(sym)
        if "FORCE" in opp or "FORCE" in reason:
            forced_symbols.append(sym)

    largest_notional_symbol = None
    largest_notional = -1.0
    for order in orders:
        try:
            notional = float(order.get("notional") or 0.0)
            sym = str(order.get("symbol") or "")
            if notional > largest_notional and sym:
                largest_notional = notional
                largest_notional_symbol = sym
        except (TypeError, ValueError):
            continue

    highest_drag_symbol = None
    priority_order = list(capped_block.get("priority_order") or [])
    highest_risk = list(capped_block.get("highest_risk_symbols") or [])
    if highest_risk:
        highest_drag_symbol = highest_risk[0]
    elif priority_order:
        highest_drag_symbol = priority_order[0]

    drag_orders = [
        o
        for o in orders
        if "DRAG" in str(o.get("reason_code") or "").upper()
        and str(o.get("symbol") or "") in symbols_to_exit
    ]
    if drag_orders:
        drag_orders.sort(key=lambda o: int(o.get("rank") or 999))
        highest_drag_symbol = str(drag_orders[0].get("symbol") or highest_drag_symbol or "")

    batch_notional = float(
        packet.get("capped_notional")
        or capped_block.get("batch_notional_capped_preview")
        or preview_doc.get("batch_notional_estimate")
        or 0.0
    )
    batch_cap = float(
        packet.get("configured_cap")
        or capped_block.get("batch_cap")
        or preview_doc.get("configured_batch_cap_manage_positions")
        or preview_doc.get("configured_batch_cap_effective")
        or 0.0
    )
    full_batch = float(
        capped_block.get("batch_notional_full") or preview_doc.get("batch_notional_estimate") or 0.0
    )
    if full_batch > batch_cap > 0:
        cap_status = "EXCEEDS_CAP_PARTIAL_BATCH"
    elif batch_notional <= batch_cap:
        cap_status = "WITHIN_CAP"
    else:
        cap_status = str(preview_doc.get("handoff_status") or "UNKNOWN")

    planned_exits = int(
        preview_doc.get("planned_exit_count")
        or packet.get("risk_reduction_intent", {}).get("planned_exit_count")
        or len(preview_doc.get("handoff_symbols") or [])
    )

    return {
        "planned_exits_count": planned_exits,
        "planned_trims_count": len(trim_symbols),
        "symbols_to_exit": symbols_to_exit,
        "symbols_to_trim": trim_symbols,
        "deferred_symbols": deferred,
        "batch_notional": round(batch_notional, 2),
        "batch_notional_cap": batch_cap,
        "batch_notional_full": round(full_batch, 2),
        "cap_status": cap_status,
        "forced_exit_symbols": forced_symbols,
        "trim_priority_symbols": trim_symbols,
        "largest_notional_symbol": largest_notional_symbol,
        "highest_drag_symbol": highest_drag_symbol,
        "concentration_risk_note": capped_block.get("concentration_risk_note"),
        "handoff_status": preview_doc.get("handoff_status"),
        "risk_reduction_summary": (packet.get("risk_reduction_intent") or {}).get("summary"),
    }


def _soak_execution_evidence(soak_doc: Dict[str, Any]) -> Tuple[int, int, bool]:
    exits = 0
    trims = 0
    broker_submit = False
    for cycle in soak_doc.get("cycles") or []:
        exits = max(exits, int(cycle.get("exits_executed") or 0))
        trims = max(trims, int(cycle.get("trims_executed") or 0))
        broker_submit = broker_submit or bool(cycle.get("broker_orders_submitted"))
    return exits, trims, broker_submit


def _safety_evidence_ok(
    soak_doc: Dict[str, Any],
    s197_doc: Dict[str, Any],
    u197_doc: Dict[str, Any],
) -> bool:
    soak_ok = soak_doc.get("status") == "PASS" and soak_doc.get("soak_state") in (
        "SOAK_PASS_SAFE_BLOCKING",
        "SOAK_PASS_NO_ACTION",
    )
    s197_ok = s197_doc.get("status") in ("PASS", "PASS_SAFE_BLOCK") or s197_doc.get(
        "post_interlock_runtime_state"
    ) in (
        "POST_INTERLOCK_RUNTIME_VERIFIED_SAFE_BLOCK",
        "POST_INTERLOCK_RUNTIME_VERIFIED",
    )
    u197_ok = u197_doc.get("status") == "PASS"
    return bool(soak_ok and s197_ok and u197_ok)


def classify_review_state(
    *,
    packet_current: bool,
    safety_ok: bool,
    live_blocked: bool,
    real_approval_exists: bool,
    open_orders_count: int,
    broker_orders_submitted: bool,
    exits_executed: int,
    trims_executed: int,
) -> Tuple[str, str]:
    if real_approval_exists:
        return (
            OPERATOR_REVIEW_BLOCKED_APPROVAL_ARTIFACT_PRESENT,
            "real_approval_artifact_present",
        )
    if not live_blocked:
        return OPERATOR_REVIEW_BLOCKED_LIVE_NOT_BLOCKED, "live_blocked=false"
    if open_orders_count > 0:
        return OPERATOR_REVIEW_BLOCKED_OPEN_ORDERS, f"open_orders_count={open_orders_count}"
    if broker_orders_submitted or exits_executed > 0 or trims_executed > 0:
        return (
            OPERATOR_REVIEW_BLOCKED_EXECUTION_DETECTED,
            f"broker_orders={broker_orders_submitted} exits={exits_executed} trims={trims_executed}",
        )
    if not safety_ok:
        return (
            OPERATOR_REVIEW_BLOCKED_SAFETY_EVIDENCE_MISSING,
            "197S_or_197T_or_197U_safety_evidence_missing",
        )
    if not packet_current:
        return OPERATOR_REVIEW_BLOCKED_PACKET_STALE, "operator_packet_stale"
    return OPERATOR_REVIEW_READY_APPROVAL_REQUIRED, "packet_current_approval_required_no_execution"


def _render_markdown(review: Dict[str, Any]) -> str:
    lines = [
        "# TRITON Phase 197V — Operator Defensive-Exit Approval Review",
        "",
        f"**Generated:** {review.get('generated_at')}",
        f"**Review state:** {review.get('review_state')}",
        f"**Status:** {review.get('status')}",
        "",
        "## Executive summary",
        "",
        review.get("risk_reduction_summary")
        or "Capped defensive paper exit batch under governance review.",
        "",
        f"- Market open: **{review.get('market_open')}**",
        f"- Operator approval required: **{review.get('operator_approval_required')}**",
        f"- Live blocked: **{review.get('live_blocked')}** ({review.get('live_block_reason')})",
        f"- Eligible for execution: **{review.get('eligible_for_execution')}**",
        f"- Real approval artifact exists: **{review.get('real_approval_artifact_exists')}**",
        "",
        "## Proposed action (not approved)",
        "",
        f"- Planned exits (full handoff): **{review.get('planned_exits_count')}**",
        f"- Planned trims: **{review.get('planned_trims_count')}**",
        f"- Symbols in capped batch: **{', '.join(review.get('symbols_to_exit') or [])}**",
        f"- Deferred symbols: **{', '.join(review.get('deferred_symbols') or [])}**",
        f"- Batch notional (capped preview): **${review.get('batch_notional'):,.2f}**",
        f"- Batch notional cap: **${review.get('batch_notional_cap'):,.2f}**",
        f"- Full handoff notional: **${review.get('batch_notional_full'):,.2f}**",
        f"- Cap status: **{review.get('cap_status')}**",
        f"- Highest drag symbol: **{review.get('highest_drag_symbol')}**",
        f"- Largest notional symbol (capped batch): **{review.get('largest_notional_symbol')}**",
        "",
        "## Risk flags",
        "",
        f"- Handoff status: {review.get('handoff_status')}",
        f"- Governance packet state: {review.get('governance_packet_state')}",
        f"- 197N state: {review.get('governance_review_state_197n')}",
        f"- 197O state: {review.get('market_hours_recheck_state_197o')}",
        f"- Dispatch interlock: {review.get('dispatch_interlock_state')}",
        f"- Open orders: {review.get('open_orders_count')}",
        f"- Broker orders submitted (soak): {review.get('broker_orders_submitted')}",
        f"- Exits executed (soak): {review.get('exits_executed')}",
        f"- Trims executed (soak): {review.get('trims_executed')}",
        "",
        review.get("concentration_risk_note") or "",
        "",
        "## Runtime safety evidence",
        "",
        f"- 197U: {review.get('phase_197u_status')} / {review.get('governance_packet_state')}",
        f"- 197T soak: {review.get('phase_197t_status')} / {review.get('soak_state')}",
        f"- 197S interlock: {review.get('phase_197s_status')} / {review.get('post_interlock_runtime_state')}",
        "",
        "## Blocked until explicit approval",
        "",
        review.get("blocked_command_note") or BLOCKED_COMMAND_NOTE,
        "",
        f"**Forbidden unless explicitly approved:** `{FORBIDDEN_ARTIFACT}`",
        "",
        f"**Packet ID:** `{review.get('operator_packet_id')}`",
        f"**Source preview hash:** `{review.get('source_preview_hash')}`",
        "",
        "## Operator decision options (informational only)",
        "",
    ]
    for opt in review.get("operator_decision_options") or []:
        if isinstance(opt, dict):
            lines.append(f"- **{opt.get('decision')}**: {opt.get('description')}")
    lines.extend(
        [
            "",
            "## Final operator instruction",
            "",
            f"> {review.get('final_operator_instruction')}",
            "",
            "---",
            "",
            "*This document is a read-only review checkpoint. It does not approve execution.*",
        ]
    )
    return "\n".join(lines)


def run_phase_197v_review(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    u197_doc = _read_json(U197_JSON)
    soak_doc = _read_json(SOAK_JSON)
    s197_doc = _read_json(S197_JSON)
    packet = _read_json(OPERATOR_PACKET_JSON)
    preview_doc = _read_json(PREVIEW_JSON)

    from services.defensive_exit_dispatch_interlock import (
        evaluate_defensive_exit_dispatch_interlock,
    )
    from services.defensive_exit_governance_review_criteria import (
        GOV_REVIEW_BLOCKED_BY_STALE_PACKET,
        evaluate_governance_review_criteria,
    )

    criteria = evaluate_governance_review_criteria(
        broker_snapshot_ok=True,
        broker_orders_submitted=False,
    )
    interlock = evaluate_defensive_exit_dispatch_interlock(broker_snapshot_ok=True)

    analytics = _extract_symbol_analytics(preview_doc, packet)
    exits_executed, trims_executed, broker_orders_submitted = _soak_execution_evidence(soak_doc)
    safety_ok = _safety_evidence_ok(soak_doc, s197_doc, u197_doc)

    packet_current = bool(criteria.get("operator_packet_current"))
    live_blocked = bool(criteria.get("live_blocked"))
    real_approval_exists = REAL_APPROVAL_PATH.is_file()
    open_orders_count = int(criteria.get("open_orders_count") or 0)
    eligible_for_execution = bool(criteria.get("eligible_for_execution"))

    review_state, review_reason = classify_review_state(
        packet_current=packet_current,
        safety_ok=safety_ok,
        live_blocked=live_blocked,
        real_approval_exists=real_approval_exists,
        open_orders_count=open_orders_count,
        broker_orders_submitted=broker_orders_submitted,
        exits_executed=exits_executed,
        trims_executed=trims_executed,
    )

    status = "PASS" if review_state == OPERATOR_REVIEW_READY_APPROVAL_REQUIRED else "FAIL"
    if review_state != OPERATOR_REVIEW_READY_APPROVAL_REQUIRED:
        notes.append(review_reason)
    if real_approval_exists:
        status = "FAIL"
        notes.append("real approval artifact must not exist")
    if not live_blocked:
        status = "FAIL"
        notes.append("live must remain blocked")
    if eligible_for_execution:
        status = "FAIL"
        notes.append("eligible_for_execution must be false")
    if open_orders_count != 0:
        status = "FAIL"
        notes.append("open orders must be 0")
    if broker_orders_submitted or exits_executed != 0 or trims_executed != 0:
        status = "FAIL"
        notes.append("unexpected execution in soak evidence")

    n197_state = criteria.get("governance_review_state")
    o197_state = None
    try:
        from services.defensive_exit_market_hours_governance_recheck import (
            run_market_hours_governance_recheck,
        )

        o197_state = run_market_hours_governance_recheck(
            broker_snapshot_ok=True,
            broker_orders_submitted=False,
        ).get("market_hours_recheck_state")
    except Exception:
        o197_state = None

    payload: Dict[str, Any] = {
        "phase": "197V",
        "generated_at": _utc_now_iso(),
        "status": status,
        "review_state": review_state,
        "review_reason": review_reason,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "market_open": criteria.get("market_open"),
        "governance_packet_state": u197_doc.get("final_governance_packet_state"),
        "phase_197u_status": u197_doc.get("status"),
        "governance_review_state_197n": n197_state,
        "market_hours_recheck_state_197o": o197_state,
        "phase_197t_status": soak_doc.get("status"),
        "soak_state": soak_doc.get("soak_state"),
        "phase_197s_status": s197_doc.get("status"),
        "post_interlock_runtime_state": s197_doc.get("post_interlock_runtime_state"),
        "open_orders_count": open_orders_count,
        "live_blocked": live_blocked,
        "live_block_reason": criteria.get("live_block_reason"),
        "real_approval_artifact_exists": real_approval_exists,
        "eligible_for_execution": eligible_for_execution,
        "operator_approval_required": bool(packet.get("approval_required", True)),
        "operator_packet_current": packet_current,
        "operator_packet_id": packet.get("packet_id"),
        "source_preview_hash": packet.get("source_preview_hash")
        or criteria.get("source_preview_hash"),
        "operator_packet_path": str(OPERATOR_PACKET_JSON.relative_to(ROOT)).replace("\\", "/"),
        "capped_preview_path": str(PREVIEW_JSON.relative_to(ROOT)).replace("\\", "/"),
        "dispatch_interlock_state": interlock.get("dispatch_interlock_state"),
        "safe_block_state": u197_doc.get("safe_block_state"),
        "broker_orders_submitted": broker_orders_submitted,
        "exits_executed": exits_executed,
        "trims_executed": trims_executed,
        "blocked_command_note": BLOCKED_COMMAND_NOTE,
        "forbidden_artifact_path": FORBIDDEN_ARTIFACT,
        "final_operator_instruction": FINAL_OPERATOR_INSTRUCTION,
        "operator_decision_options": packet.get("operator_decision_options"),
        "current_blockers": packet.get("current_blockers"),
        "execution_prerequisites": packet.get("execution_prerequisites"),
        "review_markdown_path": str(OUTPUT_MD.relative_to(ROOT)).replace("\\", "/"),
        **analytics,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }

    if n197_state == GOV_REVIEW_BLOCKED_BY_STALE_PACKET:
        status = "FAIL"
        payload["status"] = status
        notes.append("197N still reports stale packet — run phase_197u refresh first")

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(f"[197V-OPERATOR-REVIEW] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[197V-OPERATOR-REVIEW] wrote {OUTPUT_MD.relative_to(ROOT)}")
        print(
            f"[197V-OPERATOR-REVIEW] status={status} review_state={review_state} "
            f"symbols={len(payload.get('symbols_to_exit') or [])} "
            f"approval_required={payload.get('operator_approval_required')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197V operator approval packet review")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197v_review(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
