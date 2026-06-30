"""
phase_197u_paper_runtime_governance_packet_refresh.py
Phase 197U — Refresh paper runtime governance review packet/evidence (no approval).
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
OUTPUT_JSON = RESULTS / "phase_197u_paper_runtime_governance_packet_refresh.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
GUARD_SNAPSHOT_PATH = RESULTS / "guard_snapshot.json"
SOAK_JSON = RESULTS / "phase_197t_safe_continuous_observation_soak.json"
S197_JSON = RESULTS / "phase_197s_post_interlock_runtime_verification.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "8680e237"
PREVIOUS_PHASE = "197T"

GOV_PACKET_REFRESHED_OPERATOR_REVIEW_REQUIRED = "GOV_PACKET_REFRESHED_OPERATOR_REVIEW_REQUIRED"
GOV_PACKET_CURRENT_OPERATOR_REVIEW_REQUIRED = "GOV_PACKET_CURRENT_OPERATOR_REVIEW_REQUIRED"
GOV_PACKET_REFRESH_BLOCKED_MISSING_SAFE_SOAK = "GOV_PACKET_REFRESH_BLOCKED_MISSING_SAFE_SOAK"
GOV_PACKET_REFRESH_BLOCKED_OPEN_ORDERS = "GOV_PACKET_REFRESH_BLOCKED_OPEN_ORDERS"
GOV_PACKET_REFRESH_BLOCKED_LIVE_NOT_BLOCKED = "GOV_PACKET_REFRESH_BLOCKED_LIVE_NOT_BLOCKED"
GOV_PACKET_REFRESH_BLOCKED_APPROVAL_ARTIFACT_PRESENT = (
    "GOV_PACKET_REFRESH_BLOCKED_APPROVAL_ARTIFACT_PRESENT"
)
GOV_PACKET_REFRESH_BLOCKED_EXECUTION_DETECTED = "GOV_PACKET_REFRESH_BLOCKED_EXECUTION_DETECTED"

SAFE_BLOCK_STATE = "SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
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


def _artifact_age_hours(path: Path) -> Optional[float]:
    doc = _read_json(path)
    for key in ("generated_at", "timestamp", "updated_at"):
        dt = _parse_iso(str(doc.get(key) or ""))
        if dt:
            return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    if path.is_file():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0
    return None


def _runtime_capped_preview(doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        doc.get("capped_preview_runtime_cap")
        or doc.get("capped_preview")
        or doc.get("capped_preview_cli_cap")
        or {}
    )


def _load_safe_soak_evidence() -> Tuple[bool, Dict[str, Any]]:
    soak = _read_json(SOAK_JSON)
    s197 = _read_json(S197_JSON)

    if soak.get("status") != "PASS":
        from services.phase_197t_safe_continuous_observation_soak import run_phase_197t_soak

        soak = run_phase_197t_soak(cycles=2, verbose=False)

    if s197.get("status") not in ("PASS", "PASS_SAFE_BLOCK", "PASS_SAFE_WAIT"):
        from services.phase_197s_post_interlock_runtime_verification import (
            run_phase_197s_verification,
        )

        s197 = run_phase_197s_verification(verbose=False)

    soak_ok = soak.get("soak_state") in (
        "SOAK_PASS_SAFE_BLOCKING",
        "SOAK_PASS_NO_ACTION",
    )
    s197_ok = s197.get("status") in ("PASS", "PASS_SAFE_BLOCK") or s197.get(
        "post_interlock_runtime_state"
    ) in (
        "POST_INTERLOCK_RUNTIME_VERIFIED_SAFE_BLOCK",
        "POST_INTERLOCK_RUNTIME_VERIFIED",
    )
    return bool(soak_ok and s197_ok), {"soak": soak, "s197": s197}


def _runtime_execution_counts(soak_doc: Dict[str, Any]) -> Tuple[int, int, bool]:
    exits = 0
    trims = 0
    broker_submit = False
    for cycle in soak_doc.get("cycles") or []:
        exits = max(exits, int(cycle.get("exits_executed") or 0))
        trims = max(trims, int(cycle.get("trims_executed") or 0))
        broker_submit = broker_submit or bool(cycle.get("broker_orders_submitted"))
    if not broker_submit:
        broker_submit = bool(soak_doc.get("broker_orders_submitted"))
    return exits, trims, broker_submit


def refresh_governance_packet(
    *,
    broker_snapshot_ok: bool = True,
) -> Tuple[Dict[str, Any], bool, str]:
    """Rebuild operator approval packet from current preview/sync. Returns (packet, refreshed, reason)."""
    from services.defensive_exit_governance_review_criteria import (
        _packet_current,
        collect_governance_review_evidence,
    )
    from services.defensive_exit_governance_sync import (
        build_defensive_exit_approval_sync,
        build_operator_approval_packet,
        write_operator_approval_packet,
    )
    from services.defensive_exit_operator_paper_approval import compute_capped_preview_hash

    before = collect_governance_review_evidence(
        broker_snapshot_ok=broker_snapshot_ok,
        broker_orders_submitted=False,
    )
    preview_doc = _read_json(PREVIEW_JSON)
    preview_hash = compute_capped_preview_hash(preview_doc) if preview_doc else ""
    packet_before = _read_json(OPERATOR_PACKET_JSON)
    was_current, stale_reason = _packet_current(packet_before, preview_hash)

    if was_current:
        return packet_before, False, "packet_already_current"

    sync = build_defensive_exit_approval_sync(cap_context="runtime")
    packet = build_operator_approval_packet(sync=sync)
    packet = dict(packet)
    packet["phase"] = "197U"
    packet["refresh_note"] = (
        "Governance packet refreshed from current capped preview; does not approve execution."
    )
    write_operator_approval_packet(packet)
    return packet, True, str(stale_reason or "packet_refreshed")


def classify_packet_refresh_state(
    *,
    soak_ok: bool,
    live_blocked: bool,
    real_approval_exists: bool,
    open_orders_count: int,
    broker_orders_submitted: bool,
    exits_executed: int,
    trims_executed: int,
    packet_refreshed: bool,
) -> Tuple[str, str]:
    if real_approval_exists:
        from services.defensive_exit_operator_paper_approval import (
            is_valid_paper_only_defensive_exit_approval,
        )

        valid, reason, _ = is_valid_paper_only_defensive_exit_approval()
        if valid:
            return (
                GOV_PACKET_CURRENT_OPERATOR_REVIEW_REQUIRED,
                "valid_paper_approval_present_post_197W",
            )
        return (
            GOV_PACKET_REFRESH_BLOCKED_APPROVAL_ARTIFACT_PRESENT,
            f"invalid_approval_artifact={reason}",
        )
    if not live_blocked:
        return GOV_PACKET_REFRESH_BLOCKED_LIVE_NOT_BLOCKED, "live_blocked=false"
    if open_orders_count > 0:
        return GOV_PACKET_REFRESH_BLOCKED_OPEN_ORDERS, f"open_orders_count={open_orders_count}"
    if broker_orders_submitted or exits_executed > 0 or trims_executed > 0:
        return (
            GOV_PACKET_REFRESH_BLOCKED_EXECUTION_DETECTED,
            f"broker_orders={broker_orders_submitted} exits={exits_executed} trims={trims_executed}",
        )
    if not soak_ok:
        return GOV_PACKET_REFRESH_BLOCKED_MISSING_SAFE_SOAK, "197T_or_197S_safe_evidence_missing"
    if packet_refreshed:
        return (
            GOV_PACKET_REFRESHED_OPERATOR_REVIEW_REQUIRED,
            "packet_refreshed_operator_review_required",
        )
    return (
        GOV_PACKET_CURRENT_OPERATOR_REVIEW_REQUIRED,
        "packet_current_operator_review_required",
    )


def run_phase_197u_refresh(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    snapshot_ok = False
    try:
        snap = subprocess.run(
            [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        jstart = (snap.stdout or "").find("{")
        if jstart >= 0:
            snapshot_ok = bool(json.loads(snap.stdout[jstart:]).get("ok"))
    except Exception:
        pass

    soak_ok, soak_bundle = _load_safe_soak_evidence()
    soak_doc = soak_bundle.get("soak") or {}
    s197_doc = soak_bundle.get("s197") or {}
    exits_executed, trims_executed, broker_orders_submitted = _runtime_execution_counts(soak_doc)

    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        evaluate_defensive_exit_dispatch_interlock,
    )
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.live_block_classifier import defensive_exit_phase_validator_gates
    from services.order_guard_snapshot_authority import evaluate_order_guard_authority
    from services.phase_197m_fix_live_block_detection_validation import (
        run_phase_197m_fix_validation,
    )
    from services.phase_197p_order_guard_snapshot_authority_validation import (
        run_phase_197p_validation,
    )

    guard = evaluate_order_guard_authority(
        broker_snapshot_ok=snapshot_ok,
        refresh_guard_if_eligible=False,
        mode="paper",
    )
    interlock = evaluate_defensive_exit_dispatch_interlock(broker_snapshot_ok=snapshot_ok)
    gates = defensive_exit_phase_validator_gates(verbose=False)

    preview_doc = _read_json(PREVIEW_JSON)
    capped_block = _runtime_capped_preview(preview_doc)
    packet_before_current = None
    try:
        from services.defensive_exit_governance_review_criteria import (
            collect_governance_review_evidence,
        )

        packet_before_current = collect_governance_review_evidence(
            broker_snapshot_ok=snapshot_ok
        ).get("operator_packet_current")
    except Exception:
        packet_before_current = None

    packet, packet_refreshed, refresh_reason = refresh_governance_packet(
        broker_snapshot_ok=snapshot_ok
    )
    criteria = evaluate_governance_review_criteria(
        broker_snapshot_ok=snapshot_ok,
        broker_orders_submitted=False,
    )

    live_blocked = bool(criteria.get("live_blocked"))
    real_approval_exists = REAL_APPROVAL_PATH.is_file()
    open_orders_count = int(criteria.get("open_orders_count") or 0)
    eligible_for_execution = bool(criteria.get("eligible_for_execution"))

    final_state, final_reason = classify_packet_refresh_state(
        soak_ok=soak_ok,
        live_blocked=live_blocked,
        real_approval_exists=real_approval_exists,
        open_orders_count=open_orders_count,
        broker_orders_submitted=broker_orders_submitted,
        exits_executed=exits_executed,
        trims_executed=trims_executed,
        packet_refreshed=packet_refreshed,
    )

    status = "PASS"
    if final_state.startswith("GOV_PACKET_REFRESH_BLOCKED"):
        status = "FAIL"
        notes.append(final_reason)
    if real_approval_exists:
        from services.defensive_exit_operator_paper_approval import (
            is_valid_paper_only_defensive_exit_approval,
        )

        valid, reason, _ = is_valid_paper_only_defensive_exit_approval()
        if not valid:
            status = "FAIL"
            notes.append(f"invalid approval artifact: {reason}")
        else:
            notes.append("valid_paper_approval_present_post_197W")
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
        notes.append("unexpected execution detected in soak evidence")

    gov_197n = criteria.get("governance_review_state")
    if gov_197n == "GOV_REVIEW_BLOCKED_BY_STALE_PACKET":
        status = "FAIL"
        notes.append("governance still reports stale packet after refresh")

    m197 = run_phase_197m_fix_validation(verbose=False)
    p197 = run_phase_197p_validation(verbose=False)

    payload: Dict[str, Any] = {
        "phase": "197U",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "final_governance_packet_state": final_state,
        "final_governance_packet_reason": final_reason,
        "packet_refreshed": packet_refreshed,
        "packet_refresh_reason": refresh_reason,
        "packet_before_current": packet_before_current,
        "packet_after_current": criteria.get("operator_packet_current"),
        "current_packet_path": str(OPERATOR_PACKET_JSON.relative_to(ROOT)).replace("\\", "/"),
        "current_packet_age_hours": _artifact_age_hours(OPERATOR_PACKET_JSON),
        "capped_batch_preview_path": str(PREVIEW_JSON.relative_to(ROOT)).replace("\\", "/"),
        "capped_batch_preview_age_hours": _artifact_age_hours(PREVIEW_JSON),
        "planned_exits_count": int(
            preview_doc.get("planned_exit_count") or len(preview_doc.get("handoff_symbols") or [])
        ),
        "planned_trims_count": 0,
        "batch_notional": float(preview_doc.get("batch_notional_estimate") or 0.0),
        "batch_notional_cap": float(
            preview_doc.get("configured_batch_cap_manage_positions")
            or preview_doc.get("configured_batch_cap_effective")
            or 0.0
        ),
        "capped_notional": criteria.get("capped_notional"),
        "configured_cap": criteria.get("configured_cap"),
        "market_open": criteria.get("market_open"),
        "soak_status": soak_doc.get("status"),
        "soak_state": soak_doc.get("soak_state"),
        "post_interlock_runtime_state": s197_doc.get("post_interlock_runtime_state"),
        "post_interlock_runtime_status": s197_doc.get("status"),
        "order_guard_authority_state": guard.get("order_guard_authority_state"),
        "live_block_state": m197.get("status"),
        "live_blocked": live_blocked,
        "live_block_reason": criteria.get("live_block_reason"),
        "governance_review_state": gov_197n,
        "governance_review_reason": criteria.get("governance_review_reason"),
        "dispatch_interlock_state": interlock.get("dispatch_interlock_state"),
        "safe_block_state": (
            SAFE_BLOCK_STATE
            if interlock.get("dispatch_interlock_state") == DISPATCH_BLOCKED_APPROVAL_REQUIRED
            else None
        ),
        "open_orders_count": open_orders_count,
        "guard_snapshot_refreshed": bool(_read_json(GUARD_SNAPSHOT_PATH).get("updated_at")),
        "broker_orders_submitted": broker_orders_submitted,
        "exits_executed": exits_executed,
        "trims_executed": trims_executed,
        "real_approval_artifact_exists": real_approval_exists,
        "eligible_for_execution": eligible_for_execution,
        "execution_readiness": criteria.get("readiness_status"),
        "paper_entry_gate_status": gates.get("gate_classification"),
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "operator_packet_id": packet.get("packet_id"),
        "source_preview_hash": criteria.get("source_preview_hash"),
        "packet": packet,
        "criteria": criteria,
        "soak_evidence": soak_bundle,
        "phase_197m_status": m197.get("status"),
        "phase_197p_status": p197.get("status"),
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197U-GOV-REFRESH] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197U-GOV-REFRESH] status={status} final_state={final_state} "
            f"refreshed={packet_refreshed} gov_state={gov_197n}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197U paper governance packet refresh")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197u_refresh(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
