"""
phase_197p_order_guard_snapshot_authority_validation.py
Phase 197P — Validate order guard snapshot authority reconciliation (read-only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197p_order_guard_snapshot_authority_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "6e17d242"
PREVIOUS_PHASE = "197O"


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


def run_phase_197p_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()

    from services.defensive_exit_market_hours_governance_recheck import (
        run_market_hours_governance_recheck,
    )
    from services.live_block_classifier import (
        classify_paper_entry_gate_for_validation,
        defensive_exit_phase_validator_gates,
        validator_live_block_check,
    )
    from services.order_guard_snapshot_authority import (
        ORDER_GUARD_FRESH,
        ORDER_GUARD_OPEN_ORDERS_PRESENT,
        ORDER_GUARD_STALE_AND_UNVERIFIED,
        ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS,
        evaluate_order_guard_authority,
    )

    snapshot_ok = False
    guard_refreshed = False
    update_mode = "none"
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
            snap_doc = json.loads(snap.stdout[jstart:])
            snapshot_ok = bool(snap_doc.get("ok"))
            gr = snap_doc.get("guard_snapshot_refresh") or {}
            guard_refreshed = bool(gr.get("updated"))
            update_mode = "broker_authority_refresh" if guard_refreshed else "none"
    except Exception as exc:
        notes.append(f"snapshot_live_orders failed: {exc}")

    authority = evaluate_order_guard_authority(
        broker_snapshot_ok=snapshot_ok,
        refresh_guard_if_eligible=False,
        mode="paper",
    )
    recheck = run_market_hours_governance_recheck(
        broker_snapshot_ok=snapshot_ok,
        broker_orders_submitted=False,
    )
    gates = defensive_exit_phase_validator_gates(verbose=False)
    gate_status = classify_paper_entry_gate_for_validation(
        gates["gate_ok"], gates["gate_reason"], gates.get("gate_details")
    )
    live_block_ok, _ = validator_live_block_check(paper_mode=True)

    auth_state = str(authority.get("order_guard_authority_state") or "")
    open_orders = int(authority.get("open_orders_count") or 0)
    acceptable_states = {
        ORDER_GUARD_FRESH,
        ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS,
    }

    if not snapshot_ok:
        status = "FAIL"
        notes.append("broker snapshot not ok")
    if auth_state not in acceptable_states:
        status = "FAIL"
        notes.append(f"unexpected order_guard_authority_state {auth_state}")
    if auth_state == ORDER_GUARD_OPEN_ORDERS_PRESENT:
        status = "FAIL"
        notes.append("open orders present")
    if auth_state == ORDER_GUARD_STALE_AND_UNVERIFIED:
        status = "FAIL"
        notes.append("guard stale and unverified")
    if open_orders != 0:
        status = "FAIL"
        notes.append("open_orders_count must be 0")
    if not live_block_ok or not authority.get("live_blocked"):
        status = "FAIL"
        notes.append("live block regression")
    if REAL_APPROVAL_PATH.is_file():
        status = "FAIL"
        notes.append("real approval artifact exists")
    if authority.get("eligible_for_execution") or recheck.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false")
    if recheck.get("broker_orders_submitted"):
        status = "FAIL"
        notes.append("broker orders submitted")

    source_authority = authority.get("source_authority")
    if guard_refreshed:
        source_authority = "BROKER_OPEN_ORDERS_SNAPSHOT"

    payload: Dict[str, Any] = {
        "phase": "197P",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "guard_snapshot.json was stale while open_orders_snapshot.csv was fresh with "
            "zero open orders, producing ambiguous GATE_FRESHNESS warnings."
        ),
        "fix": (
            "order_guard_snapshot_authority classifies guard vs broker authority; "
            "snapshot_live_orders may refresh guard from broker snapshot when safe."
        ),
        "order_guard_authority_state": auth_state,
        "order_guard_authority_reason": authority.get("order_guard_authority_reason"),
        "guard_snapshot_path": authority.get("guard_snapshot_path"),
        "guard_snapshot_exists": authority.get("guard_snapshot_exists"),
        "guard_snapshot_age_minutes": authority.get("guard_snapshot_age_minutes"),
        "guard_snapshot_fresh": authority.get("guard_snapshot_fresh"),
        "broker_snapshot_ok": snapshot_ok,
        "open_orders_snapshot_path": authority.get("open_orders_snapshot_path"),
        "open_orders_snapshot_exists": authority.get("open_orders_snapshot_exists"),
        "open_orders_snapshot_age_minutes": authority.get("open_orders_snapshot_age_minutes"),
        "open_orders_snapshot_fresh": authority.get("open_orders_snapshot_fresh"),
        "open_orders_count": open_orders,
        "recent_orders_count": authority.get("recent_orders_count"),
        "broker_orders_submitted": False,
        "live_blocked": authority.get("live_blocked"),
        "live_block_reason": authority.get("live_block_reason"),
        "real_approval_artifact_exists": authority.get("real_approval_artifact_exists"),
        "market_open": recheck.get("market_open"),
        "paper_entry_gate_status": gate_status,
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "eligible_for_execution": False,
        "updated_guard_snapshot": guard_refreshed,
        "update_mode": update_mode,
        "source_authority": source_authority,
        "gate_freshness_note": authority.get("gate_freshness_note"),
        "recheck_order_guard_authority_state": recheck.get("order_guard_authority_state"),
        "inventory": authority.get("inventory"),
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197P-ORDER-GUARD] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197P-ORDER-GUARD] status={status} state={auth_state} "
            f"guard_fresh={authority.get('guard_snapshot_fresh')} "
            f"open_orders={open_orders} guard_refreshed={guard_refreshed}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197P order guard snapshot authority validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197p_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
