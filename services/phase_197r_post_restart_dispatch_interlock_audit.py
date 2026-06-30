"""
phase_197r_post_restart_dispatch_interlock_audit.py
Phase 197R — Audit first post-restart cycle dispatch path and approval interlock.
Read-only; never approves, executes, or submits broker orders.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
CONTINUOUS_LOG_DIR = RESULTS / "continuous_loop"
MARKET_OPEN_LOG_DIR = RESULTS / "scheduled_runs"
OUTPUT_JSON = RESULTS / "phase_197r_post_restart_dispatch_interlock_audit.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "6e88d89e"
PREVIOUS_PHASE = "197Q-FIX"


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


def _latest_log(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _read_text(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _first_match(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def _first_int(text: str, pattern: str) -> Optional[int]:
    val = _first_match(text, pattern)
    if val is None:
        return None
    try:
        return int(float(val))
    except ValueError:
        return None


def parse_continuous_log(text: str) -> Dict[str, Any]:
    return {
        "continuous_loop_seen": "CONTINUOUS LOOP START" in text,
        "cycle_completed": ("[CYCLE_OK]" in text) or ("[CYCLE_FAIL]" in text),
        "cycle_ok": "[CYCLE_OK]" in text,
    }


def parse_market_open_log(text: str) -> Dict[str, Any]:
    manage_dry = _first_match(text, r"\[manage_positions\] mode=paper dry_run=(True|False)")
    place_dry = _first_match(text, r"\[PLACE\] mode=paper dry_run=(True|False)")
    mapped_rows = _first_int(text, r"mapped_exit_rows=(\d+)")
    exits_executed = _first_int(text, r"exits_executed=(\d+)")
    trims_executed = _first_int(text, r"trims_executed=(\d+)")
    dispatch_interlock_seen = "[DISPATCH_INTERLOCK]" in text
    dispatch_state = _first_match(text, r"\[DISPATCH_INTERLOCK\] status=(DISPATCH_[A-Z_]+)")
    batch_block = (
        "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED" in text or "BATCH_NOTIONAL_BLOCK" in text
    )
    batch_notional = _first_match(text, r"(?:est_batch_notional|batch_notional)=([\d.]+)")
    batch_cap = _first_match(text, r"max_batch_notional=([\d.]+)")
    paper_entry_gate = _first_match(text, r"\[PAPER_ENTRY_GATE\] status=(\S+)")
    paper_entry_reason = _first_match(text, r"\[PAPER_ENTRY_GATE\] status=\S+ reason=(.+)")
    broker_submit = bool(re.search(r"broker\.submit_order|action=submit", text, re.I))
    master_ready = bool(re.search(r"\[MasterExecutionGate\].*status=READY", text, re.IGNORECASE))
    return {
        "market_open_sequence_executed": "MARKET OPEN SEQUENCE START" in text,
        "manage_positions_dry_run": (
            None if manage_dry is None else str(manage_dry).lower() == "true"
        ),
        "placement_dry_run": None if place_dry is None else str(place_dry).lower() == "true",
        "mapped_exit_rows": mapped_rows,
        "batch_notional_block_seen": batch_block,
        "batch_notional": float(batch_notional) if batch_notional else None,
        "batch_notional_cap": float(batch_cap) if batch_cap else None,
        "exits_executed": exits_executed if exits_executed is not None else 0,
        "trims_executed": trims_executed if trims_executed is not None else 0,
        "dispatch_interlock_seen_in_log": dispatch_interlock_seen,
        "dispatch_interlock_state_in_log": dispatch_state,
        "master_execution_gate_ready_seen": master_ready,
        "paper_entry_gate_status_in_log": paper_entry_gate,
        "paper_entry_gate_reason_in_log": (paper_entry_reason or "").strip(),
        "broker_submit_seen_in_log": broker_submit,
    }


def run_phase_197r_audit(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()

    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP,
        DISPATCH_BLOCKED_GOVERNANCE,
        DISPATCH_BLOCKED_LIVE_BLOCK,
        DISPATCH_BLOCKED_MARKET_CLOSED,
        DISPATCH_BLOCKED_OPEN_ORDERS,
        DISPATCH_BLOCKED_ORDER_GUARD,
        DISPATCH_BLOCKED_READINESS,
        evaluate_defensive_exit_dispatch_interlock,
    )
    from services.live_block_classifier import defensive_exit_phase_validator_gates
    from services.runtime_process_detector import detect_runtime_loop_processes

    latest_continuous = _latest_log(CONTINUOUS_LOG_DIR, "continuous_loop_*.log")
    latest_market_open = _latest_log(MARKET_OPEN_LOG_DIR, "market_open_*.log")
    continuous_text = _read_text(latest_continuous)
    market_text = _read_text(latest_market_open)

    continuous_parsed = parse_continuous_log(continuous_text)
    market_parsed = parse_market_open_log(market_text)
    processes = detect_runtime_loop_processes()
    gates = defensive_exit_phase_validator_gates(verbose=False)
    interlock = evaluate_defensive_exit_dispatch_interlock(
        broker_snapshot_ok=True,
        planned_count=int(market_parsed.get("mapped_exit_rows") or 0),
    )

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

    broker_orders_submitted = bool(market_parsed.get("broker_submit_seen_in_log"))
    exits_executed = int(market_parsed.get("exits_executed") or 0)
    real_approval_exists = REAL_APPROVAL_PATH.is_file()
    dispatch_state = str(interlock.get("dispatch_interlock_state") or "")
    dispatch_reason = str(interlock.get("dispatch_interlock_reason") or "")

    batch_notional_status = str(interlock.get("batch_notional_status") or "")
    if market_parsed.get("batch_notional_block_seen"):
        batch_notional_status = "EXCEEDS_CAP"

    no_execution_confirmed = (
        not broker_orders_submitted and exits_executed == 0 and not real_approval_exists
    )

    if broker_orders_submitted:
        status = "FAIL"
        notes.append("broker orders submitted")
    if exits_executed != 0:
        status = "FAIL"
        notes.append("exits_executed must be 0")
    if real_approval_exists:
        status = "FAIL"
        notes.append("real approval artifact exists unexpectedly")
    if not interlock.get("live_blocked"):
        status = "FAIL"
        notes.append("live block regression")
    if interlock.get("dispatch_permitted") and not real_approval_exists:
        status = "FAIL"
        notes.append("dispatch permitted without real approval artifact")
    if dispatch_state not in (
        DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        DISPATCH_BLOCKED_GOVERNANCE,
        DISPATCH_BLOCKED_READINESS,
        DISPATCH_BLOCKED_MARKET_CLOSED,
        DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP,
        DISPATCH_BLOCKED_ORDER_GUARD,
        DISPATCH_BLOCKED_LIVE_BLOCK,
        DISPATCH_BLOCKED_OPEN_ORDERS,
    ):
        status = "FAIL"
        notes.append(f"unexpected dispatch_interlock_state {dispatch_state}")
    if not no_execution_confirmed:
        status = "FAIL"
        notes.append("no_execution_confirmed=false")
    if (
        market_parsed.get("placement_dry_run") is False
        and not market_parsed.get("dispatch_interlock_seen_in_log")
        and market_parsed.get("mapped_exit_rows")
    ):
        notes.append(
            "pre_197r_log_reached_non_dry_run_placement_before_interlock; "
            "current interlock now blocks earlier"
        )

    if dispatch_state != DISPATCH_BLOCKED_APPROVAL_REQUIRED and not real_approval_exists:
        if dispatch_state == DISPATCH_BLOCKED_MARKET_CLOSED:
            notes.append("market closed block acceptable after hours")
        elif dispatch_state in (
            DISPATCH_BLOCKED_GOVERNANCE,
            DISPATCH_BLOCKED_READINESS,
            DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP,
        ):
            notes.append(f"dispatch blocked by {dispatch_state} (also blocks execution)")

    payload: Dict[str, Any] = {
        "phase": "197R",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "Post-restart observation runtime reached non-dry-run manage/place path and relied "
            "on batch notional cap as final blocker after mapping exit rows."
        ),
        "fix": (
            "defensive_exit_dispatch_interlock blocks manage handoff before place_live_orders "
            "when real operator approval artifact is absent or other gates fail."
        ),
        "continuous_loop_seen": continuous_parsed.get("continuous_loop_seen"),
        "watchdog_loop_running": processes.get("watchdog_loop_running"),
        "continuous_loop_running": processes.get("continuous_loop_running"),
        "latest_continuous_log": (
            str(latest_continuous.relative_to(ROOT)).replace("\\", "/")
            if latest_continuous
            else None
        ),
        "latest_market_open_log": (
            str(latest_market_open.relative_to(ROOT)).replace("\\", "/")
            if latest_market_open
            else None
        ),
        "cycle_completed": continuous_parsed.get("cycle_completed"),
        "manage_positions_dry_run": market_parsed.get("manage_positions_dry_run"),
        "placement_dry_run": market_parsed.get("placement_dry_run"),
        "mapped_exit_rows": market_parsed.get("mapped_exit_rows"),
        "batch_notional_status": batch_notional_status,
        "batch_notional": interlock.get("batch_notional"),
        "batch_notional_cap": interlock.get("batch_notional_cap"),
        "exits_executed": exits_executed,
        "trims_executed": market_parsed.get("trims_executed"),
        "broker_orders_submitted": broker_orders_submitted,
        "paper_entry_gate_status": gates.get("gate_classification"),
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "real_approval_artifact_exists": real_approval_exists,
        "operator_approval_status": interlock.get("operator_approval_status"),
        "governance_authorized": interlock.get("governance_authorized"),
        "readiness_status": interlock.get("readiness_status"),
        "live_blocked": interlock.get("live_blocked"),
        "live_block_reason": interlock.get("live_block_reason"),
        "order_guard_authority_state": interlock.get("order_guard_authority_state"),
        "dispatch_interlock_state": dispatch_state,
        "dispatch_interlock_reason": dispatch_reason,
        "dispatch_interlock_seen_in_log": market_parsed.get("dispatch_interlock_seen_in_log"),
        "dispatch_interlock_state_in_log": market_parsed.get("dispatch_interlock_state_in_log"),
        "eligible_for_execution": False,
        "no_execution_confirmed": no_execution_confirmed,
        "broker_snapshot_ok": snapshot_ok,
        "market_open": interlock.get("market_open"),
        "log_audit": {
            "continuous": continuous_parsed,
            "market_open": market_parsed,
        },
        "interlock": interlock,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197R-DISPATCH-AUDIT] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197R-DISPATCH-AUDIT] status={status} dispatch={dispatch_state} "
            f"exits_executed={exits_executed} broker_orders={broker_orders_submitted} "
            f"mapped_rows={market_parsed.get('mapped_exit_rows')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197R post-restart dispatch interlock audit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197r_audit(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
