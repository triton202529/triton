"""
phase_197s_fix_dispatch_interlock_safe_block_exit_validation.py
Phase 197S-FIX3 — Normalize dispatch interlock safe-block exit semantics.
Read-only; does not approve, submit orders, or restart loops.
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
GOVERNANCE = ROOT / "data" / "governance"
CONTINUOUS_LOG_DIR = RESULTS / "continuous_loop"
MARKET_OPEN_LOG_DIR = RESULTS / "scheduled_runs"
OUTPUT_JSON = RESULTS / "phase_197s_fix_dispatch_interlock_safe_block_exit_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "7f333c81"
PREVIOUS_PHASE = "197S-FIX2"

SAFE_BLOCK_STATE = "SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED"
DISPATCH_BLOCKED_APPROVAL_REQUIRED = "DISPATCH_BLOCKED_APPROVAL_REQUIRED"
MANAGE_POSITIONS_EXPECTED_EXIT_CODE = 0
MARKET_SEQUENCE_EXPECTED_OUTCOME = "complete_or_safe_block_no_abort"
CONTINUOUS_CYCLE_EXPECTED_OUTCOME = "CYCLE_OK_or_safe_block"


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


def _static_safe_block_exit_ok() -> Tuple[bool, List[str]]:
    notes: List[str] = []
    interlock_src = (ROOT / "services/defensive_exit_dispatch_interlock.py").read_text(
        encoding="utf-8"
    )
    manage_src = (ROOT / "services/manage_positions.py").read_text(encoding="utf-8")
    place_src = (ROOT / "services/place_live_orders.py").read_text(encoding="utf-8")

    ok = True
    if SAFE_BLOCK_STATE not in interlock_src:
        ok = False
        notes.append("missing SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED constant")
    if "is_dispatch_approval_safe_block_state" not in interlock_src:
        ok = False
        notes.append("missing is_dispatch_approval_safe_block_state helper")
    if "print_dispatch_approval_safe_block" not in interlock_src:
        ok = False
        notes.append("missing print_dispatch_approval_safe_block helper")
    if "[SAFE_BLOCK]" not in interlock_src:
        ok = False
        notes.append("missing [SAFE_BLOCK] marker emission in interlock module")

    if "is_dispatch_approval_safe_block_state" not in manage_src:
        ok = False
        notes.append("manage_positions missing safe-block state check")
    if 'return 0, 0, rs, "", meta' not in manage_src:
        ok = False
        notes.append("manage_positions missing safe-block exit 0 return")
    if manage_src.count('return 2, 0, rs, "", meta') < 1:
        notes.append(
            "manage_positions still has hard-block return 2 path (expected for non-safe blocks)"
        )

    if "is_dispatch_approval_safe_block_state" not in place_src:
        ok = False
        notes.append("place_live_orders missing safe-block state check")
    if "raise SystemExit(0)" not in place_src:
        ok = False
        notes.append("place_live_orders missing safe-block exit 0 path")

    return ok, notes


def _parse_latest_runtime(
    market_text: str,
    continuous_text: str,
) -> Dict[str, Any]:
    dispatch_state = None
    m = re.search(r"\[DISPATCH_INTERLOCK\] status=(DISPATCH_[A-Z_]+)", market_text)
    if m:
        dispatch_state = m.group(1)
    stage_m = re.search(r"\[DISPATCH_INTERLOCK\][^\n]*stage=(\w+)", market_text)
    safe_block_marker_present = "[SAFE_BLOCK]" in market_text
    abort = "[ABORT] Market-open sequence aborted" in market_text
    abort_sig = None
    if abort:
        am = re.search(r"\[ABORT\] Market-open sequence aborted at step: (.+)", market_text)
        abort_sig = am.group(1).strip() if am else "market_open_abort"
    exits_executed = 0
    em = re.search(r"exits_executed=(\d+)", market_text)
    if em:
        try:
            exits_executed = int(em.group(1))
        except ValueError:
            pass
    trims_executed = 0
    tm = re.search(r"trims_executed=(\d+)", market_text)
    if tm:
        try:
            trims_executed = int(tm.group(1))
        except ValueError:
            pass
    broker_submit = bool(
        re.search(r"\[PLACE\] mode=paper dry_run=False", market_text)
        or re.search(r"broker_orders_submitted=true", market_text, re.IGNORECASE)
    )
    cycle_fail = "[CYCLE_FAIL]" in continuous_text
    cycle_ok = "[CYCLE_OK]" in continuous_text
    pre_fix_abort_evidence = (
        dispatch_state == DISPATCH_BLOCKED_APPROVAL_REQUIRED
        and abort
        and not safe_block_marker_present
    )
    return {
        "dispatch_interlock_state": dispatch_state,
        "dispatch_interlock_stage": stage_m.group(1) if stage_m else None,
        "safe_block_marker_present": safe_block_marker_present,
        "safe_block_marker_required": True,
        "latest_runtime_had_abort": abort,
        "latest_runtime_abort_signature": abort_sig,
        "broker_orders_submitted": broker_submit,
        "exits_executed": exits_executed,
        "trims_executed": trims_executed,
        "pre_fix_abort_evidence": pre_fix_abort_evidence,
        "continuous_cycle_fail": cycle_fail,
        "continuous_cycle_ok": cycle_ok,
    }


def run_phase_197s_fix3_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    from services.defensive_exit_dispatch_interlock import (
        SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED,
        evaluate_defensive_exit_dispatch_interlock,
    )
    from services.live_block_classifier import defensive_exit_phase_validator_gates

    static_ok, static_notes = _static_safe_block_exit_ok()
    notes.extend(static_notes)

    latest_market = _latest_log(MARKET_OPEN_LOG_DIR, "market_open_*.log")
    latest_continuous = _latest_log(CONTINUOUS_LOG_DIR, "continuous_loop_*.log")
    market_text = _read_text(latest_market)
    continuous_text = _read_text(latest_continuous)
    runtime = _parse_latest_runtime(market_text, continuous_text)

    interlock = evaluate_defensive_exit_dispatch_interlock(broker_snapshot_ok=True)
    gates = defensive_exit_phase_validator_gates(verbose=False)
    real_approval_exists = REAL_APPROVAL_PATH.is_file()

    dispatch_blocks = (
        interlock.get("dispatch_interlock_state") == DISPATCH_BLOCKED_APPROVAL_REQUIRED
    )
    live_blocked = bool(interlock.get("live_blocked"))
    open_orders = int(interlock.get("open_orders_count") or 0)
    eligible = bool(interlock.get("eligible_for_execution"))

    status = "PASS"
    if not static_ok:
        status = "FAIL"
        notes.append("static safe-block exit semantics incomplete")
    if not dispatch_blocks:
        status = "FAIL"
        notes.append("interlock no longer blocks missing approval")
    if real_approval_exists:
        status = "FAIL"
        notes.append("real approval artifact must not exist")
    if runtime.get("broker_orders_submitted"):
        status = "FAIL"
        notes.append("broker orders submitted in latest runtime log")
    if int(runtime.get("exits_executed") or 0) != 0:
        status = "FAIL"
        notes.append("exits_executed must be 0")
    if not live_blocked:
        status = "FAIL"
        notes.append("live must remain blocked")
    if open_orders != 0:
        status = "FAIL"
        notes.append("open orders must be 0")
    if eligible:
        status = "FAIL"
        notes.append("eligible_for_execution must be false")

    if runtime.get("pre_fix_abort_evidence"):
        notes.append(
            "latest runtime log shows pre-fix abort without [SAFE_BLOCK] — "
            "expected until next continuous-loop restart"
        )

    payload: Dict[str, Any] = {
        "phase": "197S-FIX3",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "safe_block_state": SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED,
        "manage_positions_expected_exit_code": MANAGE_POSITIONS_EXPECTED_EXIT_CODE,
        "market_sequence_expected_outcome": MARKET_SEQUENCE_EXPECTED_OUTCOME,
        "continuous_cycle_expected_outcome": CONTINUOUS_CYCLE_EXPECTED_OUTCOME,
        "dispatch_interlock_state": interlock.get("dispatch_interlock_state"),
        "dispatch_interlock_stage": runtime.get("dispatch_interlock_stage") or "manage_positions",
        "broker_orders_submitted": runtime.get("broker_orders_submitted"),
        "exits_executed": runtime.get("exits_executed"),
        "trims_executed": runtime.get("trims_executed"),
        "real_approval_artifact_exists": real_approval_exists,
        "live_blocked": live_blocked,
        "live_block_reason": interlock.get("live_block_reason"),
        "open_orders_count": open_orders,
        "eligible_for_execution": eligible,
        "paper_entry_gate_status": gates.get("gate_classification"),
        "latest_market_open_log": (
            str(latest_market.relative_to(ROOT)).replace("\\", "/") if latest_market else None
        ),
        "latest_continuous_log": (
            str(latest_continuous.relative_to(ROOT)).replace("\\", "/")
            if latest_continuous
            else None
        ),
        "latest_runtime_had_abort": runtime.get("latest_runtime_had_abort"),
        "latest_runtime_abort_signature": runtime.get("latest_runtime_abort_signature"),
        "safe_block_marker_required": runtime.get("safe_block_marker_required"),
        "safe_block_marker_present": runtime.get("safe_block_marker_present"),
        "pre_fix_abort_evidence": runtime.get("pre_fix_abort_evidence"),
        "continuous_cycle_fail": runtime.get("continuous_cycle_fail"),
        "continuous_cycle_ok": runtime.get("continuous_cycle_ok"),
        "static_safe_block_exit_ok": static_ok,
        "interlock_eval": interlock,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197S-FIX3] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197S-FIX3] status={status} static_ok={static_ok} "
            f"dispatch_blocks={dispatch_blocks} pre_fix_abort={runtime.get('pre_fix_abort_evidence')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 197S-FIX3 dispatch interlock safe-block exit validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197s_fix3_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
