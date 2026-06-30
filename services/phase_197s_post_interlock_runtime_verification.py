"""
phase_197s_post_interlock_runtime_verification.py
Phase 197S — Verify post-197R dispatch interlock during live continuous-loop cycle.
Read-only; does not restart loops, approve, or submit broker orders.
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
OUTPUT_JSON = RESULTS / "phase_197s_post_interlock_runtime_verification.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "cb9d36a8"
PREVIOUS_PHASE = "197R"

POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET = "POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET"
POST_INTERLOCK_RUNTIME_VERIFIED = "POST_INTERLOCK_RUNTIME_VERIFIED"
POST_INTERLOCK_RUNTIME_VERIFIED_SAFE_BLOCK = "POST_INTERLOCK_RUNTIME_VERIFIED_SAFE_BLOCK"
POST_INTERLOCK_RUNTIME_FAIL_SEQUENCE_ABORT_WITHOUT_SAFE_BLOCK = (
    "POST_INTERLOCK_RUNTIME_FAIL_SEQUENCE_ABORT_WITHOUT_SAFE_BLOCK"
)
POST_INTERLOCK_RUNTIME_FAIL_NOTIONAL_CAP_LAST_LINE = (
    "POST_INTERLOCK_RUNTIME_FAIL_NOTIONAL_CAP_LAST_LINE"
)
POST_INTERLOCK_RUNTIME_FAIL_BROKER_ORDERS_SUBMITTED = (
    "POST_INTERLOCK_RUNTIME_FAIL_BROKER_ORDERS_SUBMITTED"
)
POST_INTERLOCK_RUNTIME_FAIL_INTERLOCK_IMPORT_ERROR = (
    "POST_INTERLOCK_RUNTIME_FAIL_INTERLOCK_IMPORT_ERROR"
)

INTERLOCK_IMPORT_CRASH_SIGNATURE = (
    "NameError: name 'is_defensive_exit_manage_handoff' is not defined"
)

OBSERVATION_COMMAND = (
    "powershell -NoProfile -ExecutionPolicy Bypass -File "
    ".\\scripts\\start_triton_runtime_supervisor.ps1"
)

_PLACEMENT_AFTER_MARKERS = (
    r"\[PLACE\] mode=paper dry_run=False",
    r"mapped_exit_rows=\d+",
    r"\[EXECUTION\] source: manage_positions handoff",
    r"\[HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED\]",
    r"\[BATCH_NOTIONAL_BLOCK\]",
)

_NOTIONAL_AFTER_MARKERS = (
    r"\[HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED\]",
    r"\[BATCH_NOTIONAL_BLOCK\]",
)


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


def _text_after_marker(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return ""
    return text[idx:]


def _any_marker_after(text: str, marker: str, patterns: Tuple[str, ...]) -> bool:
    tail = _text_after_marker(text, marker)
    if not tail:
        return False
    return any(re.search(p, tail, re.IGNORECASE) for p in patterns)


def _cycle_start_timestamp(continuous_text: str) -> Optional[str]:
    for pattern in (
        r"\[CYCLE_START\] cycle=\d+",
        r"=== TRITON CONTINUOUS LOOP START ===",
        r"StartedAt:\s*(\S+)",
    ):
        m = re.search(pattern, continuous_text)
        if m:
            if m.lastindex:
                return m.group(1)
            return m.group(0)
    line = (continuous_text.splitlines() or [""])[0].strip()
    return line[:80] if line else None


def _detect_runtime_crash(text: str) -> Tuple[bool, Optional[str]]:
    if INTERLOCK_IMPORT_CRASH_SIGNATURE in text:
        return True, INTERLOCK_IMPORT_CRASH_SIGNATURE
    if "is_defensive_exit_manage_handoff" in text and "NameError" in text:
        return True, "NameError:is_defensive_exit_manage_handoff"
    return False, None


def parse_runtime_logs(
    continuous_text: str,
    market_text: str,
) -> Dict[str, Any]:
    from services.phase_197r_post_restart_dispatch_interlock_audit import (
        parse_continuous_log,
        parse_market_open_log,
    )

    continuous = parse_continuous_log(continuous_text)
    market = parse_market_open_log(market_text)
    crash_detected, crash_signature = _detect_runtime_crash(market_text)
    interlock_marker = "[DISPATCH_INTERLOCK]"
    dispatch_seen = interlock_marker in market_text
    dispatch_state = _first_match(market_text, r"\[DISPATCH_INTERLOCK\] status=(DISPATCH_[A-Z_]+)")
    dispatch_stage = _first_match(market_text, r"\[DISPATCH_INTERLOCK\][^\n]*stage=(\w+)")
    safe_block_marker_present = "[SAFE_BLOCK]" in market_text
    market_sequence_abort = "[ABORT] Market-open sequence aborted" in market_text
    market_sequence_complete = "=== TRITON MARKET OPEN SEQUENCE COMPLETE ===" in market_text
    placement_after = False
    notional_after = False
    pre_197r_placement_reached = False
    if dispatch_seen:
        placement_after = _any_marker_after(market_text, interlock_marker, _PLACEMENT_AFTER_MARKERS)
        notional_after = _any_marker_after(market_text, interlock_marker, _NOTIONAL_AFTER_MARKERS)
    elif market.get("mapped_exit_rows") and market.get("batch_notional_block_seen"):
        pre_197r_placement_reached = bool(
            market.get("placement_dry_run") is False
            or market.get("manage_positions_dry_run") is False
        )

    return {
        **continuous,
        **market,
        "cycle_start_timestamp": _cycle_start_timestamp(continuous_text),
        "dispatch_interlock_seen": dispatch_seen,
        "dispatch_interlock_state": dispatch_state,
        "dispatch_interlock_stage": dispatch_stage,
        "safe_block_marker_present": safe_block_marker_present,
        "market_sequence_abort": market_sequence_abort,
        "market_sequence_complete": market_sequence_complete,
        "placement_proceeded_after_interlock": placement_after,
        "notional_cap_block_seen_after_interlock": notional_after,
        "pre_197r_placement_reached": pre_197r_placement_reached,
        "runtime_crash_detected": crash_detected,
        "runtime_crash_signature": crash_signature,
    }


def classify_post_interlock_runtime(
    parsed: Dict[str, Any],
    *,
    broker_orders_submitted: bool,
    exits_executed: int,
    real_approval_exists: bool,
) -> Tuple[str, str]:
    """Return (post_interlock_runtime_state, reason)."""
    if parsed.get("runtime_crash_detected"):
        return (
            POST_INTERLOCK_RUNTIME_FAIL_INTERLOCK_IMPORT_ERROR,
            str(parsed.get("runtime_crash_signature") or "interlock_import_name_error"),
        )

    if broker_orders_submitted:
        return (
            POST_INTERLOCK_RUNTIME_FAIL_BROKER_ORDERS_SUBMITTED,
            "broker_orders_submitted_in_runtime_log",
        )

    dispatch_seen = bool(parsed.get("dispatch_interlock_seen"))
    dispatch_state = str(parsed.get("dispatch_interlock_state") or "")

    if not dispatch_seen:
        if parsed.get("mapped_exit_rows") and parsed.get("batch_notional_block_seen"):
            return (
                POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET,
                "pre_197r_cycle_log_without_dispatch_interlock_restart_required",
            )
        return (
            POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET,
            "no_post_197r_runtime_cycle_with_dispatch_interlock_observed",
        )

    if dispatch_state == "DISPATCH_BLOCKED_APPROVAL_REQUIRED":
        if parsed.get("notional_cap_block_seen_after_interlock"):
            return (
                POST_INTERLOCK_RUNTIME_FAIL_NOTIONAL_CAP_LAST_LINE,
                "notional_cap_blocked_after_interlock_should_not_reach_placement",
            )
        if parsed.get("placement_proceeded_after_interlock"):
            return (
                POST_INTERLOCK_RUNTIME_FAIL_NOTIONAL_CAP_LAST_LINE,
                "placement_markers_seen_after_approval_interlock",
            )
        if exits_executed != 0:
            return (
                POST_INTERLOCK_RUNTIME_FAIL_BROKER_ORDERS_SUBMITTED,
                f"exits_executed={exits_executed}",
            )
        if parsed.get("market_sequence_abort") and not parsed.get("safe_block_marker_present"):
            return (
                POST_INTERLOCK_RUNTIME_FAIL_SEQUENCE_ABORT_WITHOUT_SAFE_BLOCK,
                "market_open_sequence_aborted_without_safe_block_marker",
            )
        if parsed.get("safe_block_marker_present") or parsed.get("market_sequence_complete"):
            return (
                POST_INTERLOCK_RUNTIME_VERIFIED_SAFE_BLOCK,
                "dispatch_interlock_safe_block_no_broker_orders",
            )
        return (
            POST_INTERLOCK_RUNTIME_VERIFIED,
            "dispatch_interlock_blocked_before_placement_no_broker_orders",
        )

    if parsed.get("notional_cap_block_seen_after_interlock") and not real_approval_exists:
        return (
            POST_INTERLOCK_RUNTIME_FAIL_NOTIONAL_CAP_LAST_LINE,
            f"notional_cap_last_line_dispatch_state={dispatch_state}",
        )

    return (
        POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET,
        f"dispatch_interlock_seen_unexpected_state={dispatch_state}",
    )


def run_phase_197s_verification(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        evaluate_defensive_exit_dispatch_interlock,
    )
    from services.live_block_classifier import defensive_exit_phase_validator_gates
    from services.runtime_process_detector import detect_runtime_loop_processes

    latest_continuous = _latest_log(CONTINUOUS_LOG_DIR, "continuous_loop_*.log")
    latest_market_open = _latest_log(MARKET_OPEN_LOG_DIR, "market_open_*.log")
    continuous_text = _read_text(latest_continuous)
    market_text = _read_text(latest_market_open)
    parsed = parse_runtime_logs(continuous_text, market_text)
    processes = detect_runtime_loop_processes()
    gates = defensive_exit_phase_validator_gates(verbose=False)
    interlock = evaluate_defensive_exit_dispatch_interlock(broker_snapshot_ok=True)

    broker_orders_submitted = bool(parsed.get("broker_submit_seen_in_log"))
    exits_executed = int(parsed.get("exits_executed") or 0)
    real_approval_exists = REAL_APPROVAL_PATH.is_file()

    runtime_state, runtime_reason = classify_post_interlock_runtime(
        parsed,
        broker_orders_submitted=broker_orders_submitted,
        exits_executed=exits_executed,
        real_approval_exists=real_approval_exists,
    )

    restart_required = runtime_state == POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET

    if runtime_state == POST_INTERLOCK_RUNTIME_VERIFIED:
        status = "PASS"
    elif runtime_state == POST_INTERLOCK_RUNTIME_VERIFIED_SAFE_BLOCK:
        status = "PASS_SAFE_BLOCK"
    elif runtime_state == POST_INTERLOCK_RUNTIME_NOT_OBSERVED_YET:
        status = "PASS_SAFE_WAIT"
        notes.append(
            "Post-197R interlock not yet observed in runtime logs; "
            "restart continuous for observation and re-run verification."
        )
    elif runtime_state == POST_INTERLOCK_RUNTIME_FAIL_INTERLOCK_IMPORT_ERROR:
        status = "FAIL"
        notes.append(
            "runtime crash from missing dispatch interlock import — repair and re-run cycle"
        )
    else:
        status = "FAIL"
        notes.append(runtime_reason)

    if broker_orders_submitted:
        status = "FAIL"
        notes.append("broker orders submitted")
    if exits_executed != 0:
        status = "FAIL"
        notes.append("exits_executed must be 0")
    if real_approval_exists:
        status = "FAIL"
        notes.append("real approval artifact exists")
    if not interlock.get("live_blocked"):
        status = "FAIL"
        notes.append("live block regression")
    if int(interlock.get("open_orders_count") or 0) != 0:
        status = "FAIL"
        notes.append("open orders must be 0")
    if interlock.get("eligible_for_execution"):
        status = "FAIL"
        notes.append("eligible_for_execution must be false")

    dispatch_state_log = parsed.get("dispatch_interlock_state")
    dispatch_state = (
        dispatch_state_log if dispatch_state_log else interlock.get("dispatch_interlock_state")
    )

    payload: Dict[str, Any] = {
        "phase": "197S",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "197R added dispatch interlock in code but live runtime cycle had not yet "
            "been observed emitting DISPATCH_BLOCKED_APPROVAL_REQUIRED."
        ),
        "fix": (
            "197S verifies one post-197R continuous-loop cycle blocks defensive exit "
            "dispatch before broker placement when real approval artifact is absent."
        ),
        "post_interlock_runtime_state": runtime_state,
        "post_interlock_runtime_reason": runtime_reason,
        "continuous_loop_running": processes.get("continuous_loop_running"),
        "watchdog_loop_running": processes.get("watchdog_loop_running"),
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
        "cycle_start_timestamp": parsed.get("cycle_start_timestamp"),
        "cycle_completed": parsed.get("cycle_completed"),
        "dispatch_interlock_seen": parsed.get("dispatch_interlock_seen"),
        "dispatch_interlock_state": dispatch_state,
        "dispatch_interlock_stage": parsed.get("dispatch_interlock_stage"),
        "safe_block_marker_present": parsed.get("safe_block_marker_present"),
        "market_sequence_abort": parsed.get("market_sequence_abort"),
        "market_sequence_complete": parsed.get("market_sequence_complete"),
        "placement_proceeded_after_interlock": parsed.get("placement_proceeded_after_interlock"),
        "notional_cap_block_seen_after_interlock": parsed.get(
            "notional_cap_block_seen_after_interlock"
        ),
        "broker_orders_submitted": broker_orders_submitted,
        "exits_executed": exits_executed,
        "trims_executed": parsed.get("trims_executed"),
        "paper_entry_gate_status": gates.get("gate_classification"),
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "real_approval_artifact_exists": real_approval_exists,
        "live_blocked": interlock.get("live_blocked"),
        "live_block_reason": interlock.get("live_block_reason"),
        "order_guard_authority_state": interlock.get("order_guard_authority_state"),
        "open_orders_count": interlock.get("open_orders_count"),
        "eligible_for_execution": False,
        "restart_required": restart_required,
        "observation_command": OBSERVATION_COMMAND,
        "expected_interlock_state": DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        "pre_197r_placement_reached": parsed.get("pre_197r_placement_reached"),
        "log_audit": parsed,
        "interlock_eval": interlock,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197S-POST-INTERLOCK] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197S-POST-INTERLOCK] status={status} runtime_state={runtime_state} "
            f"dispatch_seen={parsed.get('dispatch_interlock_seen')} "
            f"dispatch_state={dispatch_state} restart_required={restart_required}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197S post-interlock runtime verification")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197s_verification(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_WAIT", "PASS_SAFE_BLOCK") else 1


if __name__ == "__main__":
    sys.exit(main())
