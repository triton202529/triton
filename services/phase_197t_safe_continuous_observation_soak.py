"""
phase_197t_safe_continuous_observation_soak.py
Phase 197T — Multi-cycle continuous runtime observation soak validator.
Read-only; does not restart loops, approve, or submit broker orders.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
CONTINUOUS_LOG_DIR = RESULTS / "continuous_loop"
MARKET_OPEN_LOG_DIR = RESULTS / "scheduled_runs"
GUARD_SNAPSHOT_PATH = RESULTS / "guard_snapshot.json"
OPEN_ORDERS_SNAPSHOT = RESULTS / "open_orders_snapshot.csv"
OUTPUT_JSON = RESULTS / "phase_197t_safe_continuous_observation_soak.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "bab51cc6"
PREVIOUS_PHASE = "197S-FIX3"

SOAK_READY_FOR_OBSERVATION = "SOAK_READY_FOR_OBSERVATION"
SOAK_PASS_SAFE_BLOCKING = "SOAK_PASS_SAFE_BLOCKING"
SOAK_PASS_NO_ACTION = "SOAK_PASS_NO_ACTION"
SOAK_FAIL_BROKER_ORDERS_SUBMITTED = "SOAK_FAIL_BROKER_ORDERS_SUBMITTED"
SOAK_FAIL_CYCLE_FAILURE = "SOAK_FAIL_CYCLE_FAILURE"
SOAK_FAIL_MARKET_SEQUENCE_ABORT = "SOAK_FAIL_MARKET_SEQUENCE_ABORT"
SOAK_FAIL_LIVE_NOT_BLOCKED = "SOAK_FAIL_LIVE_NOT_BLOCKED"
SOAK_FAIL_APPROVAL_ARTIFACT_PRESENT = "SOAK_FAIL_APPROVAL_ARTIFACT_PRESENT"

DEFAULT_CYCLES = 1


def _utc_now_iso() -> str:
    from datetime import timezone

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


def _read_text(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _first_match(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def _first_int(text: str, pattern: str, default: int = 0) -> int:
    val = _first_match(text, pattern)
    if val is None:
        return default
    try:
        return int(float(val))
    except ValueError:
        return default


def _parse_market_log_filename_ts(path: Path) -> Optional[datetime]:
    m = re.match(r"market_open_(\d{8})_(\d{6})\.log$", path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _parse_cycle_start_dt(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).strip()[:19])
    except ValueError:
        return None


def _list_market_open_logs() -> List[Path]:
    if not MARKET_OPEN_LOG_DIR.is_dir():
        return []
    return sorted(MARKET_OPEN_LOG_DIR.glob("market_open_*.log"), key=lambda p: p.stat().st_mtime)


def _match_market_open_log(
    cycle_start_timestamp: Optional[str],
    market_logs: List[Path],
) -> Optional[Path]:
    cycle_dt = _parse_cycle_start_dt(cycle_start_timestamp)
    if cycle_dt is None:
        return market_logs[-1] if market_logs else None
    best: Optional[Path] = None
    best_delta: Optional[float] = None
    for path in market_logs:
        fn_dt = _parse_market_log_filename_ts(path)
        if fn_dt is None:
            continue
        delta = abs((fn_dt - cycle_dt).total_seconds())
        if delta <= 180 and (best_delta is None or delta < best_delta):
            best = path
            best_delta = delta
    return best


def _parse_paper_entry_from_segment(segment: str) -> Tuple[Optional[str], Optional[str]]:
    gate = _first_match(segment, r"\[PAPER_ENTRY_GATE\] status=(\S+)")
    reason = _first_match(segment, r"\[PAPER_ENTRY_GATE\] status=\S+ reason=(.+)")
    if gate:
        return gate, reason
    blocked = _first_match(segment, r"\[PAPER_ENTRY_EXECUTION_BLOCKED\] reason=(.+)")
    if blocked:
        return "EXECUTION_BLOCKED", blocked
    if "[PAPER_ENTRY_STAGE]" in segment:
        return "STAGE_EVALUATED", "paper_entry_loop_stage_invoked"
    return None, None


def _parse_market_cycle_fields(market_text: str, segment: str) -> Dict[str, Any]:
    from services.phase_197r_post_restart_dispatch_interlock_audit import parse_market_open_log

    market = parse_market_open_log(market_text)
    broker_submit = bool(
        market.get("broker_submit_seen_in_log")
        or re.search(r"\[PLACE\] mode=paper dry_run=False", market_text)
        or re.search(r"broker_orders_submitted=true", market_text, re.I)
    )
    dispatch_seen = "[DISPATCH_INTERLOCK]" in market_text
    dispatch_state = _first_match(market_text, r"\[DISPATCH_INTERLOCK\] status=(DISPATCH_[A-Z_]+)")
    safe_block_seen = "[SAFE_BLOCK]" in market_text
    safe_block_state = _first_match(market_text, r"\[SAFE_BLOCK\] state=(\S+)")
    market_sequence_complete = "=== TRITON MARKET OPEN SEQUENCE COMPLETE ===" in market_text
    market_sequence_abort = "[ABORT] Market-open sequence aborted" in market_text
    paper_gate, paper_reason = _parse_paper_entry_from_segment(segment)
    if not paper_gate:
        paper_gate = market.get("paper_entry_gate_status_in_log")
        paper_reason = market.get("paper_entry_gate_reason_in_log")
    defensive_handoff = bool(
        dispatch_seen
        or (market.get("mapped_exit_rows") or 0) > 0
        or "manage_positions handoff" in market_text
    )
    return {
        "market_sequence_complete": market_sequence_complete,
        "market_sequence_abort": market_sequence_abort,
        "dispatch_interlock_seen": dispatch_seen,
        "dispatch_interlock_state": dispatch_state,
        "safe_block_seen": safe_block_seen,
        "safe_block_state": safe_block_state,
        "broker_orders_submitted": broker_submit,
        "exits_executed": int(market.get("exits_executed") or 0),
        "trims_executed": int(market.get("trims_executed") or 0),
        "paper_entry_gate_status": paper_gate,
        "paper_entry_gate_reason": paper_reason,
        "defensive_handoff_reached": defensive_handoff,
        "post_197s_safe_block_cycle": bool(
            safe_block_seen
            and dispatch_state == "DISPATCH_BLOCKED_APPROVAL_REQUIRED"
            and market_sequence_complete
            and not market_sequence_abort
        ),
    }


def _parse_cycles_from_continuous_text(text: str, continuous_rel_path: str) -> List[Dict[str, Any]]:
    cycles: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for line in text.splitlines():
        start_m = re.search(r"\[CYCLE_START\] cycle=(\d+)", line)
        if start_m:
            current = {
                "cycle_number": int(start_m.group(1)),
                "cycle_start_timestamp": line[:19] if len(line) >= 19 else None,
                "continuous_log_path": continuous_rel_path.replace("\\", "/"),
                "segment_lines": [],
            }
            continue
        if current is None:
            continue
        current["segment_lines"].append(line)
        if "[CYCLE_OK]" in line:
            ok_m = re.search(r"\[CYCLE_OK\] cycle=(\d+)", line)
            current["cycle_ok"] = True
            current["cycle_fail"] = False
            current["cycle_completed"] = True
            current["cycle_end_timestamp"] = line[:19] if len(line) >= 19 else None
            if ok_m:
                current["cycle_number"] = int(ok_m.group(1))
            cycles.append(current)
            current = None
        elif "[CYCLE_FAIL]" in line:
            fail_m = re.search(r"\[CYCLE_FAIL\] cycle=(\d+)", line)
            rc_m = re.search(r"exit_code=(\d+)", line)
            current["cycle_ok"] = False
            current["cycle_fail"] = True
            current["cycle_completed"] = True
            current["cycle_fail_exit_code"] = int(rc_m.group(1)) if rc_m else None
            current["cycle_end_timestamp"] = line[:19] if len(line) >= 19 else None
            if fail_m:
                current["cycle_number"] = int(fail_m.group(1))
            cycles.append(current)
            current = None
    return cycles


def collect_recent_cycles(max_cycles: int) -> List[Dict[str, Any]]:
    if not CONTINUOUS_LOG_DIR.is_dir():
        return []
    market_logs = _list_market_open_logs()
    all_cycles: List[Dict[str, Any]] = []
    log_files = sorted(
        CONTINUOUS_LOG_DIR.glob("continuous_loop_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    for log_path in log_files:
        rel = str(log_path.relative_to(ROOT))
        for raw in _parse_cycles_from_continuous_text(_read_text(log_path), rel):
            segment = "\n".join(raw.pop("segment_lines", []))
            market_path = _match_market_open_log(raw.get("cycle_start_timestamp"), market_logs)
            market_text = _read_text(market_path)
            market_fields = _parse_market_cycle_fields(market_text, segment)
            cycle = {
                **raw,
                **market_fields,
                "market_open_log_path": (
                    str(market_path.relative_to(ROOT)).replace("\\", "/") if market_path else None
                ),
            }
            all_cycles.append(cycle)

    all_cycles.sort(key=lambda c: str(c.get("cycle_start_timestamp") or ""))
    if max_cycles <= 0:
        return all_cycles
    return all_cycles[-max_cycles:]


def _read_guard_snapshot() -> Dict[str, Any]:
    if not GUARD_SNAPSHOT_PATH.is_file():
        return {}
    try:
        return json.loads(GUARD_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _open_orders_count_from_snapshot() -> int:
    guard = _read_guard_snapshot()
    if guard.get("open_orders_count") is not None:
        return int(guard.get("open_orders_count") or 0)
    if not OPEN_ORDERS_SNAPSHOT.is_file():
        return 0
    try:
        import pandas as pd

        df = pd.read_csv(OPEN_ORDERS_SNAPSHOT)
        return int(len(df))
    except Exception:
        return 0


def classify_soak_state(
    cycles: List[Dict[str, Any]],
    *,
    live_blocked: bool,
    real_approval_exists: bool,
    eligible_for_execution: bool,
    open_orders_count: int,
) -> Tuple[str, str]:
    if real_approval_exists:
        from services.defensive_exit_operator_paper_approval import (
            is_valid_paper_only_defensive_exit_approval,
        )

        valid, reason, _ = is_valid_paper_only_defensive_exit_approval()
        if not valid:
            return SOAK_FAIL_APPROVAL_ARTIFACT_PRESENT, f"invalid_approval_artifact={reason}"

    if not live_blocked:
        return SOAK_FAIL_LIVE_NOT_BLOCKED, "live_blocked=false"
    if eligible_for_execution:
        return SOAK_FAIL_LIVE_NOT_BLOCKED, "eligible_for_execution=true"
    if open_orders_count != 0:
        return SOAK_FAIL_BROKER_ORDERS_SUBMITTED, f"open_orders_count={open_orders_count}"

    if not cycles:
        return SOAK_READY_FOR_OBSERVATION, "no_continuous_cycles_found"

    for cycle in cycles:
        if cycle.get("broker_orders_submitted"):
            return (
                SOAK_FAIL_BROKER_ORDERS_SUBMITTED,
                f"broker_orders_submitted cycle={cycle.get('cycle_number')}",
            )
        if int(cycle.get("exits_executed") or 0) != 0:
            return (
                SOAK_FAIL_BROKER_ORDERS_SUBMITTED,
                f"exits_executed={cycle.get('exits_executed')} cycle={cycle.get('cycle_number')}",
            )
        if cycle.get("market_sequence_abort") and not cycle.get("safe_block_seen"):
            return (
                SOAK_FAIL_MARKET_SEQUENCE_ABORT,
                f"market_sequence_abort cycle={cycle.get('cycle_number')}",
            )
        if cycle.get("cycle_fail"):
            if cycle.get("post_197s_safe_block_cycle"):
                continue
            return (
                SOAK_FAIL_CYCLE_FAILURE,
                f"cycle_fail exit_code={cycle.get('cycle_fail_exit_code')} "
                f"cycle={cycle.get('cycle_number')}",
            )

    post_197s = [c for c in cycles if c.get("post_197s_safe_block_cycle")]
    if not post_197s:
        all_ok = all(c.get("cycle_ok") for c in cycles)
        if all_ok and not any(c.get("defensive_handoff_reached") for c in cycles):
            return SOAK_PASS_NO_ACTION, "cycles_completed_no_defensive_dispatch_path"
        return SOAK_READY_FOR_OBSERVATION, "no_post_197s_safe_block_cycle_observed_yet"

    if all(c.get("cycle_ok") for c in cycles):
        if any(c.get("safe_block_seen") for c in cycles):
            return SOAK_PASS_SAFE_BLOCKING, "all_cycles_safe_blocking_no_execution"
        if any(c.get("defensive_handoff_reached") for c in cycles):
            return SOAK_PASS_SAFE_BLOCKING, "defensive_handoff_blocked_without_failure"
        return SOAK_PASS_NO_ACTION, "cycles_completed_no_execution"

    return SOAK_FAIL_CYCLE_FAILURE, "unexplained_cycle_failure_in_soak_window"


def run_phase_197t_soak(*, cycles: int = DEFAULT_CYCLES, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    git_clean_before = _git_clean()

    from services.defensive_exit_dispatch_interlock import (
        evaluate_defensive_exit_dispatch_interlock,
    )
    from services.live_block_classifier import defensive_exit_phase_validator_gates
    from services.runtime_process_detector import detect_runtime_loop_processes

    inspected = collect_recent_cycles(cycles)
    processes = detect_runtime_loop_processes()
    gates = defensive_exit_phase_validator_gates(verbose=False)
    interlock = evaluate_defensive_exit_dispatch_interlock(broker_snapshot_ok=True)
    guard = _read_guard_snapshot()
    real_approval_exists = REAL_APPROVAL_PATH.is_file()
    live_blocked = bool(interlock.get("live_blocked"))
    open_orders_count = _open_orders_count_from_snapshot()
    eligible_for_execution = bool(interlock.get("eligible_for_execution"))
    guard_refreshed = bool(guard.get("updated_at") or guard.get("timestamp"))

    soak_state, soak_reason = classify_soak_state(
        inspected,
        live_blocked=live_blocked,
        real_approval_exists=real_approval_exists,
        eligible_for_execution=eligible_for_execution,
        open_orders_count=open_orders_count,
    )

    if soak_state == SOAK_READY_FOR_OBSERVATION:
        status = "PASS"
        notes.append("No post-197S safe-block cycle in soak window; runtime ready to observe")
    elif soak_state in (SOAK_PASS_SAFE_BLOCKING, SOAK_PASS_NO_ACTION):
        status = "PASS"
    else:
        status = "FAIL"
        notes.append(soak_reason)

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

    per_cycle: List[Dict[str, Any]] = []
    for cycle in inspected:
        per_cycle.append(
            {
                "continuous_log_path": cycle.get("continuous_log_path"),
                "market_open_log_path": cycle.get("market_open_log_path"),
                "cycle_number": cycle.get("cycle_number"),
                "cycle_start_timestamp": cycle.get("cycle_start_timestamp"),
                "cycle_end_timestamp": cycle.get("cycle_end_timestamp"),
                "cycle_completed": cycle.get("cycle_completed"),
                "cycle_ok": cycle.get("cycle_ok"),
                "cycle_fail": cycle.get("cycle_fail"),
                "cycle_fail_exit_code": cycle.get("cycle_fail_exit_code"),
                "market_sequence_complete": cycle.get("market_sequence_complete"),
                "market_sequence_abort": cycle.get("market_sequence_abort"),
                "dispatch_interlock_seen": cycle.get("dispatch_interlock_seen"),
                "dispatch_interlock_state": cycle.get("dispatch_interlock_state"),
                "safe_block_seen": cycle.get("safe_block_seen"),
                "safe_block_state": cycle.get("safe_block_state"),
                "broker_orders_submitted": cycle.get("broker_orders_submitted"),
                "exits_executed": cycle.get("exits_executed"),
                "trims_executed": cycle.get("trims_executed"),
                "paper_entry_gate_status": cycle.get("paper_entry_gate_status")
                or gates.get("gate_classification"),
                "paper_entry_gate_reason": cycle.get("paper_entry_gate_reason")
                or gates.get("gate_reason"),
                "post_197s_safe_block_cycle": cycle.get("post_197s_safe_block_cycle"),
            }
        )

    payload: Dict[str, Any] = {
        "phase": "197T",
        "generated_at": _utc_now_iso(),
        "status": status,
        "soak_state": soak_state,
        "soak_reason": soak_reason,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "cycles_requested": cycles,
        "cycles_inspected": len(inspected),
        "cycles": per_cycle,
        "watchdog_loop_running": processes.get("watchdog_loop_running"),
        "continuous_loop_running": processes.get("continuous_loop_running"),
        "open_orders_count": open_orders_count,
        "guard_snapshot_refreshed": guard_refreshed,
        "guard_snapshot_path": (
            str(GUARD_SNAPSHOT_PATH.relative_to(ROOT)).replace("\\", "/")
            if GUARD_SNAPSHOT_PATH.is_file()
            else None
        ),
        "live_blocked": live_blocked,
        "live_block_reason": interlock.get("live_block_reason"),
        "real_approval_artifact_exists": real_approval_exists,
        "eligible_for_execution": eligible_for_execution,
        "paper_entry_gate_status": gates.get("gate_classification"),
        "paper_entry_gate_reason": gates.get("gate_reason"),
        "order_guard_authority_state": interlock.get("order_guard_authority_state"),
        "interlock_eval": interlock,
        "runtime_processes": processes,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197T-SOAK] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197T-SOAK] status={status} soak_state={soak_state} "
            f"cycles_inspected={len(inspected)} watchdog={processes.get('watchdog_loop_running')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197T safe continuous observation soak")
    ap.add_argument(
        "--cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help=f"Number of latest continuous-loop cycles to inspect (default {DEFAULT_CYCLES})",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197t_soak(cycles=max(1, int(args.cycles)), verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
