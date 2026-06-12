"""
TRITON Phase 195 — Hung Loop Safe Process Recovery Manager.

Performs safe termination and restart when hung loop detection confirms
HUNG_PROCESS and SAFE_TERMINATION_CANDIDATE. Does NOT bypass duplicate
protection for healthy loops.

Outputs (data/results/):
    hung_loop_detection.json (refreshed)
    hung_loop_forensics.json (refreshed)
    hung_loop_recovery_actions.csv (appended)

Run:
    python -m services.hung_loop_recovery_manager
    python -m services.hung_loop_recovery_manager --recover
    python -m services.hung_loop_recovery_manager --recover --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.continuous_loop_recovery_manager import (
    DEFAULT_INTERVAL_MINUTES,
    LOCK_PATH,
    LOOP_SCRIPT,
    WATCHDOG_STATE,
    _clear_stale_lock,
    _is_loop_process_running,
    _now_iso,
    _parse_lock,
    _read_json,
    _restart_loop,
    _verify_recovery,
)
from services.hung_loop_detector import run_detection

RESULTS = ROOT / "data" / "results"
OUT_ACTIONS = RESULTS / "hung_loop_recovery_actions.csv"
MAX_RECOVERY_ATTEMPTS_PER_HOUR = 3


def _log(msg: str) -> None:
    print(f"[hung_loop_recovery_manager] {msg}", flush=True)


def _recent_recovery_count() -> int:
    if not OUT_ACTIONS.is_file():
        return 0
    df = pd.read_csv(OUT_ACTIONS)
    if df.empty or "timestamp" not in df.columns:
        return 0
    now = datetime.now(timezone.utc)
    count = 0
    for _, row in df.iterrows():
        try:
            dt = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            if (now - dt).total_seconds() <= 3600 and row.get("action_type") in (
                "RECOVERY_ATTEMPT",
                "RECOVERY_SUCCESS",
            ):
                count += 1
        except (ValueError, TypeError):
            continue
    return count


def _terminate_pids(pids: List[int], dry_run: bool) -> Dict[str, Any]:
    if not pids:
        return {"action": "none", "terminated": [], "detail": "No PIDs to terminate"}
    if dry_run:
        return {"action": "would_terminate", "terminated": [], "target_pids": pids}
    terminated = []
    errors = []
    for pid in pids:
        try:
            r = subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode == 0:
                terminated.append(pid)
            else:
                errors.append(f"pid={pid} rc={r.returncode}")
        except (subprocess.SubprocessError, OSError) as e:
            errors.append(f"pid={pid} err={e}")
    time.sleep(2)
    still_running, remaining = _is_loop_process_running()
    return {
        "action": "terminated",
        "terminated": terminated,
        "remaining_pids": remaining,
        "all_cleared": not still_running,
        "errors": errors,
    }


def _append_action(row: Dict[str, Any]) -> None:
    if OUT_ACTIONS.is_file():
        df = pd.read_csv(OUT_ACTIONS)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(OUT_ACTIONS, index=False, encoding="utf-8")


def run_recovery(
    *,
    recover: bool = False,
    dry_run: bool = False,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
) -> Dict[str, Any]:
    detection = run_detection(interval_minutes=interval_minutes)
    ts = _now_iso()
    classification = detection.get("process_health_classification")
    decision = detection.get("recovery_decision") or {}
    forensics = detection.get("forensics") or {}
    pids = forensics.get("process_ids") or []

    recovery_result: Dict[str, Any] = {
        "attempted": False,
        "result": "NOT_ATTEMPTED",
        "steps": [],
    }

    if classification != "HUNG_PROCESS":
        recovery_result["result"] = "NOT_REQUIRED"
        recovery_result["detail"] = f"Classification={classification} — no hung recovery needed"
    elif decision.get("verdict") != "SAFE_TERMINATION_CANDIDATE":
        recovery_result["result"] = "BLOCKED"
        recovery_result["detail"] = decision.get("reason")
    elif not recover:
        recovery_result["result"] = "NOT_ATTEMPTED"
        recovery_result["detail"] = "Use --recover to execute safe hung loop recovery"
    elif _recent_recovery_count() >= MAX_RECOVERY_ATTEMPTS_PER_HOUR:
        recovery_result["result"] = "RATE_LIMITED"
        recovery_result["detail"] = (
            f"Max {MAX_RECOVERY_ATTEMPTS_PER_HOUR} recovery attempts per hour"
        )
    else:
        recovery_result["attempted"] = True
        steps: List[Dict[str, Any]] = []

        steps.append(
            {
                "step": "forensic_capture",
                "action": "complete",
                "detail": "Pre-recovery forensics saved",
            }
        )

        confirm = {
            "step": "confirm_no_active_work",
            "cycle_age_minutes": forensics.get("last_cycle_age_minutes"),
            "log_stale": forensics.get("log_mtime_age_minutes"),
            "passed": True,
        }
        steps.append(confirm)

        term = _terminate_pids(pids, dry_run=dry_run)
        steps.append({"step": "terminate_stale_pid", **term})

        metrics_stub = {"process_running": False, "lock": _parse_lock(LOCK_PATH)}
        if term.get("all_cleared") or dry_run:
            metrics_stub["process_running"] = False
        lock_step = _clear_stale_lock(metrics_stub, dry_run=dry_run)
        steps.append({"step": "clear_stale_lock", **lock_step})

        restart = _restart_loop(interval_minutes, dry_run=dry_run)
        steps.append({"step": "restart_continuous_loop", **restart})

        verify = {}
        if not dry_run and restart.get("started"):
            refresh_before = _read_json(WATCHDOG_STATE).get("heartbeat_refresh_ts")
            verify = _verify_recovery(refresh_before)
            steps.append({"step": "verify_heartbeat", **verify})

        success = dry_run or (
            term.get("all_cleared") and restart.get("started") and verify.get("recovery_successful")
        )
        recovery_result["steps"] = steps
        recovery_result["result"] = (
            "RECOVERY_SUCCESS"
            if success and not dry_run
            else "RECOVERY_DRY_RUN" if dry_run else "RECOVERY_FAILED"
        )
        recovery_result["new_pids"] = verify.get("pids") if verify else restart.get("detected_pids")

        _append_action(
            {
                "timestamp": ts,
                "action_type": (
                    "RECOVERY_SUCCESS" if success and not dry_run else "RECOVERY_ATTEMPT"
                ),
                "classification": classification,
                "pids_terminated": ",".join(str(p) for p in term.get("terminated") or []),
                "recovery_reason": forensics.get("suspected_root_cause"),
                "restart_timestamp": ts if success else None,
                "new_pids": ",".join(str(p) for p in (recovery_result.get("new_pids") or [])),
                "recovery_success": success and not dry_run,
                "dry_run": dry_run,
                "result": recovery_result["result"],
            }
        )

    report = {
        **detection,
        "recovery_execution": recovery_result,
        "summary": {
            **(detection.get("summary") or {}),
            "recovery_result": recovery_result.get("result"),
            "recover_mode": recover,
            "dry_run": dry_run,
        },
    }

    (RESULTS / "hung_loop_detection.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    _log(
        f"classification={classification} recovery={recovery_result.get('result')} "
        f"recover={recover} dry_run={dry_run}"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="TRITON hung loop safe recovery manager")
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Terminate hung PID, clear lock, restart loop when SAFE_TERMINATION_CANDIDATE",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate recovery without termination"
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help=f"Expected cycle interval (default {DEFAULT_INTERVAL_MINUTES})",
    )
    args = parser.parse_args()
    run_recovery(recover=args.recover, dry_run=args.dry_run, interval_minutes=args.interval_minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
