"""
phase_196q_operator_paper_arm_flow_repair_validation.py
Phase 196Q — Validate read-only paper ARM operator flow visibility.
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
LIVE = ROOT / "data" / "live"
OUTPUT_JSON = RESULTS / "phase_196q_operator_paper_arm_flow_repair.json"

PREVIOUS_COMMIT = "ab4e8e51"
PREVIOUS_PHASE = "196P"
PAPER_ARM_PATH = LIVE / "paper_arm_confirm.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _help_has_dry_run() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "services.set_paper_arm_confirm", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    return "dry-run" in text.lower() or "dry_run" in text.lower()


def run_phase_196q_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.paper_arm_status import build_paper_arm_status

    status_payload = build_paper_arm_status()
    dry_run = _help_has_dry_run()

    ps_script = ROOT / "scripts" / "show_triton_paper_arm_status.ps1"
    py_status = ROOT / "services" / "paper_arm_status.py"
    runbook = ROOT / "docs" / "TRITON_PAPER_ARM_OPERATOR_RUNBOOK.md"
    vr = (ROOT / "view_results.py").read_text(encoding="utf-8")

    positions: List[str] = []
    open_orders = 0
    live_execution = False
    try:
        from services.dashboard_status_sync import count_open_orders, load_positions_summary
        from services.portfolio_history_sync import load_live_execution_flag

        positions = list(load_positions_summary().get("positions") or [])
        open_orders = int(count_open_orders())
        live_execution = load_live_execution_flag()
    except Exception as exc:
        notes.append(f"portfolio state: {exc}")

    mtime_before = PAPER_ARM_PATH.stat().st_mtime if PAPER_ARM_PATH.is_file() else None

    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps_script),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    ps_ok = proc.returncode == 0 and "READ-ONLY" in (proc.stdout or "")

    mtime_after = PAPER_ARM_PATH.stat().st_mtime if PAPER_ARM_PATH.is_file() else None
    confirm_unchanged = mtime_before == mtime_after

    dashboard_ok = (
        "build_paper_arm_status" in vr
        and "Paper arm confirm:" in vr
        and "Live confirm:" in vr
        and "set_paper_arm_confirm" not in vr.split("render_sidebar_execution_status")[1][:800]
    )

    status = "PASS"
    if not ps_script.is_file() or not py_status.is_file():
        status = "FAIL"
        notes.append("Status scripts missing.")
    elif not ps_ok:
        status = "FAIL"
        notes.append(
            f"show_triton_paper_arm_status.ps1 failed: {(proc.stderr or proc.stdout)[:200]}"
        )
    elif not confirm_unchanged:
        status = "FAIL"
        notes.append("paper_arm_confirm.json was modified during read-only status check.")
    elif positions != ["JPM", "WFC"]:
        status = "FAIL"
        notes.append(f"Unexpected positions: {positions}")
    elif open_orders != 0:
        status = "FAIL"
        notes.append("Open orders not zero.")
    elif live_execution:
        status = "FAIL"
        notes.append("live_execution is true.")
    elif not dashboard_ok:
        status = "REVIEW"
        notes.append("Verify dashboard sidebar paper arm clarity.")
    else:
        notes.append(
            f"Paper arm status={status_payload.get('status')}; "
            f"armed={status_payload.get('paper_entry_armed')}; dry_run={dry_run}."
        )

    payload = {
        "phase": "196Q",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "paper_arm_status_source_identified": True,
        "paper_arm_status_command_created": ps_script.is_file() and py_status.is_file(),
        "paper_arm_status_command_read_only": ps_ok and confirm_unchanged,
        "paper_arm_confirm_path": str(PAPER_ARM_PATH.relative_to(ROOT)).replace("\\", "/"),
        "paper_arm_confirm_status": status_payload.get("status"),
        "arm_mode": status_payload.get("mode"),
        "operator_required": status_payload.get("operator_required"),
        "set_paper_arm_confirm_inspected": True,
        "set_paper_arm_confirm_dry_run_available": dry_run,
        "paper_arm_confirm_created": False,
        "live_confirm_created": False,
        "manual_approval_used": False,
        "execute_trades_run": False,
        "broker_orders_submitted": False,
        "live_execution": live_execution,
        "open_orders": open_orders,
        "positions": positions,
        "dashboard_clarity_updated": dashboard_ok,
        "runbook_created": runbook.is_file(),
        "paper_arm_status": status_payload,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "certification_logic_changed": False,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196Q] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196Q] status={status}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196Q paper arm flow validation")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196q_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
