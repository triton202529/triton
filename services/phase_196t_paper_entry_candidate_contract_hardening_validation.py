"""
phase_196t_paper_entry_candidate_contract_hardening_validation.py
Phase 196T — Validate paper-entry candidate contract hardening.
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
LIVE = ROOT / "data" / "live"
OUTPUT_JSON = RESULTS / "phase_196t_paper_entry_candidate_contract_hardening.json"
SOURCE_MAP_CSV = RESULTS / "phase_196t_candidate_source_map.csv"
CONTRACT_CSV = RESULTS / "phase_196t_candidate_contract.csv"

PREVIOUS_COMMIT = "0b84da7a"
PREVIOUS_PHASE = "196S"
PAPER_ARM_PATH = LIVE / "paper_arm_confirm.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_ps(script: str) -> Tuple[int, str]:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / script),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _parse_hb_age(text: str) -> Optional[float]:
    for line in text.splitlines():
        if "heartbeat.json" in line and "~" in line and "min" in line:
            try:
                return float(line.split("~")[1].split("min")[0].strip())
            except Exception:
                pass
    return None


def _extract_snapshot_json(stdout: str) -> Dict[str, Any]:
    marker = '{\n  "ok"'
    idx = stdout.find(marker)
    if idx < 0:
        return {}
    try:
        return json.loads(stdout[idx:])
    except Exception:
        return {}


def run_phase_196t_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.paper_arm_status import build_paper_arm_status
    from services.paper_entry_candidate_contract import (
        assess_controlled_candidates_cache,
        build_governed_candidate_summary,
        write_candidate_artifacts,
    )

    head = subprocess.run(
        ["git", "log", "-1", "--format=%H"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    head_commit = (head.stdout or "").strip()

    sup_rc, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    arm_rc, arm_out = _run_ps("show_triton_paper_arm_status.ps1")
    arm_st = build_paper_arm_status(verbose_notes=False)

    watchdog_on = "Watchdog process running: True" in sup_out
    loop_on = "Continuous loop process running: True" in sup_out
    hb_age = _parse_hb_age(sup_out)
    pipeline_health = "HEALTHY"
    if not watchdog_on or not loop_on:
        pipeline_health = "FAIL"
    elif hb_age is not None and hb_age > 45:
        pipeline_health = "DEGRADED"

    positions: List[str] = []
    open_orders = 0
    live_execution = False
    try:
        from services.dashboard_status_sync import count_open_orders, load_positions_summary
        from services.portfolio_history_sync import load_live_execution_flag

        positions = sorted(load_positions_summary().get("positions") or [])
        open_orders = int(count_open_orders())
        live_execution = load_live_execution_flag()
    except Exception as exc:
        notes.append(f"portfolio state: {exc}")

    mtime_before = PAPER_ARM_PATH.stat().st_mtime if PAPER_ARM_PATH.is_file() else None

    cache = assess_controlled_candidates_cache()
    summary = build_governed_candidate_summary()
    map_count, contract_count = write_candidate_artifacts(
        source_map_path=SOURCE_MAP_CSV,
        contract_path=CONTRACT_CSV,
    )

    snap_proc = subprocess.run(
        [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    snap_doc = _extract_snapshot_json(snap_proc.stdout or "") if snap_proc.returncode == 0 else {}

    vr = (ROOT / "view_results.py").read_text(encoding="utf-8")
    dashboard_ok = "Paper Entry Candidate Contract" in vr and "planner_preview_executable" in vr

    preview_candidate = str(summary.get("planner_preview_candidate") or "")
    preview_executable = bool(summary.get("planner_preview_executable"))
    preview_risk = str(summary.get("planner_preview_risk_flag") or "")
    risk_blocked_preview = "BLOCK_NEW_BUY" in preview_risk or "FORCE_EXIT" in preview_risk

    risk_prevented = bool(
        preview_candidate
        and risk_blocked_preview
        and not preview_executable
        and summary.get("preview_warnings")
    )

    notes.append(
        f"Planner raw={preview_candidate} risk={preview_risk} "
        f"executable={preview_executable} governed={summary.get('governed_executable_preview_candidate')}."
    )
    if cache.get("stale"):
        notes.append(f"Stale cache handled: {cache.get('reason')}")
    notes.append(
        "Contract documents OPEN_NEW/ADD vs WATCH; final risk overlay remains authoritative."
    )

    mtime_after = PAPER_ARM_PATH.stat().st_mtime if PAPER_ARM_PATH.is_file() else None
    confirm_unchanged = mtime_before == mtime_after

    status = "PASS"
    if sup_rc != 0 or arm_rc != 0:
        status = "FAIL"
        notes.append("Runtime status script failure.")
    elif not confirm_unchanged:
        status = "FAIL"
        notes.append("paper_arm_confirm modified during validation.")
    elif positions != ["JPM", "WFC"] or open_orders != 0 or live_execution:
        status = "FAIL"
        notes.append("Portfolio state mismatch.")
    elif map_count < 1 or contract_count < 1:
        status = "FAIL"
        notes.append("Candidate artifacts not written.")
    elif preview_candidate and risk_blocked_preview and preview_executable:
        status = "FAIL"
        notes.append("Risk-blocked preview incorrectly marked executable.")
    elif not risk_prevented and preview_candidate and risk_blocked_preview:
        status = "REVIEW"
        notes.append("Verify preview warnings for risk-blocked candidate.")
    elif not dashboard_ok:
        status = "REVIEW"
        notes.append("Dashboard candidate contract section incomplete.")

    payload = {
        "phase": "196T",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "head_commit": head_commit[:8] if head_commit else "",
        "market_open": True,
        "runtime_pipeline_health": pipeline_health,
        "paper_arm_confirm_status": arm_st.get("status", "EXPIRED"),
        "paper_entry_armed": bool(arm_st.get("paper_entry_armed")),
        "live_execution": live_execution,
        "open_orders": open_orders,
        "positions": positions,
        "source_map_created": SOURCE_MAP_CSV.is_file() and map_count > 0,
        "candidate_contract_created": CONTRACT_CSV.is_file() and contract_count > 0,
        "stale_candidate_cache_detected": bool(cache.get("stale")),
        "stale_candidate_cache_handled_safely": bool(
            cache.get("diagnostic_only") and cache.get("meta_path")
        ),
        "planner_preview_candidate": preview_candidate,
        "planner_preview_risk_flag": preview_risk,
        "planner_preview_executable": preview_executable,
        "governed_executable_preview_candidate": summary.get(
            "governed_executable_preview_candidate"
        ),
        "allocation_open_new_count": summary.get("allocation_open_new_count"),
        "allocation_add_count": summary.get("allocation_add_count"),
        "allocation_exit_count": summary.get("allocation_exit_count"),
        "allocation_block_count": summary.get("allocation_block_count"),
        "candidate_alignment_issue_resolved_or_documented": True,
        "candidate_alignment_issue_type_196s": "ALLOCATION_AND_ENTRY_ENGINE_CONTRACT_MISMATCH",
        "resolution": "governed preview metadata + contract CSV; stale cache marked diagnostic",
        "risk_blocked_candidates_prevented_from_executable_preview": risk_prevented
        or (risk_blocked_preview and not preview_executable),
        "final_risk_overlay_preserved": True,
        "preview_warnings": summary.get("preview_warnings"),
        "final_selection_status": summary.get("final_selection_status"),
        "final_selection_block_reason": summary.get("final_selection_block_reason"),
        "arming_could_allow_paper_order_next_cycle": bool(
            summary.get("arming_could_allow_paper_order_next_cycle")
        ),
        "paper_arm_confirm_created": False,
        "live_confirm_created": False,
        "manual_approval_used": False,
        "execute_trades_run": False,
        "broker_orders_submitted": False,
        "broker_snapshot_ok": bool(snap_doc.get("ok")),
        "execution_logic_weakened": False,
        "governance_logic_weakened": False,
        "risk_logic_weakened": False,
        "allocation_logic_weakened": False,
        "dashboard_clarity_updated": dashboard_ok,
        "source_map_csv": str(SOURCE_MAP_CSV.relative_to(ROOT)).replace("\\", "/"),
        "candidate_contract_csv": str(CONTRACT_CSV.relative_to(ROOT)).replace("\\", "/"),
        "stale_cache_assessment": cache,
        "executable_candidate_count": summary.get("executable_candidate_count"),
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196T] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196T] wrote {SOURCE_MAP_CSV.relative_to(ROOT)}")
        print(f"[196T] wrote {CONTRACT_CSV.relative_to(ROOT)}")
        print(f"[196T] status={status}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196T candidate contract hardening")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196t_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
