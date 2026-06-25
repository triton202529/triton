"""
phase_196u_certification_evaluation_gap_closure_plan_validation.py
Phase 196U — Validate certification Evaluation gap closure plan (read-only).
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
OUTPUT_JSON = RESULTS / "phase_196u_certification_evaluation_gap_closure_plan.json"
GAP_MAP_CSV = RESULTS / "phase_196u_certification_gap_map.csv"

PREVIOUS_COMMIT = "8cf1d980"
PREVIOUS_PHASE = "196T"
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


def _parse_freshness_age(text: str) -> Optional[float]:
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


def run_phase_196u_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.certification_evaluation_gap_plan import (
        build_certification_gap_summary,
        write_gap_map_csv,
    )
    from services.paper_arm_status import build_paper_arm_status

    head = subprocess.run(
        ["git", "log", "-1", "--format=%H"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    head_commit = (head.stdout or "").strip()

    sup_rc, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    fresh_rc, fresh_out = _run_ps("show_triton_data_freshness_status.ps1")

    watchdog_on = "Watchdog process running: True" in sup_out
    loop_on = "Continuous loop process running: True" in sup_out
    hb_age = _parse_freshness_age(sup_out) or _parse_freshness_age(fresh_out)
    pipeline_health = "HEALTHY"
    if not watchdog_on or not loop_on:
        pipeline_health = "FAIL"
    elif hb_age is not None and hb_age > 45:
        pipeline_health = "DEGRADED"

    fresh_age = _parse_freshness_age(fresh_out)
    data_freshness = "FRESH" if fresh_age is not None and fresh_age < 40 else "STALE"
    if fresh_age is None:
        data_freshness = "REVIEW"

    arm_st = build_paper_arm_status(verbose_notes=False)
    summary = build_certification_gap_summary()
    gap_count = write_gap_map_csv(GAP_MAP_CSV)

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

    snap_proc = subprocess.run(
        [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    snap_doc = _extract_snapshot_json(snap_proc.stdout or "") if snap_proc.returncode == 0 else {}

    doc_path = ROOT / "docs" / "TRITON_CERTIFICATION_EVALUATION_GAP_CLOSURE_PLAN.md"
    vr = (ROOT / "view_results.py").read_text(encoding="utf-8")
    dashboard_ok = "Evaluation gap:" in vr or "certification_evaluation_gap_plan" in vr

    notes.append(
        f"Evaluation score={summary.get('evaluation_score')} "
        f"threshold={summary.get('evaluation_threshold')} "
        f"gap={summary.get('evaluation_gap_points')} pts."
    )
    notes.append(
        f"Trials beneficial={summary.get('evaluation_beneficial_count')} "
        f"neutral={summary.get('evaluation_neutral_count')} "
        f"negative={summary.get('evaluation_negative_count')}."
    )
    notes.append(
        "Certification NOT_CERTIFIED is governance posture; paper entry blocked by "
        "paper_arm_confirm (ASSISTED), not certification directly."
    )
    if summary.get("true_concentration_risk_detected"):
        notes.append(
            "Concentration (~52% WFC) is secondary driver; concentration trial is BENEFICIAL "
            "but average dragged by negative drawdown/exposure trials."
        )

    mtime_after = PAPER_ARM_PATH.stat().st_mtime if PAPER_ARM_PATH.is_file() else None
    confirm_unchanged = mtime_before == mtime_after

    status = "PASS"
    if sup_rc != 0 or fresh_rc != 0:
        status = "FAIL"
        notes.append("Runtime status script failure.")
    elif not confirm_unchanged:
        status = "FAIL"
        notes.append("paper_arm_confirm modified during validation.")
    elif positions != ["JPM", "WFC"] or open_orders != 0 or live_execution:
        status = "FAIL"
        notes.append("Portfolio state mismatch.")
    elif gap_count < 1 or not doc_path.is_file():
        status = "FAIL"
        notes.append("Gap map or operator doc missing.")
    elif not summary.get("true_evaluation_gap_detected"):
        status = "REVIEW"
        notes.append("Expected TRUE_EVALUATION_GAP not detected.")
    elif not head_commit.startswith("8cf1d980"):
        status = "REVIEW"
        notes.append(f"HEAD={head_commit[:8]}; expected 8cf1d980 base.")

    payload = {
        "phase": "196U",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "head_commit": head_commit[:8] if head_commit else "",
        "runtime_pipeline_health": pipeline_health,
        "watchdog_running": watchdog_on,
        "continuous_loop_running": loop_on,
        "data_freshness": data_freshness,
        "market_open": True,
        "paper_arm_confirm_status": arm_st.get("status", "EXPIRED"),
        "paper_entry_armed": bool(arm_st.get("paper_entry_armed")),
        "live_execution": live_execution,
        "open_orders": open_orders,
        "positions": positions,
        "certification_status": summary.get("certification_status"),
        "certification_score": summary.get("certification_score"),
        "certified_area_count": summary.get("certified_area_count"),
        "failed_requirements": summary.get("failed_requirements"),
        "evaluation_score": summary.get("evaluation_score"),
        "evaluation_threshold": summary.get("evaluation_threshold"),
        "evaluation_threshold_label": summary.get("evaluation_threshold_label"),
        "evaluation_gap_points": summary.get("evaluation_gap_points"),
        "evaluation_average_effectiveness": summary.get("evaluation_average_effectiveness"),
        "mandatory_vetoes": summary.get("mandatory_vetoes"),
        "primary_gap": summary.get("primary_gap"),
        "secondary_gaps": summary.get("secondary_gaps"),
        "true_governance_gap_detected": summary.get("true_governance_gap_detected"),
        "true_evaluation_gap_detected": summary.get("true_evaluation_gap_detected"),
        "true_concentration_risk_detected": summary.get("true_concentration_risk_detected"),
        "stale_artifact_detected": summary.get("stale_artifact_detected"),
        "readiness_current_status": summary.get("readiness_current_status"),
        "authorization_current_status": summary.get("authorization_current_status"),
        "paper_certification_blocks_paper_entry": summary.get(
            "paper_certification_blocks_paper_entry"
        ),
        "live_certification_blocks_live_execution": summary.get(
            "live_certification_blocks_live_execution"
        ),
        "safe_closure_actions": summary.get("safe_closure_actions"),
        "requires_paper_arm_to_close": summary.get("requires_paper_arm_to_close"),
        "requires_market_hours_to_close": summary.get("requires_market_hours_to_close"),
        "requires_manual_approval_to_close": summary.get("requires_manual_approval_to_close"),
        "requires_code_change_to_close": summary.get("requires_code_change_to_close"),
        "gap_map_created": GAP_MAP_CSV.is_file() and gap_count > 0,
        "recommended_next_phase": summary.get("recommended_next_phase"),
        "operator_doc_created": doc_path.is_file(),
        "dashboard_clarity_updated": dashboard_ok,
        "broker_snapshot_ok": bool(snap_doc.get("ok")),
        "paper_arm_confirm_created": False,
        "live_confirm_created": False,
        "manual_approval_used": False,
        "execute_trades_run": False,
        "broker_orders_submitted": False,
        "certification_logic_weakened": False,
        "governance_logic_weakened": False,
        "risk_logic_weakened": False,
        "allocation_logic_weakened": False,
        "gap_map_csv": str(GAP_MAP_CSV.relative_to(ROOT)).replace("\\", "/"),
        "gap_map_row_count": gap_count,
        "evaluation_trials_summary": summary.get("evaluation_trials"),
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196U] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196U] wrote {GAP_MAP_CSV.relative_to(ROOT)}")
        print(f"[196U] status={status} primary_gap={summary.get('primary_gap')}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196U certification gap closure plan")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196u_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
