"""
phase_196r_controlled_paper_arm_decision_packet_validation.py
Phase 196R — Build read-only controlled paper ARM operator decision packet.
Does not arm paper, submit orders, or run execute_trades.
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
OUTPUT_JSON = RESULTS / "phase_196r_controlled_paper_arm_decision_packet.json"

PREVIOUS_COMMIT = "84ba7d2d"
PREVIOUS_PHASE = "196Q"
PAPER_ARM_PATH = LIVE / "paper_arm_confirm.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


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
        if "heartbeat.json:" in line and "min" in line:
            try:
                part = line.split("~")[1].split("min")[0].strip()
                return float(part)
            except Exception:
                pass
    return None


def _analyze_entry_posture() -> Dict[str, Any]:
    from services.paper_arm_status import build_paper_arm_status
    from services.paper_entry_loop_stage import _load_entry_candidates

    audit = _load_json(RESULTS / "paper_entry_loop_stage_audit.json")
    alloc = _load_json(RESULTS / "portfolio_allocation_summary.json")
    cert = _load_json(RESULTS / "capital_preservation_certification.json")
    gov = _load_json(RESULTS / "governance_authorization.json")
    readiness = _load_json(RESULTS / "execution_readiness.json")

    candidates, _ = _load_entry_candidates()
    eligible = [c for c in candidates if c.get("eligible_for_auto_entry")]
    top_eligible = ""
    if eligible:
        ranked = sorted(
            eligible,
            key=lambda c: float(c.get("edge_score") or 0),
            reverse=True,
        )
        top_eligible = str(ranked[0].get("ticker") or "")

    planned_symbols: List[str] = []
    planner_meta: Dict[str, Any] = {}
    risk_block_after_arm: List[str] = []
    try:
        from services.execute_trades import preview_planned_buy_symbols

        planned_symbols, planner_meta = preview_planned_buy_symbols(
            "paper", verbose=False, for_execute=True
        )
        from services.paper_entry_loop_stage import _risk_blocked_in_selection

        risk_block_after_arm = _risk_blocked_in_selection(planned_symbols)
    except Exception as exc:
        planner_meta = {"error": str(exc)}

    remaining_after_arm: List[str] = []
    if risk_block_after_arm:
        remaining_after_arm.append(
            f"FINAL_SELECTION_RISK_OVERLAY_BLOCK:{','.join(risk_block_after_arm)}"
        )
    if str(cert.get("certification_status") or "").upper() == "NOT_CERTIFIED":
        remaining_after_arm.append("NOT_CERTIFIED (governance display; separate from arm latch)")
    remaining_after_arm.append("LIVE_EXECUTION_BLOCKED")
    if str(readiness.get("live_execution_permitted", True)):
        pass
    remaining_after_arm.append("live_confirm EXPIRED (live only)")

    could_order = bool(
        eligible
        and planned_symbols
        and not risk_block_after_arm
        and int(alloc.get("open_new_candidates") or 0) > 0
        or (
            eligible
            and planned_symbols
            and not risk_block_after_arm
            and int(alloc.get("add_candidates") or 0) > 0
        )
    )
    if eligible and planned_symbols and risk_block_after_arm:
        could_order = False
    elif eligible and not risk_block_after_arm and planned_symbols:
        could_order = True
    else:
        could_order = False

    arm_st = build_paper_arm_status(verbose_notes=False)

    overlay_status = "PASS"
    held_flags: List[str] = []
    try:
        import pandas as pd

        ro = pd.read_csv(RESULTS / "performance_risk_overlay.csv", low_memory=False)
        for sym in ("JPM", "WFC"):
            row = ro[ro["ticker"] == sym]
            if not row.empty:
                held_flags.append(f"{sym}={row.iloc[0].get('risk_flag', 'OK')}")
        if risk_block_after_arm:
            overlay_status = "BLOCK"
        elif not eligible:
            overlay_status = "REVIEW"
    except Exception:
        overlay_status = "REVIEW"

    return {
        "audit": audit,
        "alloc": alloc,
        "cert": cert,
        "gov": gov,
        "readiness": readiness,
        "arm_st": arm_st,
        "eligible_candidates": eligible,
        "top_eligible": top_eligible,
        "planned_symbols": planned_symbols,
        "planner_meta": planner_meta,
        "risk_block_after_arm": risk_block_after_arm,
        "remaining_after_arm": remaining_after_arm,
        "could_order": could_order,
        "overlay_status": overlay_status,
        "held_risk_flags": held_flags,
    }


def _extract_snapshot_json(stdout: str) -> Dict[str, Any]:
    marker = '{\n  "ok"'
    idx = stdout.find(marker)
    if idx < 0:
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith('{"ok"'):
                try:
                    return json.loads(line)
                except Exception:
                    pass
        return {}
    try:
        return json.loads(stdout[idx:])
    except Exception:
        return {}


def run_phase_196r_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []

    head = subprocess.run(
        ["git", "log", "-1", "--format=%H"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    head_commit = (head.stdout or "").strip()
    if not head_commit.startswith("84ba7d2d"):
        notes.append(f"HEAD={head_commit[:8]}; expected 84ba7d2d or later approved.")

    sup_rc, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    fresh_rc, fresh_out = _run_ps("show_triton_data_freshness_status.ps1")
    arm_rc, arm_out = _run_ps("show_triton_paper_arm_status.ps1")

    watchdog_on = "Watchdog process running: True" in sup_out
    loop_on = "Continuous loop process running: True" in sup_out
    hb_age = _parse_freshness_age(sup_out) or _parse_freshness_age(fresh_out)
    pipeline_health = (
        "HEALTHY" if watchdog_on and loop_on and (hb_age is None or hb_age < 45) else "DEGRADED"
    )
    if not watchdog_on or not loop_on:
        pipeline_health = "FAIL"

    fresh_age = _parse_freshness_age(fresh_out)
    data_freshness = "FRESH" if fresh_age is not None and fresh_age < 40 else "STALE"
    if fresh_age is None:
        data_freshness = "REVIEW"

    snap_proc = subprocess.run(
        [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    snap_ok = snap_proc.returncode == 0
    snap_doc = _extract_snapshot_json(snap_proc.stdout or "") if snap_ok else {}

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

    posture = _analyze_entry_posture()
    audit = posture["audit"]
    alloc = posture["alloc"]
    cert = posture["cert"]
    gov = posture["gov"]
    readiness = posture["readiness"]
    arm_st = posture["arm_st"]

    audit_details = audit.get("details") or {}
    audit_checks = audit_details.get("checks") or {}

    recommended = "DO_NOT_ARM_KEEP_OBSERVING"
    if posture["could_order"] and int(alloc.get("open_new_candidates") or 0) > 0:
        recommended = "ARM_STANDARD_WINDOW_480_MIN"
    elif posture["could_order"]:
        recommended = "ARM_SHORT_WINDOW_30_MIN"
    notes.append(
        f"Eligible ENTRY after arm preview: {[c.get('ticker') for c in posture['eligible_candidates']]}; "
        f"planner preview={posture['planned_symbols']}; "
        f"risk_block={posture['risk_block_after_arm']}."
    )
    notes.append(
        f"Allocation: open_new={alloc.get('open_new_candidates')} add={alloc.get('add_candidates')} "
        f"exit={alloc.get('exit_candidates')} block={alloc.get('blocked_candidates')}."
    )
    notes.append(
        f"Certification={cert.get('certification_status')} score={cert.get('certification_score')}; "
        f"readiness={readiness.get('readiness_status')}."
    )
    if fresh_out.find("positions []") >= 0 and positions:
        notes.append(
            "Freshness script reported empty positions; broker snapshot confirms "
            + ",".join(positions)
            + "."
        )

    mtime_after = PAPER_ARM_PATH.stat().st_mtime if PAPER_ARM_PATH.is_file() else None
    confirm_unchanged = mtime_before == mtime_after

    status = "PASS"
    if sup_rc != 0 or fresh_rc != 0 or arm_rc != 0:
        status = "FAIL"
        notes.append("Runtime status script failure.")
    elif not snap_ok or not snap_doc.get("ok"):
        status = "FAIL"
        notes.append("Broker snapshot failed.")
    elif not confirm_unchanged:
        status = "FAIL"
        notes.append("paper_arm_confirm modified during validation.")
    elif positions != ["JPM", "WFC"]:
        status = "FAIL"
        notes.append(f"Unexpected positions: {positions}")
    elif open_orders != 0 or live_execution:
        status = "FAIL"
        notes.append("Open orders or live_execution unexpected.")
    elif pipeline_health == "FAIL":
        status = "REVIEW"
        notes.append("Runtime pipeline not fully healthy.")

    payload = {
        "phase": "196R",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "head_commit": head_commit[:8] if head_commit else "",
        "market_open": bool(audit_checks.get("market_open", True)),
        "runtime_pipeline_health": pipeline_health,
        "data_freshness": data_freshness,
        "watchdog_running": watchdog_on,
        "continuous_loop_running": loop_on,
        "paper_arm_mode": arm_st.get("mode", "ASSISTED"),
        "paper_arm_confirm_status": arm_st.get("status", "EXPIRED"),
        "paper_entry_armed": bool(arm_st.get("paper_entry_armed")),
        "paper_arm_confirm_created": False,
        "live_confirm_created": False,
        "live_execution": live_execution,
        "open_orders": open_orders,
        "positions": positions,
        "broker_snapshot_ok": bool(snap_doc.get("ok")),
        "execution_readiness": readiness.get("readiness_status", "REVIEW"),
        "governance_authorized": bool(gov.get("governance_authorized")),
        "certification_status": cert.get("certification_status", "REVIEW"),
        "paper_entry_latest_result": "BLOCKED" if not audit.get("gate_pass") else "REVIEW",
        "paper_entry_latest_block_reason": str(audit.get("gate_reason") or ""),
        "paper_entry_audit_timestamp": audit.get("timestamp"),
        "paper_entry_selected_candidate": None,
        "paper_entry_final_candidate": (
            (audit_checks.get("final_selected_symbols") or [None])[0]
            if audit_checks.get("final_selected_symbols")
            else None
        ),
        "paper_entry_certification_checked": False,
        "paper_entry_risk_overlay_checked": bool(
            audit_checks.get("risk_blocked_in_final_selection")
        ),
        "paper_entry_order_submission_attempted": bool(audit.get("executed")),
        "paper_entry_broker_order_submitted": False,
        "current_open_new_count": int(alloc.get("open_new_candidates") or 0),
        "current_add_count": int(alloc.get("add_candidates") or 0),
        "current_exit_count": int(alloc.get("exit_candidates") or 0),
        "current_block_count": int(alloc.get("blocked_candidates") or 0),
        "top_candidate": posture["top_eligible"]
        or (posture["planned_symbols"][0] if posture["planned_symbols"] else ""),
        "eligible_entry_candidates": [c.get("ticker") for c in posture["eligible_candidates"]],
        "planner_preview_symbols": posture["planned_symbols"],
        "held_position_risk_flags": posture["held_risk_flags"],
        "final_risk_overlay_status": posture["overlay_status"],
        "arming_could_allow_paper_order_next_cycle": posture["could_order"],
        "remaining_blocks_after_arm": posture["remaining_after_arm"],
        "operator_decision_required": True,
        "recommended_operator_options": [
            "DO_NOT_ARM_KEEP_OBSERVING",
            "ARM_SHORT_WINDOW_30_MIN",
            "ARM_STANDARD_WINDOW_480_MIN",
        ],
        "recommended_option": recommended,
        "manual_execute_trades_run": False,
        "manual_approval_used": False,
        "broker_orders_submitted": False,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "certification_logic_changed": False,
        "paper_arm_status": arm_st,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196R] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196R] status={status} recommended={recommended}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196R paper arm decision packet")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196r_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
