"""
phase_197s_fix_dispatch_interlock_runtime_import_validation.py
Phase 197S-FIX2 — Repair dispatch interlock NameError and validate imports.
"""

from __future__ import annotations

import argparse
import importlib
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
MARKET_OPEN_LOG_DIR = RESULTS / "scheduled_runs"
OUTPUT_JSON = RESULTS / "phase_197s_fix_dispatch_interlock_runtime_import_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "b8410beb"
PREVIOUS_PHASE = "197S"

CRASH_SIGNATURE = "NameError: name 'is_defensive_exit_manage_handoff' is not defined"


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


def _latest_market_log() -> Path | None:
    if not MARKET_OPEN_LOG_DIR.is_dir():
        return None
    files = sorted(
        MARKET_OPEN_LOG_DIR.glob("market_open_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _static_manage_positions_ok() -> bool:
    src = (ROOT / "services/manage_positions.py").read_text(encoding="utf-8")
    import_pos = src.find("from services.defensive_exit_dispatch_interlock import")
    call_pos = src.find("is_defensive_exit_manage_handoff(allowed)")
    if import_pos < 0 or call_pos < 0:
        return False
    block = src[import_pos:call_pos]
    return "is_defensive_exit_manage_handoff" in block


def _static_place_live_orders_ok() -> bool:
    src = (ROOT / "services/place_live_orders.py").read_text(encoding="utf-8")
    return (
        "from services.defensive_exit_dispatch_interlock import" in src
        and "assert_defensive_exit_dispatch_permitted" in src
        and "format_dispatch_interlock_block" in src
    )


def run_phase_197s_fix_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()

    import_check_manage = False
    import_check_place = False
    interlock_module_ok = False
    handoff_fn_ok = False
    dispatch_fn_ok = False

    try:
        mod = importlib.import_module("services.defensive_exit_dispatch_interlock")
        interlock_module_ok = True
        handoff_fn_ok = callable(getattr(mod, "is_defensive_exit_manage_handoff", None))
        dispatch_fn_ok = callable(getattr(mod, "evaluate_defensive_exit_dispatch_interlock", None))
    except Exception as exc:
        notes.append(f"interlock module import failed: {exc}")
        status = "FAIL"

    try:
        importlib.import_module("services.manage_positions")
        import_check_manage = True
    except Exception as exc:
        notes.append(f"manage_positions import failed: {exc}")
        status = "FAIL"
        import_check_manage = False

    try:
        importlib.import_module("services.place_live_orders")
        import_check_place = True
    except Exception as exc:
        notes.append(f"place_live_orders import failed: {exc}")
        status = "FAIL"
        import_check_place = False

    manage_static_ok = _static_manage_positions_ok()
    place_static_ok = _static_place_live_orders_ok()
    if not manage_static_ok:
        status = "FAIL"
        notes.append("manage_positions static import order invalid")
    if not place_static_ok:
        status = "FAIL"
        notes.append("place_live_orders static interlock references missing")

    latest_log = _latest_market_log()
    market_text = latest_log.read_text(encoding="utf-8", errors="replace") if latest_log else ""
    crash_detected = CRASH_SIGNATURE in market_text
    crash_repaired = handoff_fn_ok and manage_static_ok and import_check_manage

    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_BLOCKED_APPROVAL_REQUIRED,
        evaluate_defensive_exit_dispatch_interlock,
        format_dispatch_interlock_block,
        is_defensive_exit_manage_handoff,
    )
    from services.live_block_classifier import validator_live_block_check
    from types import SimpleNamespace

    planned = [SimpleNamespace(management_action="EXIT", side="sell", symbol="TEST")]
    handoff_detected = is_defensive_exit_manage_handoff(planned)
    sim = evaluate_defensive_exit_dispatch_interlock(
        broker_snapshot_ok=True,
        planned_count=1,
    )
    sim_marker = format_dispatch_interlock_block(sim, stage="simulation")
    sim_state = str(sim.get("dispatch_interlock_state") or "")

    live_block_ok, live_payload = validator_live_block_check(paper_mode=True)

    if not handoff_detected:
        status = "FAIL"
        notes.append("is_defensive_exit_manage_handoff simulation failed")
    if sim_state != DISPATCH_BLOCKED_APPROVAL_REQUIRED:
        status = "FAIL"
        notes.append(f"simulation expected {DISPATCH_BLOCKED_APPROVAL_REQUIRED} got {sim_state}")
    if DISPATCH_BLOCKED_APPROVAL_REQUIRED not in sim_marker:
        status = "FAIL"
        notes.append("simulation marker missing DISPATCH_BLOCKED_APPROVAL_REQUIRED")
    if "stage=simulation" not in sim_marker:
        status = "FAIL"
        notes.append("simulation marker missing stage=")
    if not live_block_ok:
        status = "FAIL"
        notes.append("live block regression")
    if REAL_APPROVAL_PATH.is_file():
        status = "FAIL"
        notes.append("real approval artifact exists")

    payload: Dict[str, Any] = {
        "phase": "197S-FIX2",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "manage_positions.maybe_execute_management referenced "
            "is_defensive_exit_manage_handoff before importing it, causing NameError at runtime."
        ),
        "fix": (
            "Import dispatch interlock helpers before use; add stage= markers; "
            "regression validation for import and missing-approval simulation."
        ),
        "import_check_manage_positions": import_check_manage,
        "import_check_place_live_orders": import_check_place,
        "interlock_module_import_ok": interlock_module_ok,
        "is_defensive_exit_manage_handoff_available": handoff_fn_ok,
        "dispatch_interlock_function_available": dispatch_fn_ok,
        "manage_positions_static_reference_ok": manage_static_ok,
        "place_live_orders_static_reference_ok": place_static_ok,
        "latest_runtime_crash_detected": crash_detected,
        "latest_runtime_crash_signature": CRASH_SIGNATURE if crash_detected else None,
        "latest_market_open_log": (
            str(latest_log.relative_to(ROOT)).replace("\\", "/") if latest_log else None
        ),
        "crash_repaired": crash_repaired,
        "simulated_missing_approval_interlock_state": sim_state,
        "simulated_missing_approval_marker": sim_marker,
        "broker_orders_submitted": False,
        "real_approval_artifact_exists": REAL_APPROVAL_PATH.is_file(),
        "live_blocked": live_payload.get("live_blocked"),
        "live_block_reason": live_payload.get("live_block_reason"),
        "open_orders_count": sim.get("open_orders_count"),
        "eligible_for_execution": False,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197S-FIX2] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197S-FIX2] status={status} crash_repaired={crash_repaired} "
            f"crash_in_log={crash_detected} sim_state={sim_state}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197S-FIX2 dispatch interlock import validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197s_fix_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
