"""
phase_197m_fix_live_block_detection_validation.py
Phase 197M-FIX — Repair stale live confirm handling and live block detection.
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
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197m_fix_live_block_detection_validation.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "32bf210a"
PREVIOUS_PHASE = "197M"


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


def _count_positions() -> int:
    path = RESULTS / "positions_snapshot.csv"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _count_open_orders() -> int:
    path = RESULTS / "open_orders_snapshot.csv"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def run_phase_197m_fix_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()

    from services.defensive_exit_governance_sync import build_defensive_exit_approval_sync
    from services.live_block_classifier import (
        classify_paper_entry_gate_for_validation,
        evaluate_live_block_status,
        validator_live_block_check,
        validator_paper_gate_acceptable,
    )
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.triton_env_loader import load_cli_validation_env

    env_authority = load_cli_validation_env()

    live_block_ok, live_payload = validator_live_block_check(paper_mode=True)
    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    gate_acceptable, gate_classification = validator_paper_gate_acceptable(
        gate_ok, gate_reason, gate_details
    )
    checks = gate_details.get("checks") or {}
    market_open = bool(checks.get("market_open", False))

    sync = build_defensive_exit_approval_sync(cap_context="runtime")

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

    if not live_block_ok:
        status = "FAIL"
        notes.append("FAIL_LIVE_BLOCK_REGRESSION live_blocked=False")
    if live_payload.get("live_block_reason") not in (
        "LIVE_CONFIRM_STALE_OR_EXPIRED",
        "LIVE_ENV_NOT_ENABLED",
        "LIVE_EXECUTION_NOT_PERMITTED_BY_READINESS",
        "PAPER_MODE_LIVE_BLOCKED",
        "LIVE_CONFIRM_ABSENT_OR_INVALID",
        "LIVE_BLOCKED_BY_POLICY",
    ):
        notes.append(f"unexpected live_block_reason={live_payload.get('live_block_reason')}")
    if live_payload.get("stale_live_confirm_count", 0) < 1:
        notes.append("expected stale live_confirm artifacts on disk")
    if live_payload.get("valid_live_confirm_count", 0) != 0:
        status = "FAIL"
        notes.append("valid live confirm must not authorize in current state")
    if env_authority.get("live_execution_enabled") or env_authority.get("live_force_enabled"):
        status = "FAIL"
        notes.append("live execution env must remain disabled in validation")
    if REAL_APPROVAL_PATH.is_file():
        status = "FAIL"
        notes.append("real approval artifact exists")
    if gate_ok:
        status = "FAIL"
        notes.append("paper entry gate unexpectedly open")
    if not gate_acceptable:
        status = "FAIL"
        notes.append(f"paper entry gate not acceptable: {gate_classification}")

    payload: Dict[str, Any] = {
        "phase": "197M-FIX",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "Phase validators derived live_blocked from paper_entry_gate checks with "
            "live_execution_permitted defaulting to True when gate failed early (market closed), "
            "reporting live_blocked=False despite stale live_confirm artifacts and paper-only mode."
        ),
        "fix": (
            "live_block_classifier centralizes live confirm TTL/stale detection and "
            "conservative live_blocked evaluation independent of paper entry gate early exits."
        ),
        "live_blocked": live_payload.get("live_blocked"),
        "live_block_reason": live_payload.get("live_block_reason"),
        "live_confirm_paths_found": live_payload.get("live_confirm_paths_found"),
        "live_confirm_status_by_path": live_payload.get("live_confirm_status_by_path"),
        "stale_live_confirm_count": live_payload.get("stale_live_confirm_count"),
        "valid_live_confirm_count": live_payload.get("valid_live_confirm_count"),
        "live_env_flags": live_payload.get("live_env_flags"),
        "paper_env_flags": {
            k: v for k, v in (live_payload.get("live_env_flags") or {}).items() if "PAPER" in k
        },
        "paper_mode": True,
        "market_open": market_open,
        "paper_entry_gate_status": classify_paper_entry_gate_for_validation(
            gate_ok, gate_reason, gate_details
        ),
        "paper_entry_gate_ok": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "paper_entry_gate_acceptable": gate_acceptable,
        "env_authority_state": env_authority.get("env_authority_state"),
        "env_source": env_authority.get("env_source"),
        "cli_validation_env_loaded": env_authority.get("cli_validation_env_loaded"),
        "paper_execution_enabled": env_authority.get("paper_execution_enabled"),
        "live_execution_enabled": env_authority.get("live_execution_enabled"),
        "broker_snapshot_ok": snapshot_ok,
        "open_orders_count": _count_open_orders(),
        "positions_count": _count_positions(),
        "broker_orders_submitted": False,
        "real_approval_artifact_exists": REAL_APPROVAL_PATH.is_file(),
        "governance_authorized": sync.get("governance_authorized"),
        "readiness_status": sync.get("readiness_status"),
        "live_block_payload": live_payload,
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197M-FIX] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197M-FIX] status={status} live_blocked={live_payload.get('live_blocked')} "
            f"reason={live_payload.get('live_block_reason')} gate={gate_classification}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197M-FIX live block detection validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197m_fix_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
