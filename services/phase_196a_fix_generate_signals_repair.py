"""
phase_196a_fix_generate_signals_repair.py
Phase 196A-FIX — Record market-open signal refresh repair validation.
Read-only validation artifact writer; does not submit broker orders.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_196a_fix_generate_signals_repair.json"
EFFECTIVE_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"
SIGNALS_CSV = RESULTS / "signals.csv"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_module(mod: str) -> Tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", mod],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return proc.returncode, text[-2000:] if len(text) > 2000 else text


def _paper_entry_dry_run() -> Tuple[str, str]:
    env = os.environ.copy()
    env["TRITON_ENABLE_PAPER_EXECUTION"] = "1"
    env["TRITON_ENABLE_AUTO_PAPER_ENTRIES"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "services.paper_entry_loop_stage", "--verbose", "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    gate = "UNKNOWN"
    reason = ""
    for line in text.splitlines():
        if "[PAPER_ENTRY_GATE]" in line:
            if "status=" in line:
                gate = line.split("status=", 1)[-1].split()[0]
            if "reason=" in line:
                reason = line.split("reason=", 1)[-1].strip()
    if reason.endswith(" (SAFE_BLOCK)"):
        reason = reason[: -len(" (SAFE_BLOCK)")]
    return gate, reason


def run_validation(*, verbose: bool = True) -> Dict[str, Any]:
    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )

    gs_rc, gs_out = _run_module("services.generate_signals")
    paper_gate, paper_reason = _paper_entry_dry_run()
    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=True)
    lifecycle_refreshed = EFFECTIVE_LIFECYCLE.is_file() and EFFECTIVE_LIFECYCLE.stat().st_size > 0

    stale_block = (
        "stale lifecycle/signals" in paper_reason or "execution_readiness=NOT_READY" in paper_reason
    )

    status = "PASS"
    if gs_rc != 0:
        status = "FAIL"
    if stale_block:
        status = "FAIL"

    payload: Dict[str, Any] = {
        "phase": "196A-FIX",
        "generated_at": _utc_now_iso(),
        "status": status,
        "root_cause": (
            "services.preprocess was a no-op stub (no main), so repair chain never refreshed "
            "data/processed parquets; train_model used stale data/results parquets and "
            "generate_signals failed AS_OF checks. Windows cp1252 also crashed emoji prints."
        ),
        "files_changed": [
            "services/generate_signals.py",
            "services/preprocess.py",
            "services/phase_196a.py",
            "services/phase_196a_fix_generate_signals_repair.py",
        ],
        "generate_signals_before": "rc=1 (UnicodeEncodeError on Windows and/or stale predictions)",
        "generate_signals_after_rc": gs_rc,
        "generate_signals_after_tail": gs_out,
        "lifecycle_artifact_refreshed": lifecycle_refreshed,
        "signals_csv_exists": SIGNALS_CSV.is_file(),
        "paper_entry_dry_run_status": paper_gate,
        "paper_entry_dry_run_reason": paper_reason,
        "open_orders": int(criteria.get("open_orders_count") or 0),
        "live_blocked": bool(criteria.get("live_blocked")),
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "approval_artifact_created": False,
        "approval_artifact_exists": APPROVAL_PATH.is_file(),
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196A-FIX] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196A-FIX] status={status} generate_signals_rc={gs_rc} paper_gate={paper_gate}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 196A-FIX generate_signals repair validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
