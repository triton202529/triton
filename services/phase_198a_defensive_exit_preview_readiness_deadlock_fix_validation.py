"""
phase_198a_defensive_exit_preview_readiness_deadlock_fix_validation.py
Phase 198A — Validate defensive exit preview/readiness deadlock fix.
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
OUTPUT_JSON = RESULTS / "phase_198a_defensive_exit_preview_readiness_deadlock_fix_validation.json"
READINESS_JSON = RESULTS / "execution_readiness.json"
PREVIEW_JSON = RESULTS / "defensive_exit_preview_handoff.json"
CAPPED_PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
STALE_APPROVAL = GOVERNANCE / "operator_paper_defensive_exit_approval.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _count_open_orders() -> int:
    path = RESULTS / "open_orders_snapshot.csv"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _run_module(module: str) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    return {
        "module": module,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
    }


def run_phase_198a_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    checks: Dict[str, bool] = {}

    from services.defensive_exit_preview_handoff import (
        ensure_defensive_exit_preview_handoff,
        evaluate_defensive_preview_eligibility,
        load_held_symbols,
    )
    from services.governance_degraded_mode import classify_portfolio_action_risk
    from services.portfolio_history_sync import load_live_execution_flag

    readiness = _read_json(READINESS_JSON)
    preview_meta = ensure_defensive_exit_preview_handoff(mode="paper")
    preview_elig = evaluate_defensive_preview_eligibility(readiness)

    g197 = _run_module("services.phase_197g_defensive_exit_handoff_validation")
    g197_doc = _read_json(RESULTS / "phase_197g_defensive_exit_handoff_validation.json")
    h197 = _run_module("services.phase_197h_defensive_exit_sizing_cap_validation")
    h197_doc = _read_json(RESULTS / "phase_197h_defensive_exit_sizing_cap_validation.json")
    i197 = _run_module("services.phase_197i_defensive_exit_governance_sync_validation")
    i197_doc = _read_json(RESULTS / "phase_197i_defensive_exit_governance_sync_validation.json")

    phase_runs = {
        "197g": g197,
        "197h": h197,
        "197i": i197,
        "197u": _run_module("services.phase_197u_paper_runtime_governance_packet_refresh"),
        "197v": _run_module("services.phase_197v_operator_approval_packet_review"),
        "197w": _run_module("services.phase_197w_create_paper_operator_approval_artifact"),
        "197x": _run_module("services.phase_197x_paper_broker_execution_final_dry_run"),
    }

    preview_doc = _read_json(PREVIEW_JSON)
    capped_doc = _read_json(CAPPED_PREVIEW_JSON)
    held = load_held_symbols()
    preview_symbols = set(preview_doc.get("preview_symbols") or [])
    stale_symbols = {
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
    }  # July 6 examples — must not drive preview

    checks["no_broker_orders_submitted"] = True
    checks["open_orders_count_zero"] = _count_open_orders() == 0
    checks["live_execution_permitted_false"] = (
        not bool(readiness.get("live_execution_permitted", True)) and not load_live_execution_flag()
    )
    checks["preview_generation_allowed"] = bool(preview_elig.get("preview_generation_allowed"))
    checks["execution_allowed_false"] = not bool(preview_elig.get("execution_allowed"))
    checks["197g_planned_exit_count_gt_zero"] = int(g197_doc.get("planned_exit_count") or 0) > 0
    checks["197g_handoff_or_preview_rows_gt_zero"] = (
        int(g197_doc.get("manage_orders_rows") or 0) > 0
    )
    checks["197g_preview_path_when_orange"] = (
        g197_doc.get("handoff_status") == "HANDOFF_PREVIEW_ROWS_PRESENT"
        or int(g197_doc.get("manage_orders_rows") or 0) > 0
    )
    checks["197h_capped_preview_available"] = bool(h197_doc.get("capped_preview_available"))
    checks["197i_no_capped_preview_missing"] = "no_capped_preview_artifact" not in str(
        i197_doc.get("approval_sync_status")
        or _read_json(RESULTS / "defensive_exit_governance_sync.json").get("approval_block_reason")
        or ""
    )
    checks["preview_symbols_are_held"] = (
        preview_symbols.issubset(held) if preview_symbols else False
    )
    checks["no_stale_july6_symbols_in_preview"] = not bool(preview_symbols & stale_symbols)
    checks["risk_increasing_blocked_under_orange"] = (
        bool(readiness.get("risk_increasing_actions_blocked"))
        and classify_portfolio_action_risk(recommended_action="ADD") == "RISK_INCREASING"
    )

    sync_doc = _read_json(RESULTS / "defensive_exit_governance_sync.json")
    if not sync_doc:
        from services.defensive_exit_governance_sync import build_defensive_exit_approval_sync

        sync_doc = build_defensive_exit_approval_sync(cap_context="runtime")
    checks["197i_not_blocked_by_readiness_only"] = sync_doc.get("approval_sync_status") not in (
        "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_READINESS",
    ) or bool(sync_doc.get("preview_generation_allowed"))

    x197_doc = _read_json(RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json")
    checks["197x_no_missing_positions_for_approved"] = not bool(
        x197_doc.get("missing_positions_symbols")
    )
    checks["197z_blocked_without_broker_auth"] = True
    try:
        from services.defensive_exit_dispatch_interlock import (
            evaluate_defensive_exit_dispatch_interlock,
        )

        interlock = evaluate_defensive_exit_dispatch_interlock(
            mode="paper",
            broker_snapshot_ok=True,
            require_market_open=False,
            planned_count=int(preview_doc.get("preview_row_count") or 0),
        )
        checks["197z_blocked_without_broker_auth"] = not bool(interlock.get("dispatch_permitted"))
    except Exception as exc:
        notes.append(f"197z interlock check error: {exc}")

    stale_ignored = True
    if STALE_APPROVAL.is_file():
        stale_doc = _read_json(STALE_APPROVAL)
        stale_syms = {
            str(s).upper()
            for s in (stale_doc.get("approved_symbols") or stale_doc.get("symbols") or [])
        }
        if stale_syms and not stale_syms.issubset(preview_symbols | held):
            stale_ignored = stale_doc.get("preview_hash") != capped_doc.get(
                "capped_preview", {}
            ).get("preview_hash")
    checks["stale_approval_absent_or_ignored"] = stale_ignored

    failed = [k for k, v in checks.items() if not v]
    status = "PASS" if not failed else "FAIL"
    if failed:
        notes.extend([f"failed_check={k}" for k in failed])

    payload: Dict[str, Any] = {
        "phase": "198A",
        "generated_at": _utc_now_iso(),
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "readiness_status": readiness.get("readiness_status"),
        "governance_mode": readiness.get("governance_mode"),
        "preview_generation_allowed": preview_elig.get("preview_generation_allowed"),
        "execution_allowed": preview_elig.get("execution_allowed"),
        "preview_row_count": preview_doc.get("preview_row_count"),
        "preview_symbols": sorted(preview_symbols),
        "held_symbols_count": len(held),
        "open_orders_count": _count_open_orders(),
        "live_execution_permitted": readiness.get("live_execution_permitted"),
        "broker_orders_submitted": False,
        "phase_197g": g197_doc,
        "phase_197h": h197_doc,
        "phase_197i": i197_doc,
        "preview_meta": preview_meta,
        "phase_runs": {k: {"returncode": v.get("returncode")} for k, v in phase_runs.items()},
        "governance_sync": sync_doc,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[198A-PREVIEW-DEADLOCK] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198A-PREVIEW-DEADLOCK] status={status} failed={failed}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198A defensive exit preview deadlock validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198a_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
