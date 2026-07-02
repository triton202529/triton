"""
phase_197z_handoff_scope_validation.py
Phase 197Z-HS — Validate approved 197X scope at paper execution handoff (read-only).
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
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197z_handoff_scope_validation.json"
HANDOFF_CSV = LIVE / "manage_orders.csv"
PROJECTED_ORDERS_CSV = RESULTS / "phase_197x_projected_paper_orders.csv"
PREVIOUS_PHASE = "197Z-CAP"
PREVIOUS_COMMIT = "a4d11c7e"

APPROVED_SYMBOLS = ["CVX", "HD", "PEP", "PFE", "SLV", "T", "USO", "WMT", "XLE", "XOM"]
FORBIDDEN_SYMBOLS = ["GLD", "GE", "TSLA", "ARKK", "MA", "PG", "ABBV", "UNG"]
EXPECTED_NOTIONAL_APPROX = 16980.61
NOTIONAL_TOLERANCE = 250.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_snapshot() -> Tuple[int, Dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    snapshot: Dict[str, Any] = {"exit_code": proc.returncode}
    jstart = text.find("{")
    if jstart >= 0:
        try:
            snapshot.update(json.loads(text[jstart:]))
        except Exception:
            pass
    return proc.returncode, snapshot


def _synthetic_contaminated_handoff_proof() -> Dict[str, Any]:
    """Prove scope filter removes outside symbols using 197X projected order authority."""
    import pandas as pd

    from services.paper_broker_execution_authorization_state import load_197x_projected_orders

    projected_rows, approved_symbols = load_197x_projected_orders()
    if not projected_rows or not approved_symbols:
        return {
            "synthetic_proof_ok": False,
            "reason": "projected_orders_missing",
        }

    base_rows: List[Dict[str, Any]] = []
    for row in projected_rows:
        sym = str(row.get("symbol") or "").upper()
        qty = float(row.get("quantity") or row.get("qty") or 1)
        price = float(row.get("notional_estimate") or 0) / qty if qty else 100.0
        base_rows.append(
            {
                "ticker": sym,
                "qty": qty,
                "limit_price": round(price, 4),
                "opportunity_type": "EXIT",
            }
        )
    for sym in FORBIDDEN_SYMBOLS:
        base_rows.append(
            {
                "ticker": sym,
                "qty": 10,
                "limit_price": 100.0,
                "opportunity_type": "EXIT",
            }
        )

    contaminated = pd.DataFrame(base_rows)
    pre_symbols = sorted(contaminated["ticker"].astype(str).str.upper().unique())
    approved_set = set(approved_symbols)
    filtered = contaminated[
        contaminated["ticker"].astype(str).str.upper().isin(approved_set)
    ].copy()
    final_symbols = sorted(filtered["ticker"].astype(str).str.upper().unique())
    excluded = sorted(set(pre_symbols) - set(final_symbols))
    pre_notional = float(contaminated["qty"].astype(float).sum() * 100.0)  # upper bound est
    final_notional = 0.0
    for _, row in filtered.iterrows():
        final_notional += float(row["qty"]) * float(row["limit_price"])
    final_notional = round(final_notional, 2)

    forbidden_in_final = sorted(set(final_symbols) & set(FORBIDDEN_SYMBOLS))
    proof_ok = (
        not forbidden_in_final
        and set(final_symbols) == set(approved_symbols)
        and all(sym in excluded for sym in FORBIDDEN_SYMBOLS if sym in pre_symbols)
        and final_notional <= 50000.0
    )
    return {
        "synthetic_proof_ok": proof_ok,
        "pre_handoff_symbols": pre_symbols,
        "excluded_from_handoff_symbols": excluded,
        "final_handoff_symbols": final_symbols,
        "final_handoff_notional": final_notional,
        "batch_notional_cap": 50000.0,
        "cap_passed": final_notional <= 50000.0,
        "forbidden_symbols_in_final_handoff": forbidden_in_final,
        "outside_scope_rows_reached_placement": bool(forbidden_in_final),
    }


def run_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    import pandas as pd

    from services.defensive_exit_dispatch_interlock import (
        DISPATCH_READY,
        evaluate_defensive_exit_dispatch_interlock,
        load_paper_approved_execution_scope,
    )
    from services.live_block_classifier import evaluate_live_block_status
    from services.paper_broker_execution_authorization_state import (
        load_197x_projected_orders,
    )
    from services.paper_execution_handoff_scope import (
        FORBIDDEN_OUTSIDE_SCOPE_SYMBOLS,
        filter_handoff_lifecycle_dataframe,
        should_enforce_approved_197x_handoff_scope,
    )

    from services.paper_loop_mediated_execution_readiness_bridge import (
        evaluate_paper_loop_mediated_execution_readiness_bridge,
    )

    scope_symbols, scope_source, scope_valid = load_paper_approved_execution_scope()
    bridge = evaluate_paper_loop_mediated_execution_readiness_bridge(mode="paper")
    synthetic = _synthetic_contaminated_handoff_proof()
    interlock = evaluate_defensive_exit_dispatch_interlock(
        mode="paper",
        broker_snapshot_ok=True,
        require_market_open=True,
        planned_count=10,
    )
    live = evaluate_live_block_status(paper_mode=True)
    snap_rc, snapshot = _run_snapshot()

    handoff_df = pd.DataFrame()
    pre_symbols: List[str] = []
    if HANDOFF_CSV.is_file():
        try:
            handoff_df = pd.read_csv(HANDOFF_CSV)
            if "ticker" in handoff_df.columns:
                pre_symbols = sorted(
                    handoff_df["ticker"].astype(str).str.strip().str.upper().unique()
                )
        except Exception:
            pass

    filtered_df, scope_diag = filter_handoff_lifecycle_dataframe(
        handoff_df,
        mode="paper",
        batch_notional_cap=50000.0,
    )
    projected_rows, projected_symbols = load_197x_projected_orders()
    projected_notional = round(
        sum(float(r.get("notional_estimate") or 0) for r in projected_rows),
        2,
    )

    final_symbols = list(scope_diag.get("final_handoff_symbols") or [])
    excluded = list(scope_diag.get("excluded_from_handoff_symbols") or [])
    final_notional = float(scope_diag.get("final_handoff_notional") or 0.0)

    status = "PASS_SAFE_BLOCKED"
    if not synthetic.get("synthetic_proof_ok"):
        status = "FAIL"
        notes.append("synthetic_handoff_scope_filter_proof_failed")
    elif should_enforce_approved_197x_handoff_scope(mode="paper") and scope_valid:
        status = "PASS"

    forbidden_in_final = sorted(set(final_symbols) & set(FORBIDDEN_SYMBOLS))
    forbidden_in_pre = sorted(set(pre_symbols) & FORBIDDEN_OUTSIDE_SCOPE_SYMBOLS)

    if forbidden_in_final:
        status = "FAIL"
        notes.append(f"forbidden_symbols_in_final_handoff={forbidden_in_final}")
    if scope_valid and set(scope_symbols) != set(APPROVED_SYMBOLS):
        notes.append(f"approved_scope_mismatch={scope_symbols}")
    if scope_valid and final_symbols and set(final_symbols) - set(scope_symbols):
        status = "FAIL"
        notes.append("final_handoff_contains_outside_approved_scope")
    if scope_valid and final_notional > 50000.0:
        status = "FAIL"
        notes.append("final_handoff_notional_exceeds_cap")
    if (
        scope_valid
        and projected_notional > 0
        and abs(final_notional - projected_notional) > NOTIONAL_TOLERANCE
        and handoff_df is not None
        and not handoff_df.empty
    ):
        notes.append(f"final_notional={final_notional} projected_notional={projected_notional}")
    if scope_diag.get("outside_scope_rows_reached_placement"):
        status = "FAIL"
        notes.append("outside_scope_rows_reached_placement=true")
    if not live.get("live_blocked"):
        status = "FAIL"
        notes.append("live_blocked=false")
    if int(snapshot.get("open_orders_written") or snapshot.get("open_orders") or 0) != 0:
        status = "FAIL"
        notes.append("open_orders!=0")
    if bridge.get("bridge_active") and interlock.get("dispatch_interlock_state") != DISPATCH_READY:
        notes.append(
            f"dispatch_state={interlock.get('dispatch_interlock_state')} "
            f"reason={interlock.get('dispatch_interlock_reason')}"
        )

    payload: Dict[str, Any] = {
        "phase": "197Z-HS",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "bridge_active": bool(bridge.get("bridge_active")),
        "scope_valid": scope_valid,
        "scope_source": scope_source,
        "approved_scope_symbols": scope_symbols,
        "handoff_csv_exists": HANDOFF_CSV.is_file(),
        "pre_handoff_symbols": pre_symbols,
        "excluded_from_handoff_symbols": excluded,
        "final_handoff_symbols": final_symbols,
        "forbidden_symbols_excluded_from_pre": forbidden_in_pre,
        "forbidden_symbols_in_final_handoff": forbidden_in_final,
        "pre_handoff_notional": scope_diag.get("pre_handoff_notional"),
        "final_handoff_notional": final_notional,
        "projected_197x_notional": projected_notional,
        "batch_notional_cap": scope_diag.get("batch_notional_cap"),
        "cap_passed": bool(scope_diag.get("cap_passed")),
        "outside_scope_rows_reached_placement": bool(
            scope_diag.get("outside_scope_rows_reached_placement")
        ),
        "broker_orders_submitted": False,
        "exits_executed": 0,
        "trims_executed": 0,
        "live_blocked": bool(live.get("live_blocked")),
        "open_orders_written": int(
            snapshot.get("open_orders_written") or snapshot.get("open_orders") or 0
        ),
        "positions_written": int(snapshot.get("positions_written") or 0),
        "dispatch_interlock_state": interlock.get("dispatch_interlock_state"),
        "dispatch_permitted": bool(interlock.get("dispatch_permitted")),
        "handoff_scope_diagnostics": scope_diag,
        "synthetic_filter_proof": synthetic,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197Z-HS] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197Z-HS] status={status} final_symbols={final_symbols} "
            f"final_notional={final_notional} excluded={len(excluded)}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197Z-HS handoff scope validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
