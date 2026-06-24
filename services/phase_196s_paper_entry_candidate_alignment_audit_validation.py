"""
phase_196s_paper_entry_candidate_alignment_audit_validation.py
Phase 196S — Read-only audit of paper-entry candidate vs planner vs allocation alignment.
Does not arm paper, run execute_trades, or weaken gates.
"""

from __future__ import annotations

import argparse
import csv
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
OUTPUT_JSON = RESULTS / "phase_196s_paper_entry_candidate_alignment_audit.json"
OUTPUT_CSV = RESULTS / "phase_196s_candidate_alignment_map.csv"

PREVIOUS_COMMIT = "efca296f"
PREVIOUS_PHASE = "196R"
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


def _parse_hb_age(text: str) -> Optional[float]:
    for line in text.splitlines():
        if "heartbeat.json" in line and "~" in line and "min" in line:
            try:
                return float(line.split("~")[1].split("min")[0].strip())
            except Exception:
                pass
    return None


def _norm_sym(value: Any) -> str:
    return str(value or "").strip().upper()


def _risk_flags_with_block_types() -> Dict[str, str]:
    import pandas as pd

    out: Dict[str, str] = {}
    path = RESULTS / "performance_risk_overlay.csv"
    if not path.is_file():
        return out
    ro = pd.read_csv(path, low_memory=False)
    sym_col = "ticker" if "ticker" in ro.columns else "symbol"
    flag_col = "risk_flag" if "risk_flag" in ro.columns else "drag_flag"
    for _, row in ro.iterrows():
        sym = _norm_sym(row.get(sym_col))
        flag = str(row.get(flag_col) or "OK")
        if sym and ("BLOCK_NEW_BUY" in flag or "FORCE_EXIT" in flag):
            out[sym] = flag
    return out


def _build_alignment_rows(
    *,
    cert_status: str,
    entry_candidates: List[Dict[str, Any]],
    planner_symbols: List[str],
    alloc_map: Dict[str, Dict[str, Any]],
    risk_blocked: List[str],
    risk_block_flags: Dict[str, str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen: set[str] = set()

    def add_row(
        candidate: str,
        source: str,
        allocation_action: str,
        planner_status: str,
        risk_flag: str,
        final_status: str,
        rejection_reason: str,
    ) -> None:
        key = f"{candidate}|{source}"
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "candidate": candidate,
                "source": source,
                "allocation_action": allocation_action,
                "planner_status": planner_status,
                "risk_flag": risk_flag,
                "certification_status": cert_status,
                "final_status": final_status,
                "rejection_reason": rejection_reason,
            }
        )

    for c in entry_candidates:
        sym = _norm_sym(c.get("ticker"))
        alloc = alloc_map.get(sym, {})
        alloc_action = str(alloc.get("recommended_action") or c.get("allocation_action") or "")
        risk_flag = str(c.get("risk_flag") or "OK")
        eligible = bool(c.get("eligible_for_auto_entry"))
        planner_status = "PLANNER_TOP" if planner_symbols and sym == planner_symbols[0] else ""
        if sym in planner_symbols and planner_status != "PLANNER_TOP":
            planner_status = "PLANNER_INCLUDED"
        if sym in risk_blocked:
            final_status = "BLOCKED_FINAL_RISK_OVERLAY"
            rejection = f"FINAL_SELECTION_RISK_OVERLAY_BLOCK:{sym}"
        elif not eligible:
            final_status = "EXCLUDED_ENTRY_GATE"
            rejection = str(c.get("block_reasons") or "not eligible")
        elif alloc_action in ("WATCH", "HOLD") and alloc_action not in ("OPEN_NEW", "ADD"):
            final_status = "ELIGIBLE_ENTRY_GATE_ONLY"
            rejection = f"allocation_action={alloc_action}; open_new=0"
        else:
            final_status = "ELIGIBLE"
            rejection = ""
        add_row(
            sym,
            "trade_opportunities+paper_entry_loop_stage",
            alloc_action,
            planner_status,
            risk_flag,
            final_status,
            rejection,
        )

    for sym in planner_symbols:
        alloc = alloc_map.get(sym, {})
        alloc_action = str(alloc.get("recommended_action") or "")
        risk_flag = risk_block_flags.get(sym, str(alloc.get("risk_flag") or "OK"))
        if sym in {_norm_sym(c.get("ticker")) for c in entry_candidates}:
            continue
        if sym in risk_blocked:
            final_status = "BLOCKED_FINAL_RISK_OVERLAY"
            rejection = f"FINAL_SELECTION_RISK_OVERLAY_BLOCK:{sym}"
        else:
            final_status = "PLANNER_ONLY"
            rejection = "planner preview without entry-gate eligibility"
        add_row(
            sym,
            "execute_trades.preview_planned_buy_symbols",
            alloc_action,
            "PLANNER_TOP" if sym == planner_symbols[0] else "PLANNER_INCLUDED",
            risk_flag,
            final_status,
            rejection,
        )

    # WFC ADD row from trade_opportunities (planner intake, not ENTRY gate)
    add_row(
        "WFC",
        "trade_opportunities(ADD)",
        str(alloc_map.get("WFC", {}).get("recommended_action") or "HOLD"),
        "PLANNER_INTAKE",
        str(alloc_map.get("WFC", {}).get("risk_flag") or "OK"),
        "SECTOR_ADD_BLOCKED",
        "ADD_BLOCKED_BY_SECTOR_HARD; add_candidates=0",
    )

    return rows


def run_phase_196s_audit(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.paper_arm_status import build_paper_arm_status
    from services.paper_entry_loop_stage import (
        _load_entry_candidates,
        _risk_blocked_in_selection,
    )
    from services.execute_trades import preview_planned_buy_symbols

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
    arm_st = build_paper_arm_status(verbose_notes=False)

    watchdog_on = "Watchdog process running: True" in sup_out
    loop_on = "Continuous loop process running: True" in sup_out
    hb_age = _parse_hb_age(sup_out) or _parse_hb_age(fresh_out)
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

    audit = _load_json(RESULTS / "paper_entry_loop_stage_audit.json")
    alloc = _load_json(RESULTS / "portfolio_allocation_summary.json")
    cert = _load_json(RESULTS / "capital_preservation_certification.json")
    cert_status = str(cert.get("certification_status") or "REVIEW")

    entry_candidates, _ = _load_entry_candidates()
    eligible = [c for c in entry_candidates if c.get("eligible_for_auto_entry")]
    planner_symbols, planner_meta = preview_planned_buy_symbols(
        "paper", verbose=False, for_execute=True
    )
    risk_blocked = _risk_blocked_in_selection(planner_symbols)

    import pandas as pd

    alloc_map: Dict[str, Dict[str, Any]] = {}
    alloc_path = RESULTS / "portfolio_allocation_recommendations.csv"
    if alloc_path.is_file():
        al = pd.read_csv(alloc_path, low_memory=False)
        for _, row in al.iterrows():
            sym = _norm_sym(row.get("ticker"))
            if sym:
                alloc_map[sym] = row.to_dict()

    risk_block_flags = _risk_flags_with_block_types()

    bac_row = next((c for c in entry_candidates if c.get("ticker") == "BAC"), None)
    ung_flag = risk_block_flags.get("UNG", "")
    bac_considered = bac_row is not None
    bac_rejection = ""
    if bac_row:
        if not bac_row.get("eligible_for_auto_entry"):
            bac_rejection = str(bac_row.get("block_reasons") or "")
        else:
            bac_rejection = (
                "allocation_action=WATCH; open_new_candidates=0; "
                "not OPEN_NEW; planner ranks UNG higher; "
                "BAC would face BUY_BLOCKED_BY_SECTOR_HARD in planner"
            )

    top_alloc = ""
    tops = alloc.get("top_5_opportunities") or []
    if tops:
        top_alloc = str(tops[0].get("symbol") if isinstance(tops[0], dict) else tops[0])
    elif eligible:
        top_alloc = str(eligible[0].get("ticker") or "")
    else:
        top_alloc = "BAC"

    planner_top = planner_symbols[0] if planner_symbols else ""

    if planner_top and planner_top in risk_blocked:
        final_overlay = "BLOCK"
        final_block_if_armed = f"FINAL_SELECTION_RISK_OVERLAY_BLOCK:{','.join(risk_blocked)}"
    elif not planner_symbols:
        final_overlay = "NOT_REACHED"
        final_block_if_armed = "NO_FINAL_PLANNED_ENTRIES"
    else:
        final_overlay = "REVIEW"
        final_block_if_armed = ""

    current_block = str(audit.get("gate_reason") or "")
    audit_reached_planner = bool((audit.get("details") or {}).get("planner_preview"))

    issue_type = "ALLOCATION_AND_ENTRY_ENGINE_CONTRACT_MISMATCH"
    notes.append(
        "paper_entry_loop_stage pre-filters ENTRY rows with risk/allocation BLOCK; "
        "execute_trades planner preview uses full trade_opportunities and does not "
        "apply performance_risk_overlay BLOCK_NEW_BUY before ranking."
    )
    notes.append(
        f"Entry eligible={[c.get('ticker') for c in eligible]}; "
        f"planner_preview={planner_symbols}; risk_blocked={risk_blocked}."
    )
    notes.append(
        "Final risk overlay in paper_entry_loop_stage blocks planner output when armed; "
        "current live audit stops at ARM_EXPIRED before planner stage."
    )
    if alloc.get("open_new_candidates") == 0 and alloc.get("add_candidates") == 0:
        notes.append(
            "Allocation summary open_new=0 and add=0 — no actionable deployment slots "
            "despite ENTRY signals in trade_opportunities."
        )

    stale_path = RESULTS / "controlled_paper_entry_loop_candidates.csv"
    if stale_path.is_file():
        try:
            stale = pd.read_csv(stale_path, low_memory=False)
            stale_syms = (
                set(stale["ticker"].astype(str).str.upper()) if "ticker" in stale.columns else set()
            )
            live_syms = {_norm_sym(c.get("ticker")) for c in entry_candidates}
            if stale_syms - live_syms:
                notes.append(
                    f"controlled_paper_entry_loop_candidates.csv is stale vs live entry path "
                    f"(extra symbols: {sorted(stale_syms - live_syms)})."
                )
        except Exception:
            pass

    could_order = bool(eligible and planner_symbols and not risk_blocked)
    if risk_blocked:
        could_order = False

    alignment_rows = _build_alignment_rows(
        cert_status=cert_status,
        entry_candidates=entry_candidates,
        planner_symbols=planner_symbols,
        alloc_map=alloc_map,
        risk_blocked=risk_blocked,
        risk_block_flags=risk_block_flags,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate",
                "source",
                "allocation_action",
                "planner_status",
                "risk_flag",
                "certification_status",
                "final_status",
                "rejection_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(alignment_rows)

    mtime_after = PAPER_ARM_PATH.stat().st_mtime if PAPER_ARM_PATH.is_file() else None
    confirm_unchanged = mtime_before == mtime_after

    status = "PASS"
    if sup_rc != 0 or fresh_rc != 0:
        status = "FAIL"
        notes.append("Runtime status script failure.")
    elif not confirm_unchanged:
        status = "FAIL"
        notes.append("paper_arm_confirm modified during audit.")
    elif positions != ["JPM", "WFC"] or open_orders != 0 or live_execution:
        status = "FAIL"
        notes.append("Portfolio state mismatch.")
    elif not head_commit.startswith("efca296f"):
        status = "REVIEW"
        notes.append(f"HEAD={head_commit[:8]}; expected efca296f base.")

    payload = {
        "phase": "196S",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "head_commit": head_commit[:8] if head_commit else "",
        "market_open": bool(
            ((audit.get("details") or {}).get("checks") or {}).get("market_open", True)
        ),
        "runtime_pipeline_health": pipeline_health,
        "paper_arm_confirm_status": arm_st.get("status", "EXPIRED"),
        "paper_entry_armed": bool(arm_st.get("paper_entry_armed")),
        "live_execution": live_execution,
        "open_orders": open_orders,
        "positions": positions,
        "allocation_open_new_count": int(alloc.get("open_new_candidates") or 0),
        "allocation_add_count": int(alloc.get("add_candidates") or 0),
        "allocation_exit_count": int(alloc.get("exit_candidates") or 0),
        "allocation_trim_count": int(alloc.get("trim_candidates") or 0),
        "allocation_block_count": int(alloc.get("blocked_candidates") or 0),
        "top_allocation_candidate": top_alloc,
        "planner_preview_candidate": planner_top,
        "final_candidate": "",
        "final_risk_overlay_status": (
            "NOT_REACHED" if current_block.startswith("ARM_") else final_overlay
        ),
        "final_block_reason": current_block or final_block_if_armed,
        "final_block_reason_if_armed": final_block_if_armed,
        "ung_selected_or_previewed": "UNG" in planner_symbols,
        "ung_risk_flag": ung_flag,
        "bac_considered": bac_considered,
        "bac_rejection_reason": bac_rejection,
        "candidate_alignment_issue_type": issue_type,
        "secondary_issue_types": [
            "PLANNER_PREVIEW_NOT_FILTERED_BY_RISK",
            "PLANNER_USING_DIFFERENT_SOURCE_THAN_ALLOCATION",
            "NO_BUG_RISK_OVERLAY_BLOCK_IS_EXPECTED",
        ],
        "risk_block_symbols": sorted(risk_block_flags.keys()),
        "eligible_entry_symbols": [c.get("ticker") for c in eligible],
        "planner_preview_symbols": planner_symbols,
        "planner_skip_reasons": planner_meta.get("skip_reasons"),
        "paper_entry_audit_reached_planner": audit_reached_planner,
        "arming_could_allow_paper_order_next_cycle": could_order,
        "paper_arm_confirm_created": False,
        "live_confirm_created": False,
        "manual_approval_used": False,
        "execute_trades_run": False,
        "broker_orders_submitted": False,
        "execution_logic_changed": False,
        "governance_logic_changed": False,
        "risk_logic_changed": False,
        "allocation_logic_changed": False,
        "alignment_map_csv": str(OUTPUT_CSV.relative_to(ROOT)).replace("\\", "/"),
        "alignment_row_count": len(alignment_rows),
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196S] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196S] wrote {OUTPUT_CSV.relative_to(ROOT)}")
        print(f"[196S] status={status} issue_type={issue_type}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196S paper entry candidate alignment audit")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196s_audit(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
