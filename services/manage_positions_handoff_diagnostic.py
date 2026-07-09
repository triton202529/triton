"""
manage_positions_handoff_diagnostic.py
Phase 197G — Compare planned defensive exits vs executable handoff rows.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
PLAN_CSV = RESULTS / "manage_positions_plan.csv"
HANDOFF_CSV = LIVE / "manage_orders.csv"
REPRICE_CSV = RESULTS / "manage_orders.csv"
READINESS_JSON = RESULTS / "execution_readiness.json"
HANDOFF_AUDIT_JSON = RESULTS / "manage_positions_handoff_audit.json"
PREVIEW_CSV = RESULTS / "defensive_exit_preview_handoff.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        import pandas as pd

        return pd.read_csv(path, low_memory=False).to_dict(orient="records")
    except Exception:
        return []


def _planned_defensive_exits(plan_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in plan_rows:
        status = str(row.get("status") or "").strip().lower()
        action = str(row.get("final_action") or row.get("management_action") or "").strip().upper()
        if status == "planned" and action in ("EXIT", "TRIM", "ROTATE_EXIT"):
            out.append(row)
    return out


def _handoff_rows(handoff_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in handoff_rows:
        sym = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        opp = str(row.get("opportunity_type") or row.get("effective_stance") or "").strip().upper()
        if opp in ("EXIT", "TRIM", "ROTATE_EXIT", "SELL"):
            out.append(row)
    return out


def _readiness_status() -> Dict[str, Any]:
    if not READINESS_JSON.is_file():
        return {"readiness_status": "UNKNOWN", "live_execution_permitted": True}
    try:
        doc = json.loads(READINESS_JSON.read_text(encoding="utf-8"))
        return {
            "readiness_status": str(doc.get("readiness_status") or "UNKNOWN"),
            "live_execution_permitted": bool(doc.get("live_execution_permitted", True)),
            "paper_execution_permitted": bool(doc.get("paper_execution_permitted", False)),
            "governance_authorized": doc.get("governance_authorized"),
            "governance_mode": doc.get("governance_mode"),
            "risk_reducing_paper_actions_allowed": doc.get("risk_reducing_paper_actions_allowed"),
            "failed_checks": list(doc.get("failed_checks") or []),
        }
    except Exception:
        return {"readiness_status": "UNKNOWN", "live_execution_permitted": True}


def _parse_placement_output(output: str) -> Dict[str, Any]:
    text = output or ""
    parsed: Dict[str, Any] = {
        "lifecycle_rows": None,
        "planned_rows": None,
        "in_flight_sat": None,
        "illegal_sells": None,
        "handoff_contract_mismatch": "HANDOFF_CONTRACT_MISMATCH" in text,
        "batch_empty": "BATCH_EMPTY" in text,
        "batch_notional_block": "BATCH_NOTIONAL_BLOCK" in text,
        "handoff_notional_blocked": "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED" in text,
        "master_gate_block": "MASTER_GATE" in text or "MASTER_GATE_BLOCK" in text,
        "exec_gate_block": "EXEC_GATE" in text or "EXEC_GATE_BLOCK" in text,
        "manage_block": "[MANAGE_BLOCK]" in text,
    }
    for key, pattern in (
        ("lifecycle_rows", r"lifecycle_rows=(\d+)"),
        ("planned_rows", r"planned=(\d+)"),
        ("in_flight_sat", r"in_flight_sat=(\d+)"),
        ("illegal_sells", r"illegal_sells=(\d+)"),
        ("batch_notional_estimate", r"est_batch_notional=([\d.]+)"),
        ("configured_batch_cap", r"exceeds max ([\d.]+)"),
    ):
        m = re.search(pattern, text)
        if m:
            val = m.group(1).rstrip(".")
            parsed[key] = float(val) if "." in str(val) else int(val)
    return parsed


def build_handoff_diagnostic(
    *,
    placement_output: str = "",
    exec_meta: Optional[Dict[str, Any]] = None,
    mode: str = "paper",
) -> Dict[str, Any]:
    plan_rows = _load_csv(PLAN_CSV)
    planned_exits = _planned_defensive_exits(plan_rows)
    planned_symbols = sorted(
        {str(r.get("symbol") or "").strip().upper() for r in planned_exits if r.get("symbol")}
    )

    handoff_all = _load_csv(HANDOFF_CSV)
    handoff_exit_rows = _handoff_rows(handoff_all)
    handoff_source = "exec_handoff"
    preview_meta: Dict[str, Any] = {}

    if not handoff_exit_rows:
        try:
            from services.defensive_exit_preview_handoff import (
                ensure_defensive_exit_preview_handoff,
                load_defensive_handoff_orders_for_analysis,
            )

            preview_meta = ensure_defensive_exit_preview_handoff(mode=mode)
            preview_orders, preview_source = load_defensive_handoff_orders_for_analysis()
            if preview_orders and preview_source == "preview_handoff":
                handoff_exit_rows = [
                    {
                        "ticker": o["symbol"],
                        "symbol": o["symbol"],
                        "opportunity_type": o.get("opportunity_type"),
                        "effective_stance": o.get("effective_stance"),
                        "qty": o.get("qty"),
                        "limit_price": o.get("limit_price"),
                        "preview_only": True,
                    }
                    for o in preview_orders
                ]
                handoff_source = "preview_handoff"
        except Exception:
            pass

    handoff_symbols = sorted(
        {
            str(r.get("ticker") or r.get("symbol") or "").strip().upper()
            for r in handoff_exit_rows
            if (r.get("ticker") or r.get("symbol"))
        }
    )

    readiness = _readiness_status()
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    placement = _parse_placement_output(placement_output)
    meta = dict(exec_meta or {})

    preview_generation_allowed = False
    execution_allowed = False
    try:
        from services.defensive_exit_preview_handoff import evaluate_defensive_preview_eligibility

        preview_elig = evaluate_defensive_preview_eligibility(readiness)
        preview_generation_allowed = bool(preview_elig.get("preview_generation_allowed"))
        execution_allowed = bool(preview_elig.get("execution_allowed"))
    except Exception:
        preview_elig = {}

    blocked_by_readiness = readiness_status in (
        "PARTIALLY_READY",
        "NOT_READY",
        "BLOCKED",
    )
    if str(mode).lower() == "paper" and blocked_by_readiness:
        try:
            from services.paper_loop_mediated_execution_readiness_bridge import (
                evaluate_paper_loop_mediated_execution_readiness_bridge,
            )

            bridge = evaluate_paper_loop_mediated_execution_readiness_bridge(mode="paper")
            if bridge.get("bridge_active"):
                blocked_by_readiness = False
                readiness_status = str(bridge.get("effective_readiness_status") or readiness_status)
                execution_allowed = True
        except Exception:
            pass
    if preview_generation_allowed and handoff_source == "preview_handoff" and handoff_exit_rows:
        blocked_by_readiness = False
        execution_allowed = False
    blocked_by_governance = readiness.get("governance_authorized") is False

    handoff_status = "OK"
    handoff_block_reason = ""
    batch_empty_explained = False

    if not planned_exits:
        handoff_status = "NO_PLANNED_EXITS"
    elif (
        preview_generation_allowed
        and handoff_exit_rows
        and handoff_source == "preview_handoff"
        and not execution_allowed
    ):
        handoff_status = "HANDOFF_PREVIEW_ROWS_PRESENT"
        handoff_block_reason = "preview_only_orange_degraded_governance; execution_blocked"
        batch_empty_explained = True
    elif blocked_by_readiness and not handoff_exit_rows:
        handoff_status = "HANDOFF_BLOCKED_BY_READINESS"
        handoff_block_reason = f"execution_readiness={readiness_status}"
        batch_empty_explained = True
    elif blocked_by_governance and not handoff_exit_rows:
        handoff_status = "HANDOFF_BLOCKED_BY_GOVERNANCE"
        handoff_block_reason = "governance_not_authorized"
        batch_empty_explained = True
    elif meta.get("global_abort") or meta.get("manage_block"):
        handoff_status = "HANDOFF_BLOCKED_BY_GOVERNANCE"
        handoff_block_reason = str(meta.get("block_reason") or "manage_block")
        batch_empty_explained = True
    elif planned_exits and not handoff_exit_rows and not meta.get("subprocess_ran"):
        if blocked_by_readiness:
            handoff_status = "HANDOFF_BLOCKED_BY_READINESS"
            handoff_block_reason = f"execution_readiness={readiness_status}"
        else:
            handoff_status = "HANDOFF_NOT_WRITTEN"
            handoff_block_reason = "handoff_csv_empty_before_placement"
        batch_empty_explained = True
    elif planned_exits and handoff_exit_rows and placement.get("handoff_contract_mismatch"):
        handoff_status = "HANDOFF_CONTRACT_MISMATCH"
        handoff_block_reason = "effective_stance_opportunity_type_not_mapped_to_placement_rows"
        batch_empty_explained = True
    elif planned_exits and handoff_exit_rows and placement.get("batch_empty"):
        if placement.get("lifecycle_rows") == 0:
            handoff_status = "HANDOFF_CONTRACT_MISMATCH"
            handoff_block_reason = "handoff_rows_present_but_zero_lifecycle_rows"
            batch_empty_explained = True
        elif placement.get("in_flight_sat"):
            handoff_status = "HANDOFF_SATISFIED_IN_FLIGHT"
            handoff_block_reason = f"in_flight_sat={placement.get('in_flight_sat')}"
            batch_empty_explained = True
        elif placement.get("illegal_sells"):
            handoff_status = "HANDOFF_BLOCKED_INVALID_QTY"
            handoff_block_reason = f"illegal_sells={placement.get('illegal_sells')}"
            batch_empty_explained = True
        elif meta.get("batch_empty_noop"):
            handoff_status = "HANDOFF_FILTERED_TO_EMPTY"
            handoff_block_reason = "placement_filters_removed_all_rows"
            batch_empty_explained = True
        else:
            handoff_status = "HANDOFF_FILTERED_TO_EMPTY"
            handoff_block_reason = "batch_empty_after_filters"
            batch_empty_explained = True
    elif planned_exits and handoff_exit_rows:
        from services.defensive_exit_sizing_cap import analyze_defensive_exit_sizing

        sizing = analyze_defensive_exit_sizing(placement_output=placement_output)
        if placement.get("batch_notional_block") or sizing.get("cap_excess_amount"):
            handoff_status = str(
                sizing.get("handoff_status") or "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED"
            )
            handoff_block_reason = (
                f"batch_notional={sizing.get('batch_notional_estimate')} "
                f"cap={sizing.get('configured_batch_cap_effective')} "
                f"excess={sizing.get('cap_excess_amount')}"
            )
            batch_empty_explained = True
        else:
            handoff_status = "HANDOFF_ROWS_PRESENT"
            if placement.get("planned_rows", 0) and placement.get("planned_rows", 0) > 0:
                handoff_status = "HANDOFF_EXECUTABLE"

    missing_from_handoff = sorted(set(planned_symbols) - set(handoff_symbols))
    extra_in_handoff = sorted(set(handoff_symbols) - set(planned_symbols))

    audit: Dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "phase": "197G",
        "mode": mode,
        "planned_exit_count": len(planned_exits),
        "planned_exit_symbols": planned_symbols,
        "handoff_path": str(HANDOFF_CSV.relative_to(ROOT)).replace("\\", "/"),
        "repricing_orders_path": str(REPRICE_CSV.relative_to(ROOT)).replace("\\", "/"),
        "repricing_orders_note": (
            "data/results/manage_orders.csv is open-order repricing (manage_open_orders), "
            "not defensive-exit handoff; use data/live/manage_orders.csv"
        ),
        "manage_orders_handoff_rows": len(handoff_exit_rows),
        "manage_orders_handoff_symbols": handoff_symbols,
        "missing_from_handoff_symbols": missing_from_handoff,
        "extra_in_handoff_symbols": extra_in_handoff,
        "handoff_status": handoff_status,
        "handoff_block_reason": handoff_block_reason,
        "batch_empty_explained": batch_empty_explained,
        "readiness_status": readiness_status,
        "governance_authorized": readiness.get("governance_authorized"),
        "live_execution_permitted": readiness.get("live_execution_permitted"),
        "preview_generation_allowed": preview_generation_allowed,
        "execution_allowed": execution_allowed,
        "handoff_source": handoff_source,
        "preview_handoff_path": (
            str(PREVIEW_CSV.relative_to(ROOT)).replace("\\", "/")
            if handoff_source == "preview_handoff"
            else None
        ),
        "preview_meta": preview_meta or None,
        "placement_parse": placement,
        "exec_meta": {
            k: meta.get(k)
            for k in (
                "subprocess_ran",
                "batch_empty_noop",
                "guard_skipped",
                "blocked_rows",
                "executed",
            )
        },
        "rows_blocked_by_readiness": len(planned_exits) if blocked_by_readiness else 0,
        "rows_blocked_by_governance": len(planned_exits) if blocked_by_governance else 0,
        "rows_blocked_by_invalid_qty": int(placement.get("illegal_sells") or 0),
        "rows_blocked_by_in_flight": int(placement.get("in_flight_sat") or 0),
    }
    if planned_exits and handoff_exit_rows:
        try:
            from services.defensive_exit_sizing_cap import analyze_defensive_exit_sizing

            audit["sizing_cap_analysis"] = analyze_defensive_exit_sizing(
                placement_output=placement_output
            )
        except Exception:
            pass
    return audit


def write_handoff_audit(audit: Dict[str, Any]) -> Path:
    HANDOFF_AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = HANDOFF_AUDIT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, HANDOFF_AUDIT_JSON)
    return HANDOFF_AUDIT_JSON
