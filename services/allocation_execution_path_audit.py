"""
TRITON Phase 179 — Allocation-to-Execution Path Audit (diagnostic only).

Traces approved allocation candidates through governance, authorization, lifecycle,
position management, capital allocation, and order placement to explain why
planned=0 / executed=0 despite available portfolio slots.

SAFETY: analysis only — does not modify governance, lifecycle, execution, broker,
allocation scoring, risk overlays, or place orders.

Outputs (data/results/):
    allocation_execution_path_audit.json
    allocation_execution_decision_trace.csv
    allocation_execution_blockers.csv

Run:
    python -m services.allocation_execution_path_audit
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.concentration_governance_analysis import _load_positions, _read_csv, _read_json
from services.diversification_feasibility_analysis import _classify_feasibility, _lifecycle_row
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
OUT_JSON = RESULTS / "allocation_execution_path_audit.json"
OUT_TRACE = RESULTS / "allocation_execution_decision_trace.csv"
OUT_BLOCKERS = RESULTS / "allocation_execution_blockers.csv"

FOCUS_SYMBOLS = ["XLK", "HD", "QQQ"]
ACTIONABLE_ACTIONS = {"OPEN_NEW", "ADD", "TRIM", "EXIT"}
BUY_ACTIONS = {"OPEN_NEW", "ADD"}

FINAL_STATUSES = {
    "EXECUTABLE_NOW",
    "BLOCKED_BY_GOVERNANCE",
    "BLOCKED_BY_LIFECYCLE",
    "BLOCKED_BY_POSITION_MANAGER",
    "BLOCKED_BY_CAPITAL_ALLOCATION",
    "BLOCKED_BY_EXECUTION_LAYER",
    "BLOCKED_BY_BROKER_RULES",
    "BLOCKED_BY_RISK_OVERLAY",
    "NO_ACTION_REQUIRED",
}

LAYER_ORDER = [
    ("allocation", "NO_ACTION_REQUIRED"),
    ("governance", "BLOCKED_BY_GOVERNANCE"),
    ("execution", "BLOCKED_BY_EXECUTION_LAYER"),
    ("lifecycle", "BLOCKED_BY_LIFECYCLE"),
    ("position_manager", "BLOCKED_BY_POSITION_MANAGER"),
    ("capital", "BLOCKED_BY_CAPITAL_ALLOCATION"),
    ("risk", "BLOCKED_BY_RISK_OVERLAY"),
    ("placement", "BLOCKED_BY_EXECUTION_LAYER"),
]


def _log(msg: str) -> None:
    print(f"[allocation_execution_path_audit] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _first_json(*paths: Path) -> Tuple[Path, Dict[str, Any]]:
    for p in paths:
        if p.is_file():
            return p, _read_json(p)
    return paths[0], {}


def _drop_index(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df.empty:
        return {}
    sym_col = "symbol" if "symbol" in df.columns else "ticker" if "ticker" in df.columns else None
    if not sym_col:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        sym = _norm_sym(row.get(sym_col))
        if not sym:
            continue
        st = str(row.get("status") or "").lower()
        if st in ("dropped", "blocked", "skipped"):
            out[sym] = row.to_dict()
        elif sym not in out and st == "planned":
            out[sym] = row.to_dict()
    return out


def _load_inputs() -> Dict[str, Any]:
    auth_path, auth = _first_json(
        RESULTS / "governance_authorization.json",
        RESULTS / "execution_authorization.json",
    )
    ready_path, readiness = _first_json(
        RESULTS / "execution_readiness.json",
        RESULTS / "readiness_assessment.json",
    )
    gate_path, gate = _first_json(RESULTS / "paper_submission_gate_audit.json")
    exec_plan_path, exec_plan = _first_json(RESULTS / "execution_plan.json")
    manage_plan_path, manage_plan = _first_json(RESULTS / "manage_positions_plan.json")
    capital_path, capital = _first_json(RESULTS / "capital_reallocation.json")
    alloc_path, alloc_summary = _first_json(RESULTS / "portfolio_allocation_summary.json")
    cpe_path, cpe = _first_json(RESULTS / "capital_preservation_escalation.json")
    guard_path, guard = _first_json(RESULTS / "guard_snapshot.json")

    drop_df = _read_csv(RESULTS / "execution_drop_diagnostics.csv")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    opportunities = _read_csv(RESULTS / "trade_opportunities.csv")
    positions_list = _load_positions()
    positions_map = {p["symbol"]: p for p in positions_list if p.get("symbol")}
    held = set(positions_map.keys())

    return {
        "paths": {
            "governance_authorization": str(auth_path),
            "execution_readiness": str(ready_path),
            "paper_submission_gate_audit": str(gate_path),
            "execution_plan": str(exec_plan_path),
            "manage_positions_plan": str(manage_plan_path),
            "capital_reallocation": str(capital_path),
            "portfolio_allocation_summary": str(alloc_path),
            "capital_preservation_escalation": str(cpe_path),
            "guard_snapshot": str(guard_path),
        },
        "auth": auth,
        "readiness": readiness,
        "gate": gate,
        "exec_plan": exec_plan,
        "manage_plan": manage_plan,
        "capital": capital,
        "alloc_summary": alloc_summary,
        "cpe": cpe,
        "guard": guard,
        "drop_index": _drop_index(drop_df),
        "recommendations": recommendations,
        "lifecycle": lifecycle,
        "opportunities": opportunities,
        "held": held,
        "positions": positions_map,
    }


def _candidate_symbols(ctx: Dict[str, Any]) -> List[str]:
    recs = ctx["recommendations"]
    alloc = ctx["alloc_summary"]
    symbols: List[str] = []
    seen: Set[str] = set()

    def _add(sym: str) -> None:
        s = _norm_sym(sym)
        if s and s not in seen:
            seen.add(s)
            symbols.append(s)

    for sym in FOCUS_SYMBOLS:
        _add(sym)
    if not recs.empty and "recommended_action" in recs.columns:
        open_new = recs[recs["recommended_action"].astype(str).str.upper() == "OPEN_NEW"]
        for t in open_new.get("ticker", pd.Series(dtype=str)):
            _add(t)
    for item in alloc.get("top_5_opportunities") or []:
        if isinstance(item, dict):
            _add(item.get("ticker") or item.get("symbol"))
    return symbols


def _allocation_layer(
    sym: str,
    rec: Dict[str, Any],
    alloc_summary: Dict[str, Any],
    blocked_syms: Set[str],
) -> Tuple[bool, str, Dict[str, Any]]:
    action = str(rec.get("recommended_action") or "").upper()
    if action in ("WATCH", "HOLD", "BLOCK", ""):
        return False, f"recommended_action={action or 'NONE'}", {"recommended_action": action}
    if action == "OPEN_NEW" and sym in blocked_syms:
        return False, "symbol in blocked_new_buy_symbols", {"recommended_action": action}
    if action == "OPEN_NEW":
        slots = _to_float(alloc_summary.get("available_slots"), 0)
        if slots <= 0:
            return (
                False,
                "available_slots=0",
                {"recommended_action": action, "available_slots": slots},
            )
    if action not in ACTIONABLE_ACTIONS:
        return False, f"non_actionable action={action}", {"recommended_action": action}
    return True, "allocation gate passed", {"recommended_action": action}


def _governance_layer(auth: Dict[str, Any], cpe: Dict[str, Any]) -> Tuple[bool, str]:
    gov = bool(auth.get("governance_authorized"))
    if gov:
        return True, "governance_authorized=true"
    esc = str(cpe.get("escalation_state") or auth.get("escalation_state") or "UNKNOWN")
    reason = (auth.get("gate_reasons") or {}).get(
        "governance"
    ) or f"Escalation {esc} does not meet green-governance requirement"
    return False, reason


def _execution_auth_layer(auth: Dict[str, Any]) -> Tuple[bool, str]:
    if bool(auth.get("execution_authorized")):
        return True, "execution_authorized=true"
    return False, "execution_authorized=false"


def _lifecycle_layer(
    sym: str,
    rec: Dict[str, Any],
    lc: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    action = str(rec.get("recommended_action") or "").upper()
    alloc_stance = str(rec.get("effective_stance") or rec.get("lifecycle_action") or "").upper()
    eff_stance = str(lc.get("effective_stance") or alloc_stance or "").upper()
    lc_action = str(lc.get("lifecycle_action") or rec.get("lifecycle_action") or "").upper()
    exec_blocked = lc.get("execution_blocked") is True
    block_reason = str(lc.get("execution_block_reason") or "")

    details = {
        "lifecycle_action": lc_action or alloc_stance,
        "effective_stance": eff_stance,
        "allocation_effective_stance": alloc_stance,
        "lifecycle_mismatch": alloc_stance != eff_stance
        and bool(alloc_stance)
        and bool(eff_stance),
    }

    if exec_blocked:
        return False, block_reason or "execution_blocked=true", details

    if action in BUY_ACTIONS:
        if eff_stance not in ("BUY", "ADD"):
            mismatch = ""
            if details["lifecycle_mismatch"]:
                mismatch = f" (allocation={alloc_stance} vs lifecycle={eff_stance})"
            return False, f"effective_stance={eff_stance} not BUY/ADD{mismatch}", details
        if action == "ADD" and eff_stance not in ("ADD", "BUY"):
            return False, f"ADD requires ADD/BUY stance, got {eff_stance}", details
    elif action == "TRIM" and eff_stance not in ("TRIM", "EXIT"):
        return False, f"TRIM requires TRIM/EXIT stance, got {eff_stance}", details
    elif action == "EXIT" and eff_stance != "EXIT":
        return False, f"EXIT requires EXIT stance, got {eff_stance}", details

    return True, "lifecycle actionable", details


def _position_manager_layer(
    sym: str,
    rec: Dict[str, Any],
    manage_plan: Dict[str, Any],
    capital: Dict[str, Any],
    alloc_summary: Dict[str, Any],
    held: Set[str],
    positions: Dict[str, Any],
) -> Tuple[bool, str, str]:
    action = str(rec.get("recommended_action") or "").upper()
    skip = manage_plan.get("skip_reasons") or {}
    if sym in held and action == "ADD":
        max_w = _to_float(
            (capital.get("regime_portfolio") or {}).get("max_position_weight_pct_effective"),
            _to_float(alloc_summary.get("max_position_weight_pct"), 0.35),
        )
        pos = positions.get(sym) or {}
        mv = _to_float(pos.get("market_value") or pos.get("value"), 0)
        total = _to_float(alloc_summary.get("total_position_value"), 0)
        weight = (mv / total) if total > 0 else 1.0
        if weight >= max_w * 0.99:
            return False, "ADD_EXCEEDS_MAX_POSITION_WEIGHT", f"weight={weight:.1%} cap={max_w:.0%}"
        if "ADD_EXCEEDS_MAX_POSITION_WEIGHT" in skip:
            return False, "ADD_EXCEEDS_MAX_POSITION_WEIGHT", "manage_positions skip catalog"
    if action == "OPEN_NEW" and sym in held:
        return False, "ALREADY_HELD", "OPEN_NEW on held symbol"
    return True, "position manager pass", "OK"


def _capital_layer(
    sym: str,
    rec: Dict[str, Any],
    capital: Dict[str, Any],
    guard: Dict[str, Any],
    alloc_summary: Dict[str, Any],
) -> Tuple[bool, str, str]:
    action = str(rec.get("recommended_action") or "").upper()
    selected = {_norm_sym(s) for s in (capital.get("selected_symbols") or [])}
    freed = _to_float(capital.get("freed_capital"), 0)
    buying_power = guard.get("buying_power")
    slots = _to_float(alloc_summary.get("available_slots"), 0)

    if action == "OPEN_NEW":
        if sym in selected:
            return True, "selected in capital_reallocation", f"freed_capital={freed}"
        if slots > 0 and freed <= 0 and not selected:
            return (
                True,
                "slots available; capital reallocator selected none (non-blocking at allocation)",
                f"available_slots={int(slots)} freed_capital={freed}",
            )
    if action == "ADD" and sym not in selected and freed <= 0:
        return (
            True,
            "ADD may use existing exposure budget",
            f"buying_power={buying_power}",
        )
    if buying_power is not None and _to_float(buying_power, 0) <= 0 and action in BUY_ACTIONS:
        return False, "insufficient buying_power", str(buying_power)
    return True, "capital allocation pass", f"freed_capital={freed} buying_power={buying_power}"


def _risk_layer(
    sym: str,
    rec: Dict[str, Any],
    lc: Dict[str, Any],
    drop_row: Optional[Dict[str, Any]],
    blocked_syms: Set[str],
) -> Tuple[bool, str, str]:
    risk = str(rec.get("risk_flag") or "OK").upper()
    flags: List[str] = []
    if "BLOCK_NEW_BUY" in risk or "FORCE_EXIT" in risk:
        flags.append(f"risk_flag={risk}")
    if sym in blocked_syms:
        flags.append("blocked_new_buy_list")
    if drop_row:
        rc = str(drop_row.get("reason_code") or "")
        if rc == "RISK_GATE_BLOCK":
            detail = str(drop_row.get("reason_detail") or drop_row.get("internal") or rc)
            return False, detail or rc, rc
        if rc in ("FILTERED_BY_QUALITY", "FILTERED_BY_EXECUTION_QUALITY", "INVALID_PRICE"):
            return False, str(drop_row.get("reason_detail") or rc), rc

    if str(rec.get("recommended_action") or "").upper() == "OPEN_NEW":
        _feasibility, blockers = _classify_feasibility(rec, lc, blocked_syms, sym in FOCUS_SYMBOLS)
        if blockers:
            flags.extend(blockers)

    if flags:
        return False, "; ".join(flags), risk
    return True, "risk overlay pass", risk or "OK"


def _placement_layer(
    gate: Dict[str, Any],
    exec_plan: Dict[str, Any],
    auth: Dict[str, Any],
    readiness: Dict[str, Any],
    drop_row: Optional[Dict[str, Any]],
) -> Tuple[bool, str, Dict[str, Any]]:
    paper_gate = gate.get("paper_execution_gate") or {}
    dry_run = bool(
        exec_plan.get("dry_run")
        if exec_plan.get("dry_run") is not None
        else (gate.get("dry_run_decision") or {}).get("dry_run", True)
    )
    final_submit = bool(paper_gate.get("final_submit_allowed"))
    block_reasons = list(paper_gate.get("block_reasons") or [])
    details = {
        "dry_run": dry_run,
        "final_submit_allowed": final_submit,
        "placement_eligible": final_submit and not dry_run,
        "block_reasons": block_reasons,
    }
    if drop_row and str(drop_row.get("status") or "").lower() == "planned":
        details["placement_eligible"] = True
        return True, "symbol planned in execution_drop_diagnostics", details
    if dry_run:
        return False, "execute_trades dry_run=true (--execute not passed)", details
    if not final_submit:
        if not bool(auth.get("governance_authorized")):
            return False, "final_submit_allowed=false (governance_authorized=false)", details
        if str(readiness.get("readiness_status") or "").upper() != "READY":
            return False, f"readiness={readiness.get('readiness_status')} blocks submit", details
        return False, "final_submit_allowed=false", details
    return True, "placement eligible", details


def _classify_final_status(
    passes: Dict[str, bool],
    rec_action: str,
) -> Tuple[str, str]:
    if rec_action.upper() not in ACTIONABLE_ACTIONS:
        return "NO_ACTION_REQUIRED", f"recommended_action={rec_action}"

    if all(passes.values()):
        return "EXECUTABLE_NOW", "all pipeline layers pass"

    for layer, status in LAYER_ORDER:
        if not passes.get(layer, True):
            reason_map = {
                "allocation": f"recommended_action={rec_action}",
                "governance": "governance_authorized=false",
                "execution": "execution_authorized=false",
                "lifecycle": "lifecycle not actionable",
                "position_manager": "position cap or manage_positions block",
                "capital": "capital allocation insufficient",
                "risk": "risk overlay block",
                "placement": "order placement not eligible",
            }
            return status, reason_map.get(layer, layer)
    return "BLOCKED_BY_EXECUTION_LAYER", "unknown block"


def _evaluate_candidate(sym: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    recs = ctx["recommendations"]
    lifecycle = ctx["lifecycle"]
    sub = recs[recs["ticker"].map(_norm_sym) == sym] if not recs.empty else pd.DataFrame()
    rec = sub.iloc[0].to_dict() if not sub.empty else {"ticker": sym, "recommended_action": "—"}
    lc = _lifecycle_row(lifecycle, sym)
    blocked_syms = {
        _norm_sym(s) for s in (ctx["alloc_summary"].get("blocked_new_buy_symbols") or [])
    }
    drop_row = ctx["drop_index"].get(sym)

    alloc_ok, alloc_reason, alloc_meta = _allocation_layer(
        sym, rec, ctx["alloc_summary"], blocked_syms
    )
    gov_ok, gov_reason = _governance_layer(ctx["auth"], ctx["cpe"])
    exec_ok, exec_reason = _execution_auth_layer(ctx["auth"])
    lc_ok, lc_reason, lc_meta = _lifecycle_layer(sym, rec, lc)
    pm_ok, pm_reason, pm_detail = _position_manager_layer(
        sym,
        rec,
        ctx["manage_plan"],
        ctx["capital"],
        ctx["alloc_summary"],
        ctx["held"],
        ctx["positions"],
    )
    cap_ok, cap_reason, cap_result = _capital_layer(
        sym, rec, ctx["capital"], ctx["guard"], ctx["alloc_summary"]
    )
    risk_ok, risk_reason, risk_flags = _risk_layer(sym, rec, lc, drop_row, blocked_syms)
    place_ok, place_reason, place_meta = _placement_layer(
        ctx["gate"],
        ctx["exec_plan"],
        ctx["auth"],
        ctx["readiness"],
        drop_row,
    )

    passes = {
        "allocation": alloc_ok,
        "governance": gov_ok,
        "execution": exec_ok,
        "lifecycle": lc_ok,
        "position_manager": pm_ok,
        "capital": cap_ok,
        "risk": risk_ok,
        "placement": place_ok,
    }
    action = str(rec.get("recommended_action") or "—").upper()
    final_status, blocking_reason = _classify_final_status(passes, action)
    if not alloc_ok:
        blocking_reason = alloc_reason
    elif not gov_ok:
        blocking_reason = gov_reason
    elif not exec_ok:
        blocking_reason = exec_reason
    elif not lc_ok:
        blocking_reason = lc_reason
    elif not pm_ok:
        blocking_reason = pm_reason
    elif not cap_ok:
        blocking_reason = cap_reason
    elif not risk_ok:
        blocking_reason = risk_reason
    elif not place_ok:
        blocking_reason = place_reason

    perf_risk = str(rec.get("risk_flag") or lc.get("risk_flag") or "OK")
    position_cap = pm_detail if not pm_ok else "OK"

    return {
        "symbol": sym,
        "allocation_score": _to_float(rec.get("allocation_score"), None),
        "edge_score": _to_float(rec.get("edge_score"), None),
        "confidence": _to_float(rec.get("confidence") or lc.get("confidence"), None),
        "recommended_action": action,
        "lifecycle_action": lc_meta.get("lifecycle_action"),
        "effective_stance": lc_meta.get("effective_stance"),
        "allocation_effective_stance": lc_meta.get("allocation_effective_stance"),
        "lifecycle_mismatch": lc_meta.get("lifecycle_mismatch"),
        "governance_authorized": bool(ctx["auth"].get("governance_authorized")),
        "execution_authorized": bool(ctx["auth"].get("execution_authorized")),
        "performance_risk_flags": perf_risk,
        "position_cap_status": position_cap,
        "cash_available": ctx["guard"].get("buying_power"),
        "available_slots": _to_float(ctx["alloc_summary"].get("available_slots"), None),
        "capital_allocation_result": cap_result,
        "placement_eligible": place_meta.get("placement_eligible"),
        "dry_run": place_meta.get("dry_run"),
        "final_decision": final_status,
        "blocking_reason": blocking_reason,
        "allocation_pass": alloc_ok,
        "governance_pass": gov_ok,
        "execution_pass": exec_ok,
        "lifecycle_pass": lc_ok,
        "position_manager_pass": pm_ok,
        "risk_pass": risk_ok,
        "capital_pass": cap_ok,
        "placement_pass": place_ok,
        "final_status": final_status,
    }


def _aggregate_blockers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actionable = [r for r in rows if r.get("recommended_action") in ACTIONABLE_ACTIONS]
    total = len(actionable) or len(rows)
    counts: Counter[str] = Counter()
    symbols: Dict[str, List[str]] = defaultdict(list)
    for r in actionable:
        st = str(r.get("final_status") or "UNKNOWN")
        if st == "EXECUTABLE_NOW":
            continue
        counts[st] += 1
        symbols[st].append(str(r.get("symbol")))
    blockers: List[Dict[str, Any]] = []
    for status, count in counts.most_common():
        blockers.append(
            {
                "blocker_category": status,
                "count": count,
                "affected_symbols": sorted(set(symbols[status])),
                "percent_of_candidates": round(100.0 * count / total, 1) if total else 0.0,
            }
        )
    return blockers


def _summary(rows: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    actionable = [r for r in rows if r.get("recommended_action") in ACTIONABLE_ACTIONS]
    executable = [r for r in actionable if r.get("final_status") == "EXECUTABLE_NOW"]
    blocked = [
        r
        for r in actionable
        if r.get("final_status") not in ("EXECUTABLE_NOW", "NO_ACTION_REQUIRED")
    ]
    blockers = _aggregate_blockers(rows)
    top_layer = blockers[0]["blocker_category"] if blockers else "NONE"
    top_reason = blocked[0]["blocking_reason"] if blocked else "none"
    first_executable = next(
        (r["symbol"] for r in rows if r.get("final_status") == "EXECUTABLE_NOW"),
        None,
    )
    first_if_unblocked = None
    for r in actionable:
        if r.get("final_status") != "EXECUTABLE_NOW":
            first_if_unblocked = r["symbol"]
            break

    funnel = {
        "allocation": sum(1 for r in actionable if r.get("allocation_pass")),
        "governance": sum(1 for r in actionable if r.get("governance_pass")),
        "execution": sum(1 for r in actionable if r.get("execution_pass")),
        "lifecycle": sum(1 for r in actionable if r.get("lifecycle_pass")),
        "risk": sum(1 for r in actionable if r.get("risk_pass")),
        "capital": sum(1 for r in actionable if r.get("capital_pass")),
        "placement": sum(1 for r in actionable if r.get("placement_pass")),
        "executable": len(executable),
    }

    exec_plan = ctx["exec_plan"]
    manage_plan = ctx["manage_plan"]
    exec_ts = str(exec_plan.get("timestamp") or "")
    manage_ts = str(manage_plan.get("timestamp") or "")
    use_manage = manage_ts >= exec_ts if manage_ts and exec_ts else bool(manage_plan)
    primary_plan = manage_plan if use_manage else exec_plan

    def _plan_int(plan: Dict[str, Any], key: str) -> Optional[int]:
        if key not in plan or plan.get(key) is None:
            return None
        try:
            return int(plan.get(key))
        except Exception:
            return None

    orders_planned = _plan_int(primary_plan, "orders_planned")
    if orders_planned is None:
        orders_planned = _plan_int(exec_plan, "orders_planned")
    if orders_planned is None:
        orders_planned = _plan_int(manage_plan, "orders_planned")
    orders_planned = orders_planned or 0

    orders_executed = _plan_int(primary_plan, "orders_executed")
    if orders_executed is None:
        orders_executed = _plan_int(exec_plan, "orders_executed")
    if orders_executed is None:
        orders_executed = _plan_int(manage_plan, "orders_executed")
    orders_executed = orders_executed or 0
    paradox = (
        bool(ctx["alloc_summary"].get("available_slots", 0) > 0)
        and bool(ctx["alloc_summary"].get("open_new_candidates", 0) > 0)
        and orders_planned == 0
        and orders_executed == 0
    )

    return {
        "TOTAL_CANDIDATES": len(actionable),
        "EXECUTABLE_NOW": len(executable),
        "TOTAL_BLOCKED": len(blocked),
        "TOP_BLOCKING_LAYER": top_layer,
        "TOP_BLOCKING_REASON": top_reason,
        "FIRST_SYMBOL_THAT_COULD_EXECUTE_IF_BLOCKER_REMOVED": first_if_unblocked,
        "FIRST_EXECUTABLE_SYMBOL": first_executable,
        "execution_funnel": funnel,
        "observed_orders_planned": orders_planned,
        "observed_orders_executed": orders_executed,
        "observed_manage_orders_planned": int(manage_plan.get("orders_planned") or 0),
        "observed_execute_trades_planned": int(exec_plan.get("orders_planned") or 0),
        "available_slots": ctx["alloc_summary"].get("available_slots"),
        "open_new_candidates": ctx["alloc_summary"].get("open_new_candidates"),
        "paradox_detected": paradox,
        "root_cause_hypothesis": _root_cause(rows, ctx, blockers),
    }


def _root_cause(
    rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    blockers: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    gov = not bool(ctx["auth"].get("governance_authorized"))
    esc = str(ctx["cpe"].get("escalation_state") or "UNKNOWN")
    dry = bool((ctx["gate"].get("dry_run_decision") or {}).get("dry_run", True))
    mismatches = [r for r in rows if r.get("lifecycle_mismatch")]

    if gov:
        parts.append(
            f"System governance_authorized=false (escalation {esc}) blocks final_submit_allowed "
            "even when execution_authorized=true — paper planner may build plans but submit gate holds."
        )
    if dry:
        parts.append(
            "execute_trades runs in dry_run mode (--execute not passed); "
            "maybe_execute_plan returns before order placement."
        )
    if mismatches:
        syms = ", ".join(
            f"{r['symbol']}({r.get('allocation_effective_stance')}→{r.get('effective_stance')})"
            for r in mismatches
        )
        parts.append(
            f"Lifecycle/allocation drift on {syms}: allocation OPEN_NEW/ADD stances stale vs "
            "signal_lifecycle_effective WAIT/TRIM — lifecycle layer blocks downstream."
        )
    lc_blocked = [r for r in rows if r.get("final_status") == "BLOCKED_BY_LIFECYCLE"]
    if lc_blocked:
        parts.append(
            f"Lifecycle blocks {len(lc_blocked)} candidate(s): "
            + "; ".join(f"{r['symbol']}:{r['blocking_reason']}" for r in lc_blocked[:3])
        )
    if blockers:
        top = blockers[0]
        parts.append(
            f"Primary blocker category: {top['blocker_category']} "
            f"({top['count']} symbols, {top['percent_of_candidates']}% of actionable candidates)."
        )
    if not parts:
        return "No single dominant blocker; inspect per-symbol trace."
    return " ".join(parts)


def run_audit() -> Dict[str, Any]:
    ctx = _load_inputs()
    symbols = _candidate_symbols(ctx)
    rows = [_evaluate_candidate(sym, ctx) for sym in symbols]
    blockers = _aggregate_blockers(rows)
    summary = _summary(rows, ctx)
    ts = _now_iso()

    trace_cols = [
        "symbol",
        "recommended_action",
        "allocation_pass",
        "governance_pass",
        "execution_pass",
        "lifecycle_pass",
        "risk_pass",
        "capital_pass",
        "placement_pass",
        "final_status",
        "blocking_reason",
    ]
    trace_df = pd.DataFrame(rows)[trace_cols] if rows else pd.DataFrame(columns=trace_cols)
    trace_df.to_csv(OUT_TRACE, index=False, encoding="utf-8")
    pd.DataFrame(blockers).to_csv(OUT_BLOCKERS, index=False, encoding="utf-8")

    report = {
        "generated_at": ts,
        "phase": "179",
        "mode": "allocation_execution_path_audit",
        "disclaimer": (
            "Diagnostic trace only — does not modify governance, lifecycle, execution, "
            "broker, allocation, or risk controls. No orders placed."
        ),
        "summary": summary,
        "execution_context": {
            "governance_authorized": ctx["auth"].get("governance_authorized"),
            "execution_authorized": ctx["auth"].get("execution_authorized"),
            "readiness_status": ctx["readiness"].get("readiness_status"),
            "escalation_state": ctx["cpe"].get("escalation_state"),
            "dry_run": (ctx["gate"].get("dry_run_decision") or {}).get("dry_run"),
            "final_submit_allowed": (ctx["gate"].get("paper_execution_gate") or {}).get(
                "final_submit_allowed"
            ),
            "input_paths": ctx["paths"],
        },
        "candidate_traces": rows,
        "blocker_analysis": blockers,
        "root_cause_analysis": {
            "hypothesis": summary.get("root_cause_hypothesis"),
            "paradox_detected": summary.get("paradox_detected"),
            "top_blocking_layer": summary.get("TOP_BLOCKING_LAYER"),
            "top_blocking_reason": summary.get("TOP_BLOCKING_REASON"),
        },
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only audit — no trades, orders, or rule modifications",
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_TRACE.name}, {OUT_BLOCKERS.name}")
    _log(
        f"candidates={summary['TOTAL_CANDIDATES']} executable={summary['EXECUTABLE_NOW']} "
        f"blocked={summary['TOTAL_BLOCKED']} top={summary['TOP_BLOCKING_LAYER']}"
    )
    return report


def main() -> None:
    run_audit()


if __name__ == "__main__":
    main()
