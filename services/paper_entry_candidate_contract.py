"""
paper_entry_candidate_contract.py
Phase 196T — Governed paper-entry candidate source map and contract (read-only).

Does not arm paper, submit orders, or weaken risk/allocation gates.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"

TRADE_OPPORTUNITIES = RESULTS / "trade_opportunities.csv"
ALLOC_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
ALLOC_RECOMMENDATIONS = RESULTS / "portfolio_allocation_recommendations.csv"
RISK_OVERLAY = RESULTS / "performance_risk_overlay.csv"
LIFECYCLE_EFFECTIVE = RESULTS / "signal_lifecycle_effective.csv"
CERTIFICATION = RESULTS / "capital_preservation_certification.json"
PAPER_ENTRY_AUDIT = RESULTS / "paper_entry_loop_stage_audit.json"
LEGACY_CANDIDATES_CSV = RESULTS / "controlled_paper_entry_loop_candidates.csv"
LEGACY_CANDIDATES_META = RESULTS / "controlled_paper_entry_loop_candidates.meta.json"

SOURCE_MAP_ROWS: List[Dict[str, str]] = [
    {
        "source_name": "trade_opportunities",
        "path": "data/results/trade_opportunities.csv",
        "writer": "services/trade_opportunity pipeline",
        "reader": "paper_entry_loop_stage, execute_trades",
        "freshness_expectation": "cycle-aligned with signals/lifecycle",
        "candidate_meaning": "Raw ranked opportunities (ENTRY/ADD/BUY rows)",
        "risk_overlay_applied": "no at source",
        "allocation_action_applied": "no at source",
        "certification_applied": "no",
        "paper_arm_applied": "no",
    },
    {
        "source_name": "portfolio_allocation_recommendations",
        "path": "data/results/portfolio_allocation_recommendations.csv",
        "writer": "services/portfolio_allocation_engine.py",
        "reader": "paper_entry_loop_stage entry gate",
        "freshness_expectation": "cycle-aligned",
        "candidate_meaning": "OPEN_NEW/ADD/WATCH/BLOCK/HOLD/EXIT advisory actions",
        "risk_overlay_applied": "yes when action=BLOCK",
        "allocation_action_applied": "yes",
        "certification_applied": "no",
        "paper_arm_applied": "no",
    },
    {
        "source_name": "portfolio_allocation_summary",
        "path": "data/results/portfolio_allocation_summary.json",
        "writer": "services/portfolio_allocation_engine.py",
        "reader": "dashboard, phase validation",
        "freshness_expectation": "cycle-aligned",
        "candidate_meaning": "Aggregate open_new/add/exit/block counts",
        "risk_overlay_applied": "indirect via blocked counts",
        "allocation_action_applied": "yes",
        "certification_applied": "no",
        "paper_arm_applied": "no",
    },
    {
        "source_name": "performance_risk_overlay",
        "path": "data/results/performance_risk_overlay.csv",
        "writer": "services/performance_risk_overlay.py",
        "reader": "paper_entry_loop_stage, final risk overlay",
        "freshness_expectation": "cycle-aligned",
        "candidate_meaning": "BLOCK_NEW_BUY / FORCE_EXIT flags (authoritative for entry block)",
        "risk_overlay_applied": "yes",
        "allocation_action_applied": "no",
        "certification_applied": "no",
        "paper_arm_applied": "no",
    },
    {
        "source_name": "signal_lifecycle_effective",
        "path": "data/results/signal_lifecycle_effective.csv",
        "writer": "lifecycle reconciliation pipeline",
        "reader": "execute_trades planner gate",
        "freshness_expectation": "cycle-aligned",
        "candidate_meaning": "Effective stance/state per symbol",
        "risk_overlay_applied": "no",
        "allocation_action_applied": "no",
        "certification_applied": "no",
        "paper_arm_applied": "no",
    },
    {
        "source_name": "paper_entry_loop_stage",
        "path": "services/paper_entry_loop_stage.py",
        "writer": "continuous loop stage",
        "reader": "continuous loop, audit json",
        "freshness_expectation": "per loop cycle",
        "candidate_meaning": "OPEN_NEW/ADD rows from portfolio_allocation_recommendations.csv",
        "risk_overlay_applied": "yes before planner",
        "allocation_action_applied": "yes (BLOCK action)",
        "certification_applied": "no",
        "paper_arm_applied": "yes (ASSISTED latch)",
    },
    {
        "source_name": "execute_trades_planner_preview",
        "path": "services/execute_trades.preview_planned_buy_symbols",
        "writer": "execute_trades build_execution_plan",
        "reader": "paper_entry_loop_stage final selection",
        "freshness_expectation": "on-demand per gate eval",
        "candidate_meaning": "Score-ranked BUY plan (sector/diversification filtered)",
        "risk_overlay_applied": "no before rank; governed metadata added in 196T",
        "allocation_action_applied": "no in planner rank",
        "certification_applied": "no",
        "paper_arm_applied": "no",
    },
    {
        "source_name": "paper_entry_final_risk_overlay",
        "path": "services/paper_entry_loop_stage._risk_blocked_in_selection",
        "writer": "paper_entry_loop_stage",
        "reader": "paper_entry_loop_stage",
        "freshness_expectation": "after planner preview each cycle",
        "candidate_meaning": "Authoritative BLOCK_NEW_BUY/FORCE_EXIT block on final symbols",
        "risk_overlay_applied": "yes authoritative",
        "allocation_action_applied": "no",
        "certification_applied": "no",
        "paper_arm_applied": "no",
    },
    {
        "source_name": "capital_preservation_certification",
        "path": "data/results/capital_preservation_certification.json",
        "writer": "certification pipeline",
        "reader": "dashboard, governance display",
        "freshness_expectation": "periodic refresh",
        "candidate_meaning": "Governance certification status (not runtime freshness)",
        "risk_overlay_applied": "no",
        "allocation_action_applied": "no",
        "certification_applied": "yes",
        "paper_arm_applied": "no",
    },
    {
        "source_name": "controlled_paper_entry_loop_candidates",
        "path": "data/results/controlled_paper_entry_loop_candidates.csv",
        "writer": "phase_195y validation snapshot only",
        "reader": "diagnostic/historical only (not executable)",
        "freshness_expectation": "195Y snapshot; stale if mismatches live entry path",
        "candidate_meaning": "Legacy validation export — not an execution source",
        "risk_overlay_applied": "at snapshot time only",
        "allocation_action_applied": "at snapshot time only",
        "certification_applied": "no",
        "paper_arm_applied": "no",
    },
]


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _load_risk_flags() -> Dict[str, str]:
    import pandas as pd

    out: Dict[str, str] = {}
    if not RISK_OVERLAY.is_file():
        return out
    try:
        ro = pd.read_csv(RISK_OVERLAY, low_memory=False)
        sym_col = "ticker" if "ticker" in ro.columns else "symbol"
        flag_col = "risk_flag" if "risk_flag" in ro.columns else "drag_flag"
        for _, row in ro.iterrows():
            sym = _norm(row.get(sym_col))
            if sym:
                out[sym] = str(row.get(flag_col) or "OK")
    except Exception:
        pass
    return out


def _risk_set(flag: str) -> set[str]:
    return {p for p in (_norm(x) for x in str(flag).split("|")) if p and p != "OK"}


def is_risk_blocked_symbol(symbol: str, risk_flags: Optional[Dict[str, str]] = None) -> bool:
    flags = risk_flags if risk_flags is not None else _load_risk_flags()
    rs = _risk_set(flags.get(_norm(symbol), "OK"))
    return "BLOCK_NEW_BUY" in rs or "FORCE_EXIT" in rs


def assess_controlled_candidates_cache() -> Dict[str, Any]:
    """Return staleness assessment for legacy 195Y candidates CSV."""
    result: Dict[str, Any] = {
        "path": str(LEGACY_CANDIDATES_CSV.relative_to(ROOT)).replace("\\", "/"),
        "exists": LEGACY_CANDIDATES_CSV.is_file(),
        "stale": False,
        "executable_source": False,
        "diagnostic_only": True,
        "reason": "",
    }
    if not result["exists"]:
        result["reason"] = "file missing; not used for execution"
        return result

    try:
        from services.paper_entry_loop_stage import _load_governed_allocation_candidates

        live_candidates, handoff = _load_governed_allocation_candidates()
        live_syms = {_norm(c.get("ticker")) for c in live_candidates}

        import pandas as pd

        stale_df = pd.read_csv(LEGACY_CANDIDATES_CSV, low_memory=False)
        stale_syms = set()
        if "ticker" in stale_df.columns:
            stale_syms = {str(x).strip().upper() for x in stale_df["ticker"].dropna()}

        extra = sorted(stale_syms - live_syms)
        missing = sorted(live_syms - stale_syms)
        mtime = datetime.fromtimestamp(
            LEGACY_CANDIDATES_CSV.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        result["snapshot_mtime_utc"] = mtime
        result["stale_symbols_extra"] = extra
        result["stale_symbols_missing_vs_live"] = missing

        if extra or missing:
            result["stale"] = True
            result["reason"] = (
                "195Y validation snapshot diverges from live paper_entry_loop_stage "
                f"(extra={extra}, missing={missing}); diagnostic only"
            )
        else:
            result["reason"] = "symbols match live entry path but file remains diagnostic-only"

        meta = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stale": result["stale"],
            "executable_source": False,
            "diagnostic_only": True,
            "reason": result["reason"],
            "live_entry_symbols": sorted(live_syms),
            "snapshot_symbols": sorted(stale_syms),
        }
        LEGACY_CANDIDATES_META.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_CANDIDATES_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        result["meta_path"] = str(LEGACY_CANDIDATES_META.relative_to(ROOT)).replace("\\", "/")
    except Exception as exc:
        result["stale"] = True
        result["reason"] = f"staleness check error: {exc}"

    return result


def enrich_planner_preview_metadata(
    symbols: List[str],
    meta: Dict[str, Any],
    *,
    alloc_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Add governed preview fields for audit/display.
    Does not change planner symbol list or execution plan.
    """
    risk_flags = _load_risk_flags()
    alloc = alloc_summary if alloc_summary is not None else _load_json(ALLOC_SUMMARY)
    open_new = int(alloc.get("open_new_candidates") or 0)
    add_count = int(alloc.get("add_candidates") or 0)

    raw_top = _norm(symbols[0]) if symbols else ""
    raw_flag = risk_flags.get(raw_top, "OK") if raw_top else ""
    raw_blocked = is_risk_blocked_symbol(raw_top, risk_flags) if raw_top else False

    governed_top = ""
    for sym in symbols:
        s = _norm(sym)
        if s and not is_risk_blocked_symbol(s, risk_flags):
            governed_top = s
            break

    symbols_governed = [s for s in symbols if not is_risk_blocked_symbol(s, risk_flags)]
    executable = bool(raw_top and not raw_blocked and (open_new > 0 or add_count > 0))

    warnings: List[str] = []
    if raw_top and raw_blocked:
        warnings.append(
            "planner preview candidate is not executable because final risk overlay blocks it"
        )
    if raw_top and not raw_blocked and open_new == 0 and add_count == 0:
        warnings.append(
            "allocation open_new=0 and add=0 — no actionable deployment slots despite signals"
        )
    if raw_top and governed_top and raw_top != governed_top:
        warnings.append(
            f"raw planner top {raw_top} differs from governed executable preview {governed_top or 'none'}"
        )

    final_status = "BLOCKED"
    final_reason = ""
    if not raw_top:
        final_status = "NO_PLANNER_CANDIDATE"
    elif raw_blocked:
        final_status = "BLOCKED"
        final_reason = f"FINAL_SELECTION_RISK_OVERLAY_BLOCK:{raw_top}"
    elif open_new == 0 and add_count == 0:
        final_status = "NOT_ACTIONABLE"
        final_reason = "allocation_open_new_and_add_zero"
    else:
        final_status = "PENDING_GOVERNANCE"
        final_reason = "passes risk overlay; requires arm+approval gates"

    meta["preview_candidate"] = raw_top
    meta["preview_candidate_risk_flag"] = raw_flag
    meta["preview_candidate_executable"] = executable and not raw_blocked
    meta["governed_executable_preview_candidate"] = governed_top
    meta["symbols_governed"] = symbols_governed
    meta["allocation_open_new_count"] = open_new
    meta["allocation_add_count"] = add_count
    meta["preview_warnings"] = warnings
    meta["allocation_action"] = _load_alloc_action(raw_top) if raw_top else ""
    meta["final_selection_status"] = final_status
    meta["final_selection_block_reason"] = final_reason
    meta["governed_preview_note"] = (
        "Raw planner rank may include risk-blocked symbols; "
        "governed_executable_preview_candidate excludes BLOCK_NEW_BUY/FORCE_EXIT."
    )
    return meta


def _load_alloc_action(symbol: str) -> str:
    import pandas as pd

    if not ALLOC_RECOMMENDATIONS.is_file():
        return ""
    try:
        al = pd.read_csv(ALLOC_RECOMMENDATIONS, low_memory=False)
        row = al[al["ticker"].astype(str).str.upper() == _norm(symbol)]
        if row.empty:
            return ""
        return str(row.iloc[0].get("recommended_action") or "")
    except Exception:
        return ""


def _load_lifecycle_action(symbol: str) -> Tuple[str, str]:
    import pandas as pd

    if not LIFECYCLE_EFFECTIVE.is_file():
        return "", ""
    try:
        le = pd.read_csv(LIFECYCLE_EFFECTIVE, low_memory=False)
        sym_col = "ticker" if "ticker" in le.columns else "symbol"
        row = le[le[sym_col].astype(str).str.upper() == _norm(symbol)]
        if row.empty:
            return "", ""
        r = row.iloc[0]
        return str(r.get("decision_action") or r.get("effective_stance") or ""), str(
            r.get("effective_stance") or ""
        )
    except Exception:
        return "", ""


def _load_allocation_open_new_add_symbols() -> Tuple[List[str], List[str], Dict[str, str]]:
    import pandas as pd

    open_new: List[str] = []
    add_syms: List[str] = []
    actions: Dict[str, str] = {}
    if not ALLOC_RECOMMENDATIONS.is_file():
        return open_new, add_syms, actions
    try:
        df = pd.read_csv(ALLOC_RECOMMENDATIONS, low_memory=False)
        sym_col = "ticker" if "ticker" in df.columns else "symbol"
        if sym_col not in df.columns:
            return open_new, add_syms, actions
        for _, row in df.iterrows():
            sym = _norm(row.get(sym_col))
            action = _norm(row.get("recommended_action"))
            if not sym:
                continue
            actions[sym] = action
            if action == "OPEN_NEW":
                open_new.append(sym)
            elif action == "ADD":
                add_syms.append(sym)
    except Exception:
        pass
    return sorted(set(open_new)), sorted(set(add_syms)), actions


def build_candidate_contract_rows() -> List[Dict[str, str]]:
    """Build governed candidate contract rows for all relevant symbols."""
    from services.paper_arm_status import build_paper_arm_status
    from services.paper_entry_loop_stage import _load_governed_allocation_candidates
    from services.execute_trades import preview_planned_buy_symbols

    cert = _load_json(CERTIFICATION)
    cert_status = str(cert.get("certification_status") or "UNKNOWN")
    arm_st = build_paper_arm_status(verbose_notes=False)
    arm_status = str(arm_st.get("status") or "UNKNOWN")
    alloc = _load_json(ALLOC_SUMMARY)
    risk_flags = _load_risk_flags()

    entry_candidates, handoff = _load_governed_allocation_candidates()
    eligible_alloc = handoff.get("eligible_entry_symbols") or []
    planner_symbols, planner_meta = preview_planned_buy_symbols(
        "paper",
        verbose=False,
        for_execute=True,
        source="allocation",
        allowed_symbols=list(eligible_alloc),
    )
    enrich_planner_preview_metadata(planner_symbols, planner_meta, alloc_summary=alloc)

    alloc_open_new, alloc_add, alloc_actions = _load_allocation_open_new_add_symbols()
    symbols_of_interest: List[str] = list(alloc_open_new)
    symbols_of_interest.extend(alloc_add)
    for c in entry_candidates:
        symbols_of_interest.append(_norm(c.get("ticker")))
    for s in planner_symbols:
        symbols_of_interest.append(_norm(s))
    symbols_of_interest = sorted({s for s in symbols_of_interest if s})

    rows: List[Dict[str, str]] = []
    for sym in symbols_of_interest:
        ec = next((c for c in entry_candidates if _norm(c.get("ticker")) == sym), None)
        alloc_action = alloc_actions.get(sym) or _load_alloc_action(sym)
        lifecycle_action, effective_stance = _load_lifecycle_action(sym)
        risk_flag = risk_flags.get(sym, str((ec or {}).get("risk_flag") or "OK"))
        risk_blocked = is_risk_blocked_symbol(sym, risk_flags)

        if ec:
            source = "portfolio_allocation_recommendations"
            entry_eligible = bool(ec.get("eligible_for_auto_entry"))
        elif sym in alloc_open_new or sym in alloc_add:
            source = "portfolio_allocation_recommendations"
            entry_eligible = False
        else:
            source = "planner_intake_only"
            entry_eligible = False

        if sym in planner_symbols:
            if sym == planner_meta.get("preview_candidate"):
                planner_status = "PLANNER_RAW_TOP"
            elif sym in (planner_meta.get("symbols_governed") or []):
                planner_status = "PLANNER_GOVERNED"
            else:
                planner_status = "PLANNER_INCLUDED"
        else:
            planner_status = ""

        allocation_status = alloc_action or "UNKNOWN"

        if risk_blocked:
            final_status = "BLOCKED"
            rejection = f"FINAL_SELECTION_RISK_OVERLAY_BLOCK:{sym}"
            executable = "false"
        elif alloc_action not in ("OPEN_NEW", "ADD"):
            final_status = "NOT_ACTIONABLE"
            rejection = f"allocation_action={alloc_action or 'UNKNOWN'}"
            executable = "false"
        elif ec and not entry_eligible:
            final_status = "EXCLUDED"
            rejection = str(ec.get("block_reasons") or "allocation_gate_block")
            executable = "false"
        elif sym in planner_symbols and sym not in (handoff.get("eligible_entry_symbols") or []):
            final_status = "CONTRACT_MISMATCH"
            rejection = "planner_symbol_outside_allocation_eligible"
            executable = "false"
        elif entry_eligible and alloc_action in ("OPEN_NEW", "ADD") and not risk_blocked:
            final_status = "ENTRY_ELIGIBLE"
            rejection = ""
            executable = "true"
        else:
            final_status = "REVIEW"
            rejection = ""
            executable = "false"

        rows.append(
            {
                "symbol": sym,
                "source": source,
                "allocation_action": alloc_action,
                "lifecycle_action": lifecycle_action,
                "effective_stance": effective_stance,
                "risk_flag": risk_flag,
                "allocation_status": allocation_status,
                "planner_status": planner_status,
                "certification_status": cert_status,
                "paper_arm_status": arm_status,
                "final_selection_status": final_status,
                "rejection_reason": rejection,
                "executable_candidate": executable,
            }
        )
    return rows


def build_governed_candidate_summary() -> Dict[str, Any]:
    """Summary payload for dashboard and validation."""
    from services.execute_trades import preview_planned_buy_symbols
    from services.paper_entry_loop_stage import _load_governed_allocation_candidates

    alloc = _load_json(ALLOC_SUMMARY)
    cache = assess_controlled_candidates_cache()
    contract_rows = build_candidate_contract_rows()
    _, handoff = _load_governed_allocation_candidates()
    eligible_alloc = handoff.get("eligible_entry_symbols") or []

    planner_symbols, planner_meta = preview_planned_buy_symbols(
        "paper",
        verbose=False,
        for_execute=True,
        source="allocation",
        allowed_symbols=list(eligible_alloc),
    )
    enrich_planner_preview_metadata(planner_symbols, planner_meta, alloc_summary=alloc)

    executable_rows = [r for r in contract_rows if r.get("executable_candidate") == "true"]
    planner_set = {_norm(s) for s in planner_symbols if s}
    contract_mismatch = bool(
        eligible_alloc and planner_set and not planner_set.issubset(set(eligible_alloc))
    )
    could_order = bool(
        executable_rows
        and not contract_mismatch
        and planner_meta.get("preview_candidate_executable")
        and not is_risk_blocked_symbol(str(planner_meta.get("preview_candidate") or ""))
    )

    return {
        "allocation_open_new_count": int(
            handoff.get("allocation_open_new_count") or alloc.get("open_new_candidates") or 0
        ),
        "allocation_add_count": int(
            handoff.get("allocation_add_count") or alloc.get("add_candidates") or 0
        ),
        "allocation_exit_count": int(alloc.get("exit_candidates") or 0),
        "allocation_block_count": int(alloc.get("blocked_candidates") or 0),
        "eligible_allocation_symbols": list(eligible_alloc),
        "rejected_allocation_symbols": handoff.get("rejected_entry_symbols") or {},
        "governed_source": "portfolio_allocation_recommendations.csv",
        "planner_input_mode": planner_meta.get("planner_input_mode"),
        "planner_considered_symbols": planner_meta.get("planner_considered_symbols") or [],
        "planner_selected_symbols": planner_meta.get("planner_selected_symbols") or planner_symbols,
        "planner_rejected_symbols": planner_meta.get("planner_rejected_symbols") or {},
        "planner_preview_candidate": str(planner_meta.get("preview_candidate") or ""),
        "planner_preview_risk_flag": str(planner_meta.get("preview_candidate_risk_flag") or ""),
        "planner_preview_executable": bool(planner_meta.get("preview_candidate_executable")),
        "governed_executable_preview_candidate": str(
            planner_meta.get("governed_executable_preview_candidate") or ""
        ),
        "preview_warnings": list(planner_meta.get("preview_warnings") or []),
        "final_selection_status": str(planner_meta.get("final_selection_status") or ""),
        "final_selection_block_reason": str(planner_meta.get("final_selection_block_reason") or ""),
        "contract_mismatch_detected": contract_mismatch,
        "would_execute_if_market_open": could_order,
        "stale_candidate_cache": cache,
        "contract_rows": contract_rows,
        "executable_candidate_count": len(executable_rows),
        "arming_could_allow_paper_order_next_cycle": could_order,
        "source_map_rows": SOURCE_MAP_ROWS,
    }


def write_candidate_artifacts(
    *,
    source_map_path: Path,
    contract_path: Path,
) -> Tuple[int, int]:
    summary = build_governed_candidate_summary()
    source_map_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)

    with source_map_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(SOURCE_MAP_ROWS[0].keys()),
        )
        writer.writeheader()
        writer.writerows(SOURCE_MAP_ROWS)

    contract_fields = [
        "symbol",
        "source",
        "allocation_action",
        "lifecycle_action",
        "effective_stance",
        "risk_flag",
        "allocation_status",
        "planner_status",
        "certification_status",
        "paper_arm_status",
        "final_selection_status",
        "rejection_reason",
        "executable_candidate",
    ]
    with contract_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=contract_fields)
        writer.writeheader()
        writer.writerows(summary["contract_rows"])

    return len(SOURCE_MAP_ROWS), len(summary["contract_rows"])
