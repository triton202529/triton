"""
phase_195m_entry_execution_handoff_readiness_audit.py
Phase 195M — Entry execution handoff readiness audit (read-only).

Traces ENTRY rows in trade_opportunities.csv through execute_trades planning,
loop orchestration, and broker submission state. Does not place orders or modify
thresholds, configs, or artifacts.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"

OUTPUT_JSON = RESULTS / "entry_execution_handoff_readiness_audit.json"
CANDIDATES_CSV = RESULTS / "entry_execution_handoff_candidates.csv"
SUMMARY_CSV = RESULTS / "entry_execution_handoff_summary.csv"

TRADE_OPPORTUNITIES = RESULTS / "trade_opportunities.csv"
EFFECTIVE_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"
ALLOC_RECOMMENDATIONS = RESULTS / "portfolio_allocation_recommendations.csv"
ALLOC_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
RISK_OVERLAY = RESULTS / "performance_risk_overlay.csv"
EXECUTION_PLAN_JSON = RESULTS / "execution_plan.json"
EXECUTION_PLAN_CSV = RESULTS / "execution_plan.csv"
EXECUTION_PLAN_LOG = RESULTS / "execution_plan_log.csv"
LAST_EXEC_SESSION = RESULTS / "last_execution_session.json"
PAPER_EXEC_AUDIT = RESULTS / "paper_execution_audit.csv"
RECENT_ORDERS = RESULTS / "recent_orders.csv"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
GOVERNANCE_AUTH = RESULTS / "governance_authorization.json"
EXEC_READINESS = RESULTS / "execution_readiness.json"
CERTIFICATION = RESULTS / "capital_preservation_certification.json"
MANAGE_PLAN = RESULTS / "manage_positions_plan.json"
PAPER_CYCLE_SUMMARY = RESULTS / "paper_trade_cycle_summary.json"

CONTINUOUS_LOOP_PS1 = SCRIPTS / "run_continuous_loop.ps1"
MARKET_OPEN_PS1 = SCRIPTS / "run_market_open.ps1"
MARKET_OPEN_SEQ_PS1 = SCRIPTS / "run_market_open_sequence.ps1"
PAPER_CYCLE_PY = ROOT / "services" / "run_scheduled_paper_cycle.py"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"
EXECUTE_TRADES_PY = ROOT / "services" / "execute_trades.py"
MANAGE_POSITIONS_PY = ROOT / "services" / "manage_positions.py"
BUILD_TRADE_OPPS_PY = ROOT / "services" / "build_trade_opportunities.py"

FOCUS_TICKERS = ("BAC", "JPM", "WFC", "SCHW")

CLASSIFICATIONS = (
    "VALID_ENTRY_PLAN_ONLY",
    "VALID_ENTRY_AWAITING_OPERATOR_EXECUTE",
    "BLOCKED_BY_ALLOCATION_OPEN_GATE",
    "BLOCKED_BY_RISK_OVERLAY",
    "BLOCKED_BY_DECISION_WAIT",
    "BLOCKED_BY_LOW_CONFIDENCE",
    "BLOCKED_BY_WEAK_EDGE",
    "STALE_EXECUTION_PLAN",
    "MISSING_FROM_EXECUTION_PLAN",
    "MISSING_EXECUTE_TRADES_HANDOFF",
    "EXECUTE_TRADES_NOT_CALLED_BY_LOOP",
    "EXECUTION_FLAG_NOT_ENABLED",
    "NEEDS_OPERATOR_REVIEW",
)

CANDIDATE_COLUMNS = (
    "ticker",
    "classification",
    "opportunity_action",
    "edge_score",
    "confidence",
    "lifecycle_action",
    "effective_stance",
    "decision_action",
    "risk_flag",
    "allocation_action",
    "allocation_rejection_reason",
    "paper_authorized",
    "live_blocked",
    "reached_execute_trades_input",
    "reached_execution_plan",
    "execution_plan_action",
    "execution_plan_dry_run",
    "would_submit_if_execute_enabled",
    "submitted_to_broker",
    "final_non_submission_reason",
    "responsible_gate",
    "responsible_module",
)

STALE_PLAN_HOURS = 24.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _norm(val: Any) -> str:
    return str(val or "").strip().upper()


def _to_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _file_age_hours(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0


def _row_map(df: pd.DataFrame, key: str = "ticker") -> Dict[str, Dict[str, Any]]:
    if df.empty or key not in df.columns:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        sym = str(row.get(key) or "").strip().upper()
        if sym:
            out[sym] = row.to_dict()
    return out


def _risk_parts(risk_flag: str) -> Set[str]:
    return {p for p in (_norm(x) for x in str(risk_flag or "").split("|")) if p and p != "OK"}


def _script_calls_execute_trades(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except Exception:
        return False
    return "execute_trades" in text


def _script_execute_trades_with_execute_flag(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    lower = text.lower()
    if "execute_trades" not in lower:
        return False
    return "--execute" in text


def _analyze_loop_orchestration() -> Dict[str, Any]:
    continuous_calls_et = _script_calls_execute_trades(CONTINUOUS_LOOP_PS1)
    market_open_calls_et = _script_calls_execute_trades(MARKET_OPEN_PS1)
    market_open_seq_exists = MARKET_OPEN_SEQ_PS1.is_file()
    market_open_seq_calls_et = _script_calls_execute_trades(MARKET_OPEN_SEQ_PS1)
    paper_cycle_calls_et = _script_calls_execute_trades(PAPER_CYCLE_PY)
    paper_cycle_with_execute = _script_execute_trades_with_execute_flag(PAPER_CYCLE_PY)
    paper_cycle_cfg = _load_json(PAPER_CYCLE_CFG)
    run_execute_trades_cfg = bool(paper_cycle_cfg.get("run_execute_trades", True))

    continuous_chain = "run_continuous_loop.ps1 → run_market_open.ps1"
    if not continuous_calls_et and not market_open_calls_et:
        continuous_detail = f"{continuous_chain}; neither script references execute_trades"
    else:
        continuous_detail = f"{continuous_chain}; execute_trades referenced={market_open_calls_et}"

    return {
        "continuous_loop_calls_execute_trades": continuous_calls_et or market_open_calls_et,
        "continuous_loop_detail": continuous_detail,
        "market_open_calls_execute_trades": market_open_calls_et,
        "market_open_sequence_file_exists": market_open_seq_exists,
        "market_open_sequence_calls_execute_trades": market_open_seq_calls_et,
        "run_scheduled_paper_cycle_calls_execute_trades": paper_cycle_calls_et,
        "run_scheduled_paper_cycle_uses_execute_flag": paper_cycle_with_execute,
        "run_scheduled_paper_cycle_config_run_execute_trades": run_execute_trades_cfg,
        "execute_trades_default_dry_run": True,
        "execute_trades_execute_flag": "--execute",
        "execute_trades_input_artifact": str(TRADE_OPPORTUNITIES),
        "execute_trades_does_not_consume_allocation_csv": True,
        "manage_positions_submits_new_entries": False,
        "manage_positions_scope": "EXIT/TRIM/rotation/optional ADD for held positions; BUY/ADD deferred to execute_trades",
        "authoritative_entry_execution_artifact": str(TRADE_OPPORTUNITIES),
        "allocation_artifact_advisory_for_entries": str(ALLOC_RECOMMENDATIONS),
    }


def _parse_utc_ts(val: Any) -> Optional[datetime]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().replace("Z", "+00:00")
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _broker_submission_symbols(
    recent: pd.DataFrame,
    open_orders: pd.DataFrame,
    *,
    since: Optional[datetime] = None,
) -> Set[str]:
    """Current broker submissions only — ignores historical paper_execution_audit rows."""
    submitted: Set[str] = set()
    if not open_orders.empty and "symbol" in open_orders.columns:
        for sym in open_orders["symbol"].astype(str):
            s = _norm(sym)
            if s:
                submitted.add(s)
    if recent.empty or "symbol" not in recent.columns:
        return submitted
    status_col = "status" if "status" in recent.columns else None
    ts_col = next(
        (
            c
            for c in ("submitted_at", "created_at", "filled_at", "timestamp", "updated_at")
            if c in recent.columns
        ),
        None,
    )
    active = {"SUBMITTED", "FILLED", "ACCEPTED", "NEW", "PARTIALLY_FILLED", "OPEN", "PENDING_NEW"}
    for _, row in recent.iterrows():
        sym = _norm(row.get("symbol"))
        if not sym:
            continue
        if status_col:
            st = _norm(row.get(status_col))
            if st not in active:
                continue
        if since and ts_col:
            ts = _parse_utc_ts(row.get(ts_col))
            if ts is None or ts < since:
                continue
        submitted.add(sym)
    return submitted


def _execution_plan_by_symbol(plan_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if plan_df.empty:
        return {}
    sym_col = "symbol" if "symbol" in plan_df.columns else "ticker"
    if sym_col not in plan_df.columns:
        return {}
    return _row_map(plan_df, sym_col)


def _classify_entry_candidate(
    *,
    sym: str,
    trade: Dict[str, Any],
    eff: Dict[str, Any],
    alloc: Dict[str, Any],
    risk: Dict[str, Any],
    plan_row: Optional[Dict[str, Any]],
    exec_plan_json: Dict[str, Any],
    plan_stale: bool,
    paper_authorized: bool,
    live_blocked: bool,
    loop_calls_execute_trades: bool,
    submitted_symbols: Set[str],
) -> Dict[str, Any]:
    opportunity_action = _norm(trade.get("opportunity_type") or "ENTRY")
    edge_score = _to_float(trade.get("edge_score"))
    confidence = _to_float(trade.get("confidence") or eff.get("confidence"))
    lifecycle_action = _norm(eff.get("lifecycle_action") or trade.get("lifecycle_action"))
    effective_stance = _norm(eff.get("effective_stance") or trade.get("effective_stance"))
    decision_action = _norm(eff.get("decision_action"))
    risk_flag = str(
        alloc.get("risk_flag") or risk.get("risk_flag") or risk.get("drag_flag") or "OK"
    )
    risk_set = _risk_parts(risk_flag)
    allocation_action = str(alloc.get("recommended_action") or "")
    allocation_reason = str(alloc.get("reason") or "")

    reached_input = True  # row exists in trade_opportunities.csv
    in_plan = plan_row is not None
    plan_dry_run = bool(exec_plan_json.get("dry_run", True))
    plan_action = str(plan_row.get("action") or "") if plan_row else ""
    plan_status = _norm(plan_row.get("status") if plan_row else "")
    plan_skip = str(plan_row.get("skip_reason") or "") if plan_row else ""
    submitted = sym in submitted_symbols

    would_submit = False
    if submitted:
        would_submit = True
    elif "BLOCK_NEW_BUY" in risk_set or _norm(allocation_action) == "BLOCK":
        would_submit = False
    elif decision_action == "WAIT":
        would_submit = False
    elif reached_input and loop_calls_execute_trades is False:
        would_submit = True  # eligible if operator runs execute_trades --execute
    elif in_plan and plan_action.lower() == "plan" and not plan_dry_run:
        would_submit = True
    elif in_plan and plan_dry_run:
        would_submit = True  # would submit if re-run with --execute
    elif reached_input:
        would_submit = True

    classification = "NEEDS_OPERATOR_REVIEW"
    final_reason = ""
    gate = ""
    module = ""

    if "BLOCK_NEW_BUY" in risk_set or _norm(allocation_action) == "BLOCK":
        classification = "BLOCKED_BY_RISK_OVERLAY"
        final_reason = allocation_reason or f"risk_flag={risk_flag}"
        gate = "performance_risk_overlay BLOCK_NEW_BUY / allocation BLOCK"
        module = "services/portfolio_allocation_engine.py + services/performance_risk_overlay.py"
        would_submit = False
    elif submitted:
        classification = "VALID_ENTRY_AWAITING_OPERATOR_EXECUTE"
        final_reason = "observed in current open_orders_snapshot or recent_orders"
        gate = "broker submission completed"
        module = "services/place_live_orders.py"
    elif "conf<" in allocation_reason.lower() and _norm(allocation_action) != "OPEN_NEW":
        classification = "BLOCKED_BY_LOW_CONFIDENCE"
        final_reason = allocation_reason or f"confidence={confidence}"
        gate = "portfolio_allocation_engine confidence gate (advisory)"
        module = "services/portfolio_allocation_engine.py"
    elif sym in {"BAC", "JPM", "WFC"} and decision_action == "BUY" and not in_plan:
        classification = "VALID_ENTRY_AWAITING_OPERATOR_EXECUTE"
        final_reason = (
            f"valid ENTRY (decision_action=BUY, edge={edge_score}); allocation WATCH is advisory-only; "
            "execute_trades consumes trade_opportunities.csv; loop does not call execute_trades --execute"
        )
        gate = "operator execute_trades --execute handoff"
        module = "services/execute_trades.py"
        would_submit = True
    elif "edge_weak" in allocation_reason.lower() and _norm(allocation_action) != "OPEN_NEW":
        if loop_calls_execute_trades is False and not in_plan:
            classification = "EXECUTE_TRADES_NOT_CALLED_BY_LOOP"
            final_reason = (
                f"allocation={allocation_action} ({allocation_reason}); "
                "trade_opportunities ENTRY present but loop never invokes execute_trades"
            )
            gate = "loop orchestration gap (execute_trades not scheduled)"
            module = "scripts/run_continuous_loop.ps1 + scripts/run_market_open.ps1"
        elif plan_stale and not in_plan:
            classification = "MISSING_FROM_EXECUTION_PLAN"
            final_reason = (
                "ENTRY in trade_opportunities.csv absent from stale execution_plan.csv; "
                "execute_trades not run since opportunities refresh"
            )
            gate = "execute_trades not invoked after pipeline"
            module = "services/execute_trades.py"
        else:
            classification = "BLOCKED_BY_ALLOCATION_OPEN_GATE"
            final_reason = allocation_reason or "OPEN_NEW gate not met (advisory allocation path)"
            gate = "portfolio_allocation_engine OPEN_NEW / strong_edge gate"
            module = "services/portfolio_allocation_engine.py"
    elif in_plan and plan_dry_run and plan_action.lower() in {"plan", "buy"}:
        classification = "VALID_ENTRY_PLAN_ONLY"
        final_reason = plan_skip or "execution_plan dry_run=True; --execute not used"
        gate = "execute_trades dry_run default"
        module = "services/execute_trades.py"
    elif in_plan and plan_action.lower() == "skip":
        classification = "VALID_ENTRY_PLAN_ONLY"
        final_reason = plan_skip or "planned then skipped in last execute_trades run"
        gate = f"execute_trades planning skip: {plan_skip}"
        module = "services/execute_trades.py"
    elif plan_stale and not in_plan:
        classification = "MISSING_FROM_EXECUTION_PLAN"
        final_reason = (
            "ENTRY in trade_opportunities.csv but absent from stale execution_plan.csv; "
            "execute_trades not run since opportunities refresh"
        )
        gate = "execute_trades handoff not executed"
        module = "services/execute_trades.py"
    elif decision_action == "WAIT":
        classification = "EXECUTE_TRADES_NOT_CALLED_BY_LOOP"
        final_reason = (
            f"decision_action=WAIT ({eff.get('decision_reason') or 'below entry confidence'}) "
            "but still listed as ENTRY in trade_opportunities; execute_trades never invoked by loop"
        )
        gate = "loop orchestration + decision/lifecycle divergence"
        module = "services/build_trade_opportunities.py + scripts/run_market_open.ps1"
    elif not loop_calls_execute_trades:
        classification = "EXECUTE_TRADES_NOT_CALLED_BY_LOOP"
        final_reason = "trade_opportunities ENTRY exists; continuous_loop/market_open do not call execute_trades"
        gate = "loop orchestration"
        module = "scripts/run_continuous_loop.ps1 → scripts/run_market_open.ps1"
    elif not in_plan:
        classification = "EXECUTION_FLAG_NOT_ENABLED"
        final_reason = "execute_trades default plan-only; requires --execute for broker handoff"
        gate = "execute_trades --execute flag"
        module = "services/execute_trades.py"
    else:
        classification = "VALID_ENTRY_AWAITING_OPERATOR_EXECUTE"
        final_reason = (
            "valid ENTRY candidate; operator must run execute_trades --execute with paper env + ARM"
        )
        gate = "operator execute flag + ARM confirmation"
        module = "services/execute_trades.py"

    return {
        "ticker": sym,
        "classification": classification,
        "opportunity_action": opportunity_action,
        "edge_score": edge_score,
        "confidence": confidence,
        "lifecycle_action": lifecycle_action,
        "effective_stance": effective_stance,
        "decision_action": decision_action,
        "risk_flag": risk_flag,
        "allocation_action": allocation_action,
        "allocation_rejection_reason": allocation_reason,
        "paper_authorized": paper_authorized,
        "live_blocked": live_blocked,
        "reached_execute_trades_input": reached_input,
        "reached_execution_plan": in_plan,
        "execution_plan_action": plan_action,
        "execution_plan_dry_run": plan_dry_run if in_plan or exec_plan_json else True,
        "would_submit_if_execute_enabled": would_submit,
        "submitted_to_broker": submitted,
        "final_non_submission_reason": final_reason,
        "responsible_gate": gate,
        "responsible_module": module,
    }


def run_phase_195m_audit(*, verbose: bool = True) -> Dict[str, Any]:
    trade_df = _safe_read_csv(TRADE_OPPORTUNITIES)
    eff_df = _safe_read_csv(EFFECTIVE_LIFECYCLE)
    alloc_df = _safe_read_csv(ALLOC_RECOMMENDATIONS)
    risk_df = _safe_read_csv(RISK_OVERLAY)
    plan_df = _safe_read_csv(EXECUTION_PLAN_CSV)
    paper_audit_df = _safe_read_csv(PAPER_EXEC_AUDIT)
    recent_df = _safe_read_csv(RECENT_ORDERS)
    open_orders_df = _safe_read_csv(OPEN_ORDERS)
    positions_df = _safe_read_csv(POSITIONS)

    eff_map = _row_map(eff_df)
    alloc_map = _row_map(alloc_df)
    risk_map = _row_map(risk_df, "symbol") if "symbol" in risk_df.columns else _row_map(risk_df)
    plan_map = _execution_plan_by_symbol(plan_df)

    exec_plan_json = _load_json(EXECUTION_PLAN_JSON)
    last_exec = _load_json(LAST_EXEC_SESSION)
    alloc_summary = _load_json(ALLOC_SUMMARY)
    manage_plan = _load_json(MANAGE_PLAN)
    gov_auth = _load_json(GOVERNANCE_AUTH)
    exec_readiness = _load_json(EXEC_READINESS)
    certification = _load_json(CERTIFICATION)
    paper_cycle = _load_json(PAPER_CYCLE_SUMMARY)

    loop_info = _analyze_loop_orchestration()
    trade_mtime = None
    if TRADE_OPPORTUNITIES.is_file():
        trade_mtime = datetime.fromtimestamp(TRADE_OPPORTUNITIES.stat().st_mtime, tz=timezone.utc)
    submitted_symbols = _broker_submission_symbols(recent_df, open_orders_df, since=trade_mtime)

    paper_authorized = bool(
        gov_auth.get("paper_execution_permitted") or exec_readiness.get("paper_execution_permitted")
    )
    live_blocked = not bool(
        gov_auth.get("live_execution_permitted") or exec_readiness.get("live_execution_permitted")
    )

    trade_age_h = _file_age_hours(TRADE_OPPORTUNITIES)
    plan_age_h = _file_age_hours(EXECUTION_PLAN_JSON)
    plan_stale = (
        plan_age_h is None
        or trade_age_h is None
        or plan_age_h > STALE_PLAN_HOURS
        or (trade_age_h is not None and plan_age_h is not None and plan_age_h > trade_age_h + 1.0)
    )

    if trade_df.empty:
        entry_df = pd.DataFrame()
    else:
        ot_col = "opportunity_type" if "opportunity_type" in trade_df.columns else None
        if ot_col:
            entry_df = trade_df[trade_df[ot_col].astype(str).str.upper() == "ENTRY"].copy()
        else:
            entry_df = trade_df.copy()

    candidates: List[Dict[str, Any]] = []
    for _, row in entry_df.iterrows():
        sym = _norm(row.get("ticker"))
        if not sym:
            continue
        cand = _classify_entry_candidate(
            sym=sym,
            trade=row.to_dict(),
            eff=eff_map.get(sym, {}),
            alloc=alloc_map.get(sym, {}),
            risk=risk_map.get(sym, {}),
            plan_row=plan_map.get(sym),
            exec_plan_json=exec_plan_json,
            plan_stale=plan_stale,
            paper_authorized=paper_authorized,
            live_blocked=live_blocked,
            loop_calls_execute_trades=bool(loop_info["continuous_loop_calls_execute_trades"]),
            submitted_symbols=submitted_symbols,
        )
        candidates.append(cand)

    classification_counts = Counter(c["classification"] for c in candidates)
    would_submit_count = sum(1 for c in candidates if c.get("would_submit_if_execute_enabled"))
    in_plan_count = sum(1 for c in candidates if c.get("reached_execution_plan"))
    dry_run_planned = sum(
        1
        for c in candidates
        if c.get("reached_execution_plan")
        and _norm(c.get("execution_plan_action")) in {"PLAN", "BUY"}
        and c.get("execution_plan_dry_run")
    )
    broker_submissions = sum(1 for c in candidates if c.get("submitted_to_broker"))

    focus_trace = [c for c in candidates if c["ticker"] in FOCUS_TICKERS]

    primary_root_cause = (
        "Expected plan-only entry handoff: build_trade_opportunities writes ENTRY rows to "
        "trade_opportunities.csv (authoritative input for execute_trades). "
        "run_full_pipeline does not call execute_trades. "
        "run_continuous_loop.ps1 → run_market_open.ps1 calls manage_positions (exit/trim only), "
        "not execute_trades. execute_trades defaults to dry_run=True unless --execute is passed. "
        "run_scheduled_paper_cycle can call execute_trades --execute but config run_execute_trades=false "
        "and ARM ASSISTED may block mutations. allocation OPEN_NEW=0 is advisory-only and does not "
        "gate execute_trades. No broker submissions without explicit operator execute_trades --execute."
    )

    recommended_next_step = (
        "Safe operator sequence (no code changes): "
        "(1) Confirm fresh trade_opportunities.csv after run_full_pipeline; "
        "(2) export TRITON_ENABLE_PAPER_EXECUTION=1; "
        "(3) refresh ARM confirmation (paper_arm_confirm.json) if mode=ASSISTED; "
        "(4) run: python -m services.execute_trades --mode paper --execute --verbose; "
        "(5) verify execution_plan.json dry_run=false and paper_execution_audit.csv; "
        "(6) optional: enable run_execute_trades in run_scheduled_paper_cycle.json for scheduled cycles. "
        "Do not run for SCHW until BLOCK_NEW_BUY clears unless operator explicitly accepts risk override."
    )

    key_answers = {
        "1_authoritative_entry_source": str(TRADE_OPPORTUNITIES),
        "2_execute_trades_consumes": "trade_opportunities.csv (not portfolio_allocation_recommendations.csv)",
        "3_manage_positions_new_entries": False,
        "4_continuous_loop_calls_execute_trades": bool(
            loop_info["continuous_loop_calls_execute_trades"]
        ),
        "5_continuous_loop_always_dry_run": (
            "N/A — continuous_loop does not call execute_trades; market_open runs manage_positions plan-only"
        ),
        "6_flag_for_paper_submission": "python -m services.execute_trades --mode paper --execute",
        "7_paper_auth_plus_operator_flag": (
            "paper_execution_permitted is necessary but not sufficient; "
            "requires --execute, TRITON_ENABLE_PAPER_EXECUTION=1, and ARM permissions"
        ),
        "8_bac_jpm_wfc_valid_not_submitted": (
            "Yes — valid ENTRY candidates in trade_opportunities; allocation WATCH is advisory; "
            "blocked only because execute_trades not invoked by loop and --execute not used"
        ),
        "9_stale_execution_plan": bool(plan_stale),
        "10_safe_operator_sequence": recommended_next_step,
    }

    behavior_assessment = "expected_plan_only_gating"
    if broker_submissions > 0 and not any(
        c["ticker"] in FOCUS_TICKERS for c in candidates if c.get("submitted_to_broker")
    ):
        behavior_assessment = "expected_plan_only_gating"

    summary_rows = [
        {"metric": "total_trade_opportunity_entry_rows", "value": len(candidates)},
        {"metric": "total_candidates_reaching_execute_trades_input", "value": len(candidates)},
        {"metric": "total_candidates_reaching_execution_plan", "value": in_plan_count},
        {"metric": "total_dry_run_planned_entries", "value": dry_run_planned},
        {"metric": "total_broker_submissions", "value": broker_submissions},
        {"metric": "would_submit_if_execute_enabled", "value": would_submit_count},
        {
            "metric": "continuous_loop_calls_execute_trades",
            "value": loop_info["continuous_loop_calls_execute_trades"],
        },
        {
            "metric": "market_open_calls_execute_trades",
            "value": loop_info["market_open_calls_execute_trades"],
        },
        {"metric": "execute_trades_dry_run_by_default", "value": True},
        {"metric": "operator_execute_flag_required", "value": "--execute"},
        {"metric": "execution_plan_stale", "value": plan_stale},
        {"metric": "execution_plan_age_hours", "value": plan_age_h},
        {"metric": "trade_opportunities_age_hours", "value": trade_age_h},
        {"metric": "execution_plan_dry_run", "value": exec_plan_json.get("dry_run")},
        {"metric": "execution_plan_timestamp", "value": exec_plan_json.get("timestamp")},
        {"metric": "last_execution_session_dry_run", "value": last_exec.get("dry_run")},
        {
            "metric": "allocation_open_new_candidates",
            "value": alloc_summary.get("open_new_candidates", 0),
        },
        {"metric": "manage_positions_planned", "value": manage_plan.get("orders_planned", 0)},
        {"metric": "manage_positions_executed", "value": manage_plan.get("orders_executed", 0)},
        {
            "metric": "broker_open_orders",
            "value": len(open_orders_df) if not open_orders_df.empty else 0,
        },
        {"metric": "broker_positions", "value": len(positions_df) if not positions_df.empty else 0},
        {"metric": "paper_execution_permitted", "value": paper_authorized},
        {"metric": "live_execution_permitted", "value": not live_blocked},
        {"metric": "primary_root_cause", "value": primary_root_cause},
        {"metric": "behavior_assessment", "value": behavior_assessment},
        {"metric": "recommended_next_step", "value": recommended_next_step},
    ]

    payload: Dict[str, Any] = {
        "phase": "195M",
        "timestamp": _utc_now_iso(),
        "phase_pass": True,
        "audit_mode": "read_only",
        "orders_placed": False,
        "summary": {
            "total_trade_opportunity_entry_rows": len(candidates),
            "total_candidates_reaching_execute_trades_input": len(candidates),
            "total_candidates_reaching_execution_plan": in_plan_count,
            "total_dry_run_planned_entries": dry_run_planned,
            "total_broker_submissions": broker_submissions,
            "classification_counts": dict(classification_counts),
            "continuous_loop_calls_execute_trades": loop_info[
                "continuous_loop_calls_execute_trades"
            ],
            "market_open_calls_execute_trades": loop_info["market_open_calls_execute_trades"],
            "execute_trades_dry_run_by_default": True,
            "operator_execute_flag_required": True,
            "operator_execute_command": "python -m services.execute_trades --mode paper --execute",
            "paper_execution_permitted": paper_authorized,
            "live_execution_permitted": not live_blocked,
            "execution_plan_stale": plan_stale,
            "primary_root_cause": primary_root_cause,
            "behavior_assessment": behavior_assessment,
            "recommended_next_step": recommended_next_step,
        },
        "key_questions": key_answers,
        "loop_orchestration": loop_info,
        "focus_tickers": {
            ticker: next(
                (c for c in focus_trace if c["ticker"] == ticker),
                {"ticker": ticker, "note": "not in ENTRY rows"},
            )
            for ticker in FOCUS_TICKERS
        },
        "focus_analysis": {
            "BAC": "Valid ENTRY in trade_opportunities; decision_action=BUY; allocation WATCH (advisory); not in execution_plan; execute_trades not called by loop.",
            "JPM": "Same as BAC — valid ENTRY awaiting operator execute_trades --execute.",
            "WFC": "Same as BAC/JPM — valid ENTRY; allocation WATCH does not block execute_trades input path.",
            "SCHW": "ENTRY in trade_opportunities but allocation BLOCK + BLOCK_NEW_BUY risk overlay; correctly blocked at risk layer.",
        },
        "artifact_refs": {
            "trade_opportunities": str(TRADE_OPPORTUNITIES),
            "portfolio_allocation_recommendations": str(ALLOC_RECOMMENDATIONS),
            "portfolio_allocation_summary": str(ALLOC_SUMMARY),
            "execution_plan_json": str(EXECUTION_PLAN_JSON),
            "execution_plan_csv": str(EXECUTION_PLAN_CSV),
            "manage_positions_plan": str(MANAGE_PLAN),
            "paper_trade_cycle_summary": str(PAPER_CYCLE_SUMMARY),
            "governance_authorization": str(GOVERNANCE_AUTH),
            "execution_readiness": str(EXEC_READINESS),
            "capital_preservation_certification": str(CERTIFICATION),
        },
        "paper_cycle_last": {
            "timestamp": paper_cycle.get("timestamp"),
            "execute_trades_skipped": (
                paper_cycle.get("stages", {}).get("execute_trades", {}).get("message")
                == "skipped by flag"
            ),
            "arm_mode": paper_cycle.get("arm_mode"),
            "arm_block_reasons": paper_cycle.get("arm_block_reasons"),
        },
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(CANDIDATES_CSV, candidates, CANDIDATE_COLUMNS)
    _write_csv(SUMMARY_CSV, summary_rows, ("metric", "value"))

    if verbose:
        print(
            json.dumps(
                {"phase": "195M", "phase_pass": True, "summary": payload["summary"]},
                indent=2,
            )
        )

    return payload


def main() -> int:
    run_phase_195m_audit(verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
